"""File-backed workflow orchestration for large WaterZ decoding jobs.

This module is intentionally artifact-oriented instead of thread-oriented.
Tasks are registered as small JSON manifests on a shared filesystem so the
same workflow can run in:

- serial mode for debugging
- multiple local worker processes
- multiple machines sharing one workflow directory

Workers coordinate by atomically claiming ready tasks from the manifest store.
That makes "wait for all chunk decodes to finish, then reduce offsets" a
natural workflow step without requiring one long-lived Python scheduler.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

__all__ = [
    "TaskRecord",
    "TaskSpec",
    "TaskState",
    "WorkflowOrchestrator",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_worker_id() -> str:
    return socket.gethostname() or "worker"


def _task_filename(task_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", task_id).strip("._-") or "task"
    digest = hashlib.sha1(task_id.encode("utf-8")).hexdigest()[:12]
    return f"{slug}-{digest}.json"


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


class TaskState(str, Enum):
    """Lifecycle states for one workflow task."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in {TaskState.SUCCEEDED, TaskState.FAILED}


@dataclass(frozen=True)
class TaskSpec:
    """Immutable task definition stored in the workflow manifest."""

    name: str
    stage: str
    key: str
    deps: tuple[str, ...] = ()
    payload: Dict[str, Any] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=dict)

    @property
    def task_id(self) -> str:
        return f"{self.stage}:{self.key}"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["deps"] = list(self.deps)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskSpec":
        return cls(
            name=str(data["name"]),
            stage=str(data["stage"]),
            key=str(data["key"]),
            deps=tuple(str(v) for v in data.get("deps", [])),
            payload=dict(data.get("payload", {})),
            resources=dict(data.get("resources", {})),
        )


@dataclass
class TaskRecord:
    """Mutable workflow state for one task."""

    spec: TaskSpec
    state: TaskState = TaskState.PENDING
    attempts: int = 0
    worker_id: Optional[str] = None
    job_id: Optional[str] = None
    created_at: str = field(default_factory=_utc_now)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    @property
    def task_id(self) -> str:
        return self.spec.task_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "state": self.state.value,
            "attempts": self.attempts,
            "worker_id": self.worker_id,
            "job_id": self.job_id,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskRecord":
        return cls(
            spec=TaskSpec.from_dict(data["spec"]),
            state=TaskState(str(data.get("state", TaskState.PENDING.value))),
            attempts=int(data.get("attempts", 0)),
            worker_id=data.get("worker_id"),
            job_id=data.get("job_id"),
            created_at=str(data.get("created_at", _utc_now())),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            result=dict(data["result"]) if isinstance(data.get("result"), dict) else data.get("result"),
            error=data.get("error"),
        )


class WorkflowOrchestrator:
    """Shared-filesystem workflow manifest with dependency-aware task claiming."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.tasks_dir = self.root / "tasks"
        self.locks_dir = self.root / "locks"
        self.meta_path = self.root / "workflow.json"
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.locks_dir.mkdir(parents=True, exist_ok=True)
        if not self.meta_path.exists():
            _atomic_write_json(self.meta_path, {"created_at": _utc_now(), "version": 1})

    def register(self, tasks: Sequence[TaskSpec]) -> None:
        """Register tasks if they do not already exist."""
        for spec in tasks:
            path = self._record_path(spec.task_id)
            if path.exists():
                continue
            self._write_record(TaskRecord(spec=spec))

    def list_records(self) -> list[TaskRecord]:
        records = [self._load_record(path) for path in sorted(self.tasks_dir.glob("*.json"))]
        return sorted(records, key=lambda r: (r.spec.stage, r.spec.key, r.spec.name))

    def get_record(self, task_id: str) -> TaskRecord:
        return self._load_record(self._record_path(task_id))

    def claim_ready_task(
        self,
        *,
        worker_id: Optional[str] = None,
        allowed_names: Optional[Iterable[str]] = None,
        allowed_stages: Optional[Iterable[str]] = None,
        job_id: Optional[str] = None,
    ) -> Optional[TaskRecord]:
        """Claim one ready task, returning None if nothing is runnable."""
        allowed_name_set = set(allowed_names) if allowed_names is not None else None
        allowed_stage_set = set(allowed_stages) if allowed_stages is not None else None
        worker_id = worker_id or _default_worker_id()

        for record in self.list_records():
            if record.state is not TaskState.PENDING:
                continue
            if allowed_name_set is not None and record.spec.name not in allowed_name_set:
                continue
            if allowed_stage_set is not None and record.spec.stage not in allowed_stage_set:
                continue
            if not self._deps_satisfied(record.spec.deps):
                continue
            claimed = self._claim_task(record.task_id, worker_id=worker_id, job_id=job_id)
            if claimed is not None:
                return claimed
        return None

    def complete_task(self, task_id: str, result: Optional[Dict[str, Any]] = None) -> TaskRecord:
        """Mark a running task as successful."""
        with self._task_lock(task_id):
            record = self.get_record(task_id)
            if record.state is not TaskState.RUNNING:
                raise RuntimeError(f"Cannot complete task {task_id!r} in state {record.state.value!r}.")
            record.state = TaskState.SUCCEEDED
            record.result = result
            record.error = None
            record.finished_at = _utc_now()
            self._write_record(record)
            return record

    def fail_task(self, task_id: str, error: str) -> TaskRecord:
        """Mark a running task as failed."""
        with self._task_lock(task_id):
            record = self.get_record(task_id)
            if record.state is not TaskState.RUNNING:
                raise RuntimeError(f"Cannot fail task {task_id!r} in state {record.state.value!r}.")
            record.state = TaskState.FAILED
            record.error = error
            record.finished_at = _utc_now()
            self._write_record(record)
            return record

    def reset_failed_tasks(self) -> None:
        """Move failed tasks back to pending for another orchestration pass."""
        for record in self.list_records():
            if record.state is not TaskState.FAILED:
                continue
            with self._task_lock(record.task_id):
                current = self.get_record(record.task_id)
                if current.state is not TaskState.FAILED:
                    continue
                current.state = TaskState.PENDING
                current.worker_id = None
                current.job_id = None
                current.started_at = None
                current.finished_at = None
                current.error = None
                self._write_record(current)

    def run_worker(
        self,
        handlers: Mapping[str, Callable[[TaskRecord], Optional[Dict[str, Any]]]],
        *,
        worker_id: Optional[str] = None,
        max_tasks: Optional[int] = None,
        poll_interval: float = 1.0,
        idle_timeout: Optional[float] = None,
        allowed_names: Optional[Iterable[str]] = None,
        allowed_stages: Optional[Iterable[str]] = None,
        job_id: Optional[str] = None,
    ) -> int:
        """Run tasks until the workflow is drained or the worker idles out."""
        worker_id = worker_id or _default_worker_id()
        executed = 0
        idle_started: Optional[float] = None

        while True:
            if max_tasks is not None and executed >= max_tasks:
                return executed

            record = self.claim_ready_task(
                worker_id=worker_id,
                allowed_names=allowed_names,
                allowed_stages=allowed_stages,
                job_id=job_id,
            )
            if record is None:
                if self.is_finished():
                    return executed
                if idle_timeout is not None:
                    now = time.monotonic()
                    if idle_started is None:
                        idle_started = now
                    elif (now - idle_started) >= idle_timeout:
                        return executed
                time.sleep(max(0.0, poll_interval))
                continue

            idle_started = None
            handler = handlers.get(record.spec.name)
            if handler is None:
                self.fail_task(record.task_id, f"No handler registered for task {record.spec.name!r}.")
                raise KeyError(f"No handler registered for task {record.spec.name!r}.")

            try:
                result = handler(record)
            except Exception as exc:  # pragma: no cover
                self.fail_task(record.task_id, f"{type(exc).__name__}: {exc}")
                raise

            self.complete_task(record.task_id, result=result)
            executed += 1

    def wait_for_completion(
        self,
        *,
        stage: Optional[str] = None,
        task_ids: Optional[Sequence[str]] = None,
        poll_interval: float = 1.0,
        timeout: Optional[float] = None,
        fail_fast: bool = True,
    ) -> Dict[str, TaskRecord]:
        """Block until all selected tasks finish."""
        if stage is not None and task_ids is not None:
            raise ValueError("Specify either stage or task_ids, not both.")

        selected = set(task_ids) if task_ids is not None else None
        deadline = None if timeout is None else time.monotonic() + timeout

        while True:
            records = self.list_records()
            if stage is not None:
                records = [record for record in records if record.spec.stage == stage]
            elif selected is not None:
                records = [record for record in records if record.task_id in selected]

            if fail_fast:
                failures = [record for record in records if record.state is TaskState.FAILED]
                if failures:
                    failed = failures[0]
                    raise RuntimeError(
                        f"Workflow task {failed.task_id!r} failed: {failed.error or 'unknown error'}"
                    )

            if records and all(record.state.terminal for record in records):
                return {record.task_id: record for record in records}

            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for workflow tasks to finish.")

            time.sleep(max(0.0, poll_interval))

    def is_finished(self) -> bool:
        records = self.list_records()
        return bool(records) and all(record.state.terminal for record in records)

    def stage_counts(self) -> Dict[str, Dict[str, int]]:
        """Return counts by stage and state for lightweight monitoring."""
        summary: Dict[str, Dict[str, int]] = {}
        for record in self.list_records():
            stage_summary = summary.setdefault(record.spec.stage, {})
            stage_summary[record.state.value] = stage_summary.get(record.state.value, 0) + 1
        return summary

    def _record_path(self, task_id: str) -> Path:
        return self.tasks_dir / _task_filename(task_id)

    def _write_record(self, record: TaskRecord) -> None:
        _atomic_write_json(self._record_path(record.task_id), record.to_dict())

    def _load_record(self, path: Path) -> TaskRecord:
        with path.open("r", encoding="utf-8") as handle:
            return TaskRecord.from_dict(json.load(handle))

    def _claim_task(
        self,
        task_id: str,
        *,
        worker_id: str,
        job_id: Optional[str],
    ) -> Optional[TaskRecord]:
        with self._task_lock(task_id):
            record = self.get_record(task_id)
            if record.state is not TaskState.PENDING:
                return None
            if not self._deps_satisfied(record.spec.deps):
                return None
            record.state = TaskState.RUNNING
            record.attempts += 1
            record.worker_id = worker_id
            record.job_id = job_id
            record.started_at = _utc_now()
            record.finished_at = None
            record.error = None
            self._write_record(record)
            return record

    def _deps_satisfied(self, deps: Sequence[str]) -> bool:
        for dep in deps:
            dep_record = self.get_record(dep)
            if dep_record.state is not TaskState.SUCCEEDED:
                return False
        return True

    def _lock_path(self, task_id: str) -> Path:
        return self.locks_dir / _task_filename(task_id).replace(".json", ".lock")

    class _TaskLock:
        def __init__(self, path: Path) -> None:
            self.path = path
            self.fd: Optional[int] = None

        def __enter__(self) -> "WorkflowOrchestrator._TaskLock":
            while True:
                try:
                    self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    return self
                except FileExistsError:
                    time.sleep(0.05)

        def __exit__(self, exc_type, exc, tb) -> None:
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass

    def _task_lock(self, task_id: str) -> "WorkflowOrchestrator._TaskLock":
        return self._TaskLock(self._lock_path(task_id))

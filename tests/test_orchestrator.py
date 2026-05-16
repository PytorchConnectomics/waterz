from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
import waterz as wz


def test_workflow_orchestrator_runs_tasks_in_dependency_order(tmp_path: Path) -> None:
    workflow = wz.WorkflowOrchestrator(tmp_path / "wf")
    tasks = [
        wz.TaskSpec(name="decode_chunk", stage="decode", key="a"),
        wz.TaskSpec(name="decode_chunk", stage="decode", key="b"),
        wz.TaskSpec(
            name="compute_offsets",
            stage="offsets",
            key="global",
            deps=("decode:a", "decode:b"),
        ),
    ]
    workflow.register(tasks)

    seen: list[str] = []

    def handle_decode(record: wz.TaskRecord) -> dict:
        seen.append(record.task_id)
        return {"max_id": record.spec.key}

    def handle_offsets(record: wz.TaskRecord) -> dict:
        seen.append(record.task_id)
        return {"ready": True}

    executed = workflow.run_worker(
        {
            "decode_chunk": handle_decode,
            "compute_offsets": handle_offsets,
        },
        worker_id="serial",
        poll_interval=0.01,
    )

    assert executed == 3
    assert seen[:2] == ["decode:a", "decode:b"]
    assert seen[2] == "offsets:global"
    assert workflow.get_record("offsets:global").state is wz.TaskState.SUCCEEDED


def test_wait_for_completion_observes_external_worker_updates(tmp_path: Path) -> None:
    workflow = wz.WorkflowOrchestrator(tmp_path / "wf")
    workflow.register([wz.TaskSpec(name="decode_chunk", stage="decode", key="a")])

    def external_worker() -> None:
        claimed = workflow.claim_ready_task(worker_id="remote-0")
        assert claimed is not None
        time.sleep(0.05)
        workflow.complete_task(claimed.task_id, {"max_id": 11})

    thread = threading.Thread(target=external_worker)
    thread.start()
    records = workflow.wait_for_completion(
        stage="decode", poll_interval=0.01, timeout=2.0
    )
    thread.join()

    assert list(records) == ["decode:a"]
    assert records["decode:a"].result == {"max_id": 11}


def test_list_records_retries_transient_missing_task_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = wz.WorkflowOrchestrator(tmp_path / "wf")
    workflow.register(
        [
            wz.TaskSpec(name="decode_chunk", stage="decode", key="a"),
            wz.TaskSpec(name="decode_chunk", stage="decode", key="b"),
        ]
    )

    original_load_record = workflow._load_record
    missing_path = workflow._record_path("decode:a")
    missing_once = {"value": True}

    def _load_record(path):
        if path == missing_path and missing_once["value"]:
            missing_once["value"] = False
            raise FileNotFoundError(path)
        return original_load_record(path)

    monkeypatch.setattr(workflow, "_load_record", _load_record)

    records = workflow.list_records()

    assert [record.task_id for record in records] == ["decode:a", "decode:b"]
    assert not missing_once["value"]


def test_get_record_retries_transient_missing_task_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = wz.WorkflowOrchestrator(tmp_path / "wf")
    workflow.register([wz.TaskSpec(name="decode_chunk", stage="decode", key="a")])

    original_load_record = workflow._load_record
    missing_once = {"value": True}

    def _load_record(path):
        if missing_once["value"]:
            missing_once["value"] = False
            raise FileNotFoundError(path)
        return original_load_record(path)

    monkeypatch.setattr(workflow, "_load_record", _load_record)

    record = workflow.get_record("decode:a")

    assert record.task_id == "decode:a"
    assert not missing_once["value"]


def test_build_large_decode_tasks_creates_expected_dependency_chain() -> None:
    chunks = wz.build_chunk_grid((5, 6, 7), (3, 3, 4))
    tasks = wz.build_large_decode_tasks(chunks, write_output=True)
    task_ids = {task.task_id: task for task in tasks}

    assert "offsets:global" in task_ids
    assert "relabel:global" in task_ids
    assert "assemble:h5" in task_ids
    assert any(task.stage == "connect" for task in tasks)

    for task in tasks:
        if task.stage == "connect":
            assert "offsets:global" in task.deps
        if task.stage == "apply":
            assert "relabel:global" in task.deps
        if task.stage == "assemble":
            assert all(dep.startswith("apply:") for dep in task.deps)


def test_overlap_large_decode_runs_stitch_and_rg_in_parallel() -> None:
    chunks = wz.build_chunk_grid((6, 4, 4), (3, 4, 4))
    tasks = wz.build_large_decode_tasks_overlap(chunks, write_output=True)
    task_ids = {task.task_id: task for task in tasks}

    stitch_ids = {task.task_id for task in tasks if task.stage == "stitch"}
    rg_tasks = [task for task in tasks if task.stage == "build_rg"]

    assert stitch_ids
    assert rg_tasks
    for task in rg_tasks:
        assert "offsets:global" in task.deps
        assert f"fragment:{task.key}" in task.deps
        assert not stitch_ids.intersection(task.deps)

    assert stitch_ids.issubset(set(task_ids["agglomerate:global"].deps))
    assert "merge_rg:global" in task_ids["agglomerate:global"].deps

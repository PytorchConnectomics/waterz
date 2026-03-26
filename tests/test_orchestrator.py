from __future__ import annotations

import threading
import time
from pathlib import Path

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
    records = workflow.wait_for_completion(stage="decode", poll_interval=0.01, timeout=2.0)
    thread.join()

    assert list(records) == ["decode:a"]
    assert records["decode:a"].result == {"max_id": 11}


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

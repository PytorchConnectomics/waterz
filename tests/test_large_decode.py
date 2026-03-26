from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import waterz as wz

h5py = pytest.importorskip("h5py")


def _write_affinities(path: Path, affs: np.ndarray) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("main", data=affs, dtype=affs.dtype)


def test_large_decode_runner_serial_end_to_end(tmp_path: Path) -> None:
    aff_path = tmp_path / "affinities.h5"
    out_path = tmp_path / "segmentation.h5"
    workflow_root = tmp_path / "workflow"

    affs = np.ones((3, 4, 4, 4), dtype=np.float32)
    _write_affinities(aff_path, affs)

    runner = wz.LargeDecodeRunner.create(
        affinity_path=str(aff_path),
        workflow_root=str(workflow_root),
        chunk_shape=(2, 4, 4),
        thresholds=(0.0,),
        channel_order="zyx",
        write_output=True,
        output_path=str(out_path),
        border_one_sided_threshold=0.9,
        border_affinity_threshold=0.5,
    )

    executed = runner.run_serial(poll_interval=0.01)
    assert executed == len(runner.orchestrator.list_records())
    runner.wait(timeout=5.0, poll_interval=0.01)

    with h5py.File(out_path, "r") as handle:
        seg = np.array(handle["main"], dtype=np.uint64)

    assert seg.shape == (4, 4, 4)
    assert seg.dtype == np.uint64
    assert np.all(seg == 1)

    counts = runner.orchestrator.stage_counts()
    assert counts["assemble"]["succeeded"] == 1

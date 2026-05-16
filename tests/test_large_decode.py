from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import waterz as wz

h5py = pytest.importorskip("h5py")


def _write_affinities(path: Path, affs: np.ndarray) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("main", data=affs, dtype=affs.dtype)


def _two_z_regions_affinity() -> np.ndarray:
    affs = np.ones((3, 6, 8, 8), dtype=np.float32)
    affs[0, 3, :, :] = 0.0
    return affs


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
        one_sided_threshold=0.9,
        affinity_threshold=0.5,
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


def test_large_decode_overlap_uses_waterz_3d_init(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    aff_path = tmp_path / "affinities.h5"
    out_path = tmp_path / "segmentation.h5"
    workflow_root = tmp_path / "workflow"
    _write_affinities(aff_path, _two_z_regions_affinity())

    import waterz.seg_init as seg_init

    def _fail_2d_fragments(*args, **kwargs):
        raise AssertionError("overlap large decode should not call 2D compute_fragments")

    monkeypatch.setattr(seg_init, "compute_fragments", _fail_2d_fragments)

    runner = wz.LargeDecodeRunner.create(
        affinity_path=str(aff_path),
        workflow_root=str(workflow_root),
        chunk_shape=(6, 8, 8),
        overlap=(1, 0, 0),
        thresholds=(1.0,),
        aff_threshold_low=0.3,
        aff_threshold_high=0.999,
        channel_order="zyx",
        write_output=True,
        output_path=str(out_path),
    )

    executed = runner.run_serial(poll_interval=0.01)
    assert executed == len(runner.orchestrator.list_records())

    with h5py.File(out_path, "r") as handle:
        seg = np.array(handle["main"], dtype=np.uint64)

    assert seg.shape == (6, 8, 8)
    labels = set(np.unique(seg).tolist()) - {0}
    assert len(labels) == 2
    assert np.all(seg[:3] == seg[0, 0, 0])
    assert np.all(seg[3:] == seg[3, 0, 0])
    assert seg[0, 0, 0] != seg[3, 0, 0]


def test_large_decode_overlap_can_use_2d_fragments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    aff_path = tmp_path / "affinities.h5"
    workflow_root = tmp_path / "workflow"
    _write_affinities(aff_path, np.ones((3, 4, 5, 5), dtype=np.float32))

    import waterz._agglomerate as agglomerate
    import waterz.seg_init as seg_init

    def _fail_3d_init(*args, **kwargs):
        raise AssertionError("fragment_init='2d' should not call waterz 3D init")

    called = {"2d": False}
    original_compute_fragments = seg_init.compute_fragments

    def _compute_fragments(*args, **kwargs):
        called["2d"] = True
        return original_compute_fragments(*args, **kwargs)

    monkeypatch.setattr(agglomerate, "initialize_fragments_3d", _fail_3d_init)
    monkeypatch.setattr(seg_init, "compute_fragments", _compute_fragments)

    runner = wz.LargeDecodeRunner.create(
        affinity_path=str(aff_path),
        workflow_root=str(workflow_root),
        chunk_shape=(4, 5, 5),
        overlap=(1, 0, 0),
        fragment_init="2d",
        thresholds=(1.0,),
        chunk_affinity_threshold=1.0,
        aff_threshold_low=0.3,
        channel_order="zyx",
    )

    record = runner.orchestrator.claim_ready_task(worker_id="test")
    assert record is not None
    result = runner.handle_fragment_chunk(record)

    assert called["2d"]
    assert result["fragment_init"] == "2d"
    assert result["chunk_affinity_threshold"] == 1.0

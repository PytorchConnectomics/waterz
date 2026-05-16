from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import waterz as wz

h5py = pytest.importorskip("h5py")


def _write_affinities(path: Path, affs: np.ndarray) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("main", data=affs, dtype=affs.dtype)


def _write_mask(path: Path, mask: np.ndarray) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("main", data=mask, dtype=mask.dtype)


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


def test_large_decode_overlap_uses_waterz_3d_init(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    aff_path = tmp_path / "affinities.h5"
    out_path = tmp_path / "segmentation.h5"
    workflow_root = tmp_path / "workflow"
    _write_affinities(aff_path, _two_z_regions_affinity())

    import waterz.seg_init as seg_init

    def _fail_2d_fragments(*args, **kwargs):
        raise AssertionError(
            "overlap large decode should not call 2D compute_fragments"
        )

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


def test_large_decode_overlap_can_use_2d_fragments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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


def test_affinity_mask_is_applied_before_waterz_init(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aff_path = tmp_path / "affinities.h5"
    mask_path = tmp_path / "mask.h5"
    workflow_root = tmp_path / "workflow"
    _write_affinities(aff_path, np.ones((3, 4, 5, 6), dtype=np.float32))

    mask = np.ones((4, 5, 6), dtype=np.uint8)
    mask[1, 2, 3] = 0
    _write_mask(mask_path, mask)

    import waterz._agglomerate as agglomerate

    captured = {}

    def _initialize_fragments_3d(affs, **kwargs):
        captured["masked_value"] = affs[:, 1, 2, 3].copy()
        captured["unmasked_value"] = affs[:, 1, 2, 4].copy()
        return np.ones(affs.shape[1:], dtype=np.uint64), 1

    monkeypatch.setattr(
        agglomerate, "initialize_fragments_3d", _initialize_fragments_3d
    )

    runner = wz.LargeDecodeRunner.create(
        affinity_path=str(aff_path),
        affinity_mask_path=str(mask_path),
        workflow_root=str(workflow_root),
        chunk_shape=(4, 5, 6),
        overlap=(0, 0, 0),
        fragment_init="waterz",
        thresholds=(1.0,),
        chunk_affinity_threshold=1.0,
        channel_order="zyx",
    )

    record = runner.orchestrator.claim_ready_task(worker_id="test")
    assert record is not None
    runner.handle_fragment_chunk(record)

    np.testing.assert_array_equal(
        captured["masked_value"], np.zeros(3, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        captured["unmasked_value"], np.ones(3, dtype=np.float32)
    )


def test_build_rg_converts_float16_affinities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aff_path = tmp_path / "affinities.h5"
    workflow_root = tmp_path / "workflow"
    _write_affinities(aff_path, np.ones((3, 4, 5, 6), dtype=np.float16))

    runner = wz.LargeDecodeRunner.create(
        affinity_path=str(aff_path),
        workflow_root=str(workflow_root),
        chunk_shape=(4, 5, 6),
        overlap=(1, 0, 0),
        thresholds=(1.0,),
        channel_order="zyx",
    )
    chunk = runner.chunks[0]
    ov_chunk = runner.overlap_chunk_map[chunk.key]
    ov_shape = tuple(ov_chunk.stop[i] - ov_chunk.start[i] for i in range(3))
    runner._write_chunk_seg(
        runner._raw_chunk_path(chunk.key),
        np.ones(ov_shape, dtype=np.uint64),
    )
    runner._write_json(
        runner._offsets_path(),
        {
            "chunk_offsets": {chunk.key: 0},
            "chunk_max_ids": {chunk.key: 1},
            "global_max_id": 1,
        },
    )

    import waterz._merge as merge

    captured = {}

    def _get_region_graph_rich(seg, affs, scoring_function):
        captured["dtype"] = affs.dtype
        assert seg.shape == affs.shape[1:]
        return (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.uint64),
            np.array([], dtype=np.uint64),
            np.array([], dtype=np.uint64),
        )

    monkeypatch.setattr(merge, "get_region_graph_rich", _get_region_graph_rich)

    result = runner.handle_build_rg_chunk(
        wz.TaskRecord(
            spec=wz.TaskSpec(
                name="build_rg_chunk",
                stage="build_rg",
                key=chunk.key,
            )
        )
    )

    assert captured["dtype"] == np.float32
    assert result["num_edges"] == 0


def test_overlap_stitch_writes_pairs_without_mutating_chunks(tmp_path: Path) -> None:
    aff_path = tmp_path / "affinities.h5"
    workflow_root = tmp_path / "workflow"
    _write_affinities(aff_path, np.ones((3, 5, 4, 4), dtype=np.float32))

    runner = wz.LargeDecodeRunner.create(
        affinity_path=str(aff_path),
        workflow_root=str(workflow_root),
        chunk_shape=(3, 4, 4),
        overlap=(1, 0, 0),
        thresholds=(1.0,),
        channel_order="zyx",
    )
    border = runner.borders[0]
    src_seg = np.ones((4, 4, 4), dtype=np.uint64)
    dst_seg = np.full((3, 4, 4), 2, dtype=np.uint64)
    runner._write_chunk_seg(runner._raw_chunk_path(border.src.key), src_seg)
    runner._write_chunk_seg(runner._raw_chunk_path(border.dst.key), dst_seg)
    runner._write_json(
        runner._offsets_path(),
        {
            "chunk_offsets": {border.src.key: 0, border.dst.key: 1},
            "chunk_max_ids": {border.src.key: 1, border.dst.key: 2},
            "global_max_id": 3,
        },
    )

    record = wz.TaskRecord(
        spec=wz.TaskSpec(name="stitch_overlap", stage="stitch", key=border.key)
    )
    result = runner.handle_stitch_overlap(record)

    np.testing.assert_array_equal(
        runner._read_chunk_seg(runner._raw_chunk_path(border.dst.key)), dst_seg
    )
    pairs = np.load(result["pair_path"])
    np.testing.assert_array_equal(pairs, np.array([[3, 1]], dtype=np.uint64))


def test_global_agglomerate_consumes_overlap_pairs(tmp_path: Path) -> None:
    aff_path = tmp_path / "affinities.h5"
    workflow_root = tmp_path / "workflow"
    _write_affinities(aff_path, np.ones((3, 5, 4, 4), dtype=np.float32))

    runner = wz.LargeDecodeRunner.create(
        affinity_path=str(aff_path),
        workflow_root=str(workflow_root),
        chunk_shape=(3, 4, 4),
        overlap=(1, 0, 0),
        thresholds=(1.0,),
        channel_order="zyx",
    )
    border = runner.borders[0]
    runner._write_json(
        runner._offsets_path(),
        {
            "chunk_offsets": {border.src.key: 0, border.dst.key: 1},
            "chunk_max_ids": {border.src.key: 1, border.dst.key: 2},
            "global_max_id": 3,
        },
    )
    runner._merged_rg_path().parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        runner._merged_rg_path(),
        rg_affs=np.array([], dtype=np.float32),
        id1=np.array([], dtype=np.uint64),
        id2=np.array([], dtype=np.uint64),
        contact_areas=np.array([], dtype=np.uint64),
    )
    runner._stitch_path(border.key).parent.mkdir(parents=True, exist_ok=True)
    np.save(runner._stitch_path(border.key), np.array([[3, 1]], dtype=np.uint64))

    result = runner.handle_agglomerate(
        wz.TaskRecord(
            spec=wz.TaskSpec(name="agglomerate", stage="agglomerate", key="global")
        )
    )
    mapping = np.load(result["relabel_path"])

    assert result["num_stitch_pairs"] == 1
    assert mapping[1] == mapping[3]

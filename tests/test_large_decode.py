from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import waterz as wz
from waterz.large_workflow import ChunkRef

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


def test_affinity_layout_xyz_matches_zyx(tmp_path: Path) -> None:
    """Feeding the same affinities in XYZ layout (with affinity_layout=xyz)
    must reproduce the segmentation produced from the ZYX-stored version."""
    aff_zyx_path = tmp_path / "aff_zyx.h5"
    aff_xyz_path = tmp_path / "aff_xyz.h5"
    mask_zyx_path = tmp_path / "mask_zyx.h5"
    mask_xyz_path = tmp_path / "mask_xyz.h5"
    out_zyx_path = tmp_path / "seg_zyx.h5"
    out_xyz_path = tmp_path / "seg_xyz.h5"

    # Use a non-symmetric volume so an erroneous axis swap would corrupt the
    # boundary geometry and produce a different segmentation. Anisotropic Z
    # mimics the NISB setup where Z is the thinnest physical axis.
    rng = np.random.default_rng(0)
    aff_zyx = rng.uniform(0.0, 1.0, size=(3, 4, 7, 8)).astype(np.float32)
    # Carve a low-affinity slab to encourage at least two distinct segments.
    aff_zyx[0, 2, :, :] = 0.0
    mask_zyx = np.ones((4, 7, 8), dtype=np.uint8)

    _write_affinities(aff_zyx_path, aff_zyx)
    _write_mask(mask_zyx_path, mask_zyx)

    # XYZ-stored mirrors: transpose spatial axes 0,1,2 -> 2,1,0 and channels
    # 0,1,2 -> 2,1,0 (so source ch0 means X-aff in the XYZ store, matching the
    # channel_order="xyz" reorder applied by _read_affinity_chunk).
    aff_xyz = np.transpose(aff_zyx[[2, 1, 0]], (0, 3, 2, 1))
    mask_xyz = np.transpose(mask_zyx, (2, 1, 0))
    _write_affinities(aff_xyz_path, aff_xyz)
    _write_mask(mask_xyz_path, mask_xyz)

    common_kwargs = dict(
        chunk_shape=(2, 7, 8),
        overlap=(1, 0, 0),
        thresholds=(0.5,),
        aff_threshold_low=0.3,
        aff_threshold_high=0.999,
        write_output=True,
        # Treat affinities as already destination-stored to keep the test
        # focused on axis-order plumbing rather than BANIS conversion.
        edge_offset=1,
    )

    runner_zyx = wz.LargeDecodeRunner.create(
        affinity_path=str(aff_zyx_path),
        affinity_mask_path=str(mask_zyx_path),
        workflow_root=str(tmp_path / "wf_zyx"),
        output_path=str(out_zyx_path),
        channel_order="zyx",
        affinity_layout="zyx",
        output_layout="zyx",
        **common_kwargs,
    )
    runner_zyx.run_serial(poll_interval=0.01)
    runner_zyx.wait(timeout=10.0, poll_interval=0.01)

    runner_xyz = wz.LargeDecodeRunner.create(
        affinity_path=str(aff_xyz_path),
        affinity_mask_path=str(mask_xyz_path),
        workflow_root=str(tmp_path / "wf_xyz"),
        output_path=str(out_xyz_path),
        channel_order="xyz",
        affinity_layout="xyz",
        output_layout="xyz",
        **common_kwargs,
    )
    runner_xyz.run_serial(poll_interval=0.01)
    runner_xyz.wait(timeout=10.0, poll_interval=0.01)

    with h5py.File(out_zyx_path, "r") as handle:
        seg_zyx = np.array(handle["main"], dtype=np.uint64)
    with h5py.File(out_xyz_path, "r") as handle:
        seg_xyz = np.array(handle["main"], dtype=np.uint64)

    # output_layout="xyz" writes the assembled volume in XYZ — transpose back
    # to ZYX before comparison.
    assert seg_xyz.shape == (8, 7, 4)
    seg_xyz_as_zyx = np.transpose(seg_xyz, (2, 1, 0))
    assert seg_xyz_as_zyx.shape == seg_zyx.shape

    # Segment IDs may differ — what we need is identical partitioning.
    def _normalize(seg: np.ndarray) -> np.ndarray:
        out = np.zeros_like(seg)
        for new_id, old_id in enumerate(np.unique(seg), start=0):
            out[seg == old_id] = new_id
        return out

    assert np.array_equal(_normalize(seg_zyx), _normalize(seg_xyz_as_zyx))


def test_xyz_channel_order_applies_before_source_edge_shift(tmp_path: Path) -> None:
    """NISB/BANIS source-stored affinities must reorder channels before shift."""
    aff_path = tmp_path / "aff_xyz.h5"
    workflow_root = tmp_path / "workflow"

    zyx_shape = (5, 6, 7)
    zyx_channels = np.zeros((3, *zyx_shape), dtype=np.float32)
    zz, yy, xx = np.indices(zyx_shape)
    for channel in range(3):
        zyx_channels[channel] = (channel + 1) * 1000 + zz * 100 + yy * 10 + xx

    # Store as NISB does: spatial axes XYZ and source channels X,Y,Z.
    aff_xyz = np.transpose(zyx_channels[[2, 1, 0]], (0, 3, 2, 1))
    _write_affinities(aff_path, aff_xyz)

    runner = wz.LargeDecodeRunner.create(
        affinity_path=str(aff_path),
        workflow_root=str(workflow_root),
        chunk_shape=(5, 6, 7),
        channel_order="xyz",
        affinity_layout="xyz",
        edge_offset=0,
    )

    chunk = ChunkRef(index=(0, 0, 0), start=(1, 1, 1), stop=(4, 5, 6))
    actual = runner._read_affinity_chunk(chunk)

    expected = np.zeros((3, 3, 4, 5), dtype=np.float32)
    for channel in range(3):
        slices = []
        for axis in range(3):
            if axis == channel:
                slices.append(slice(chunk.start[axis] - 1, chunk.stop[axis] - 1))
            else:
                slices.append(slice(chunk.start[axis], chunk.stop[axis]))
        expected[channel] = zyx_channels[(channel, *slices)]

    np.testing.assert_array_equal(actual, expected)


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

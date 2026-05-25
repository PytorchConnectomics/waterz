"""File-backed extra-large WaterZ decoding.

This module wires the workflow planner and orchestrator into a runnable
large-volume pipeline over HDF5 affinity tensors.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np

from ._merge import (
    get_region_graph,
    merge_function_to_scoring,
    merge_region_graphs,
    merge_segments,
)
from ._waterz import waterz as _run_waterz
from .face_merge import face_merge_pairs
from .large_workflow import (
    BorderRef,
    ChunkRef,
    build_border_adjacency,
    build_chunk_grid,
    build_chunk_grid_overlap,
    build_large_decode_tasks,
    build_large_decode_tasks_overlap,
)
from .orchestrator import TaskRecord, WorkflowOrchestrator
from .overlap_stitch import build_overlap_remap
from .region_graph import merge_id

__all__ = [
    "LargeDecodeConfig",
    "LargeDecodeRunner",
    "decode_large",
]


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-") or "artifact"


def _normalize_thresholds(
    thresholds: Union[float, Sequence[float]],
) -> tuple[float, ...]:
    if isinstance(thresholds, (int, float)):
        return (float(thresholds),)
    return tuple(sorted(float(value) for value in thresholds))


def _compression_kwargs(
    compression: Optional[str], compression_level: int
) -> Dict[str, Any]:
    if compression in (None, "", "none"):
        return {}
    kwargs: Dict[str, Any] = {"compression": compression}
    if compression == "gzip":
        kwargs["compression_opts"] = int(compression_level)
    return kwargs


def _unwrap_affinity_scoring(scoring_function: str) -> str:
    """Return raw-affinity scoring from waterz's merge-score wrapper."""
    scoring_function = str(scoring_function)
    for prefix in ("OneMinus<", "One255Minus<"):
        if scoring_function.startswith(prefix) and scoring_function.endswith(">"):
            return scoring_function[len(prefix) : -1]
    return scoring_function


def _require_h5py():
    import h5py

    return h5py


@dataclass
class LargeDecodeConfig:
    affinity_path: str
    workflow_root: str
    chunk_shape: tuple[int, int, int]
    affinity_dataset: str = "main"
    affinity_mask_path: str = ""
    affinity_mask_dataset: str = "main"
    thresholds: tuple[float, ...] = (0.3,)
    merge_function: str = "aff50_his256"
    chunk_affinity_threshold: float = 1.0
    fragment_init: str = "waterz"
    aff_threshold_low: float = 0.0001
    aff_threshold_high: float = 0.9999
    channel_order: str = "zyx"
    # Spatial axis order of the source affinity H5 (and matching mask H5).
    # "zyx" means the H5 spatial axes are (Z, Y, X) — waterz-native, no transpose.
    # "xyz" means the H5 spatial axes are (X, Y, Z); per-chunk reads transpose
    # to waterz (Z, Y, X) in memory. YAML volume_shape/chunk_shape/overlap are
    # always interpreted in waterz ZYX order regardless of this setting.
    affinity_layout: str = "zyx"
    # Spatial axis order of the written assembled.h5. "zyx" writes in waterz
    # native order. "xyz" creates the dataset with transposed shape and
    # transposes each chunk's seg from ZYX to XYZ before pasting, so downstream
    # consumers see the original H5 axis order.
    output_layout: str = "zyx"
    border_threshold: float = 0.0
    compute_fragments: bool = False
    seed_method: str = "maxima_distance"
    dust_merge: bool = True
    dust_merge_size: int = 0
    dust_merge_affinity: float = 0.0
    dust_remove_size: int = 0
    write_output: bool = False
    output_path: Optional[str] = None
    output_dataset: str = "main"
    min_overlap: int = 1
    iou_threshold: float = 0.0
    one_sided_threshold: float = 0.9
    one_sided_min_size: int = 0
    affinity_threshold: float = 0.0
    overlap: tuple[int, int, int] = (0, 0, 0)
    compression: Optional[str] = "gzip"
    compression_level: int = 4
    force_rebuild: bool = False
    volume_shape: Optional[tuple[int, int, int]] = None
    use_aff_uint8: bool = False
    use_seg_uint32: bool = False
    edge_offset: int = 1

    def __post_init__(self) -> None:
        self.affinity_path = str(Path(self.affinity_path))
        self.workflow_root = str(Path(self.workflow_root))
        if self.affinity_mask_path:
            self.affinity_mask_path = str(Path(self.affinity_mask_path))
        self.chunk_shape = tuple(int(v) for v in self.chunk_shape)
        self.overlap = tuple(int(v) for v in self.overlap)
        self.thresholds = _normalize_thresholds(self.thresholds)
        self.chunk_affinity_threshold = float(self.chunk_affinity_threshold)
        if not 0.0 <= self.chunk_affinity_threshold <= 1.0:
            raise ValueError("chunk_affinity_threshold must be in [0, 1].")
        self.fragment_init = str(self.fragment_init).lower()
        if self.fragment_init not in {"waterz", "2d"}:
            raise ValueError("fragment_init must be 'waterz' or '2d'.")
        self.channel_order = str(self.channel_order).lower()
        if self.channel_order not in {"zyx", "xyz"}:
            raise ValueError("channel_order must be 'zyx' or 'xyz'.")
        self.affinity_layout = str(self.affinity_layout).lower()
        if self.affinity_layout not in {"zyx", "xyz"}:
            raise ValueError("affinity_layout must be 'zyx' or 'xyz'.")
        self.output_layout = str(self.output_layout).lower()
        if self.output_layout not in {"zyx", "xyz"}:
            raise ValueError("output_layout must be 'zyx' or 'xyz'.")
        self.edge_offset = int(self.edge_offset)
        if self.edge_offset not in (0, 1):
            raise ValueError(
                "edge_offset must be 0 (BANIS source-stored) or 1 (waterz destination-stored)."
            )
        if self.output_path is not None:
            self.output_path = str(Path(self.output_path))
        if self.volume_shape is not None:
            self.volume_shape = tuple(int(v) for v in self.volume_shape)

    @property
    def scoring_function(self) -> str:
        return merge_function_to_scoring(self.merge_function)

    @property
    def affinity_scoring_function(self) -> str:
        return _unwrap_affinity_scoring(self.scoring_function)

    @property
    def resolved_output_path(self) -> Path:
        if self.output_path is not None:
            return Path(self.output_path)
        return Path(self.workflow_root) / "assembled.h5"

    def channel_index(self, axis: str) -> int:
        axis = str(axis).lower()
        if self.channel_order == "zyx":
            return {"z": 0, "y": 1, "x": 2}[axis]
        return {"x": 0, "y": 1, "z": 2}[axis]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["chunk_shape"] = list(self.chunk_shape)
        data["overlap"] = list(self.overlap)
        data["thresholds"] = list(self.thresholds)
        if self.volume_shape is not None:
            data["volume_shape"] = list(self.volume_shape)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LargeDecodeConfig":
        thresholds = data.get("thresholds", (0.3,))
        if isinstance(thresholds, (int, float)):
            thresholds = (float(thresholds),)
        return cls(
            affinity_path=str(data["affinity_path"]),
            workflow_root=str(data["workflow_root"]),
            chunk_shape=tuple(int(v) for v in data["chunk_shape"]),
            affinity_dataset=str(data.get("affinity_dataset", "main")),
            affinity_mask_path=str(data.get("affinity_mask_path", "")),
            affinity_mask_dataset=str(data.get("affinity_mask_dataset", "main")),
            thresholds=tuple(float(v) for v in thresholds),
            merge_function=str(data.get("merge_function", "aff50_his256")),
            chunk_affinity_threshold=float(data.get("chunk_affinity_threshold", 1.0)),
            fragment_init=str(data.get("fragment_init", "waterz")),
            aff_threshold_low=float(data.get("aff_threshold_low", 0.0001)),
            aff_threshold_high=float(data.get("aff_threshold_high", 0.9999)),
            channel_order=str(data.get("channel_order", "zyx")),
            affinity_layout=str(data.get("affinity_layout", "zyx")),
            output_layout=str(data.get("output_layout", "zyx")),
            border_threshold=float(data.get("border_threshold", 0.0)),
            compute_fragments=bool(data.get("compute_fragments", False)),
            seed_method=str(data.get("seed_method", "maxima_distance")),
            dust_merge=bool(data.get("dust_merge", True)),
            dust_merge_size=int(data.get("dust_merge_size", 0)),
            dust_merge_affinity=float(data.get("dust_merge_affinity", 0.0)),
            dust_remove_size=int(data.get("dust_remove_size", 0)),
            write_output=bool(data.get("write_output", False)),
            output_path=data.get("output_path"),
            output_dataset=str(data.get("output_dataset", "main")),
            min_overlap=int(data.get("min_overlap", 1)),
            iou_threshold=float(data.get("iou_threshold", 0.0)),
            one_sided_threshold=float(data.get("one_sided_threshold", 0.9)),
            one_sided_min_size=int(data.get("one_sided_min_size", 0)),
            affinity_threshold=float(data.get("affinity_threshold", 0.0)),
            overlap=tuple(int(v) for v in data.get("overlap", (0, 0, 0))),
            compression=data.get("compression", "gzip"),
            compression_level=int(data.get("compression_level", 4)),
            force_rebuild=bool(data.get("force_rebuild", False)),
            volume_shape=(
                tuple(int(v) for v in data["volume_shape"])
                if data.get("volume_shape") is not None
                else None
            ),
            use_aff_uint8=bool(data.get("use_aff_uint8", False)),
            use_seg_uint32=bool(data.get("use_seg_uint32", False)),
            edge_offset=int(data.get("edge_offset", 1)),
        )


class LargeDecodeRunner:
    """Runnable file-backed large-volume WaterZ decode workflow."""

    CONFIG_FILENAME = "large_decode_config.json"

    def __init__(self, config: LargeDecodeConfig) -> None:
        self.config = config
        self.root = Path(config.workflow_root)
        self.orchestrator = WorkflowOrchestrator(self.root)

    @classmethod
    def create(
        cls,
        *,
        affinity_path: str,
        workflow_root: str,
        chunk_shape: Sequence[int],
        thresholds: Union[float, Sequence[float]] = 0.3,
        merge_function: str = "aff50_his256",
        chunk_affinity_threshold: float = 1.0,
        fragment_init: str = "waterz",
        affinity_dataset: str = "main",
        affinity_mask_path: str = "",
        affinity_mask_dataset: str = "main",
        aff_threshold_low: float = 0.0001,
        aff_threshold_high: float = 0.9999,
        channel_order: str = "zyx",
        affinity_layout: str = "zyx",
        output_layout: str = "zyx",
        edge_offset: int = 1,
        write_output: bool = False,
        output_path: Optional[str] = None,
        output_dataset: str = "main",
        min_overlap: int = 1,
        iou_threshold: float = 0.0,
        one_sided_threshold: float = 0.9,
        one_sided_min_size: int = 0,
        affinity_threshold: float = 0.0,
        overlap: Sequence[int] = (0, 0, 0),
        compression: Optional[str] = "gzip",
        compression_level: int = 4,
        force_rebuild: bool = False,
    ) -> "LargeDecodeRunner":
        config = LargeDecodeConfig(
            affinity_path=str(affinity_path),
            workflow_root=str(workflow_root),
            chunk_shape=tuple(int(v) for v in chunk_shape),
            affinity_dataset=str(affinity_dataset),
            affinity_mask_path=str(affinity_mask_path),
            affinity_mask_dataset=str(affinity_mask_dataset),
            thresholds=_normalize_thresholds(thresholds),
            merge_function=str(merge_function),
            chunk_affinity_threshold=float(chunk_affinity_threshold),
            fragment_init=str(fragment_init),
            aff_threshold_low=float(aff_threshold_low),
            aff_threshold_high=float(aff_threshold_high),
            channel_order=str(channel_order),
            affinity_layout=str(affinity_layout),
            output_layout=str(output_layout),
            edge_offset=int(edge_offset),
            write_output=bool(write_output),
            output_path=str(output_path) if output_path is not None else None,
            output_dataset=str(output_dataset),
            min_overlap=int(min_overlap),
            iou_threshold=float(iou_threshold),
            one_sided_threshold=float(one_sided_threshold),
            one_sided_min_size=int(one_sided_min_size),
            affinity_threshold=float(affinity_threshold),
            overlap=tuple(int(v) for v in overlap),
            compression=compression,
            compression_level=int(compression_level),
            force_rebuild=bool(force_rebuild),
        )
        runner = cls(config)
        runner.initialize()
        return runner

    @classmethod
    def load(cls, workflow_root: Union[str, Path]) -> "LargeDecodeRunner":
        config_path = Path(workflow_root) / cls.CONFIG_FILENAME
        with config_path.open("r", encoding="utf-8") as handle:
            config = LargeDecodeConfig.from_dict(json.load(handle))
        return cls(config)

    @property
    def _use_overlap_pipeline(self) -> bool:
        return any(v > 0 for v in self.config.overlap)

    @property
    def chunks(self) -> list[ChunkRef]:
        if self.config.volume_shape is None:
            raise RuntimeError(
                "LargeDecodeRunner is not initialized with a volume shape."
            )
        return build_chunk_grid(self.config.volume_shape, self.config.chunk_shape)

    @property
    def overlap_chunks(self) -> list[ChunkRef]:
        """Chunks with overlap extensions for the overlap pipeline."""
        if self.config.volume_shape is None:
            raise RuntimeError(
                "LargeDecodeRunner is not initialized with a volume shape."
            )
        return build_chunk_grid_overlap(
            self.config.volume_shape, self.config.chunk_shape, self.config.overlap
        )

    @property
    def chunk_map(self) -> Dict[str, ChunkRef]:
        return {chunk.key: chunk for chunk in self.chunks}

    @property
    def overlap_chunk_map(self) -> Dict[str, ChunkRef]:
        return {chunk.key: chunk for chunk in self.overlap_chunks}

    @property
    def borders(self) -> list[BorderRef]:
        return build_border_adjacency(self.chunks)

    @property
    def border_map(self) -> Dict[str, BorderRef]:
        return {border.key: border for border in self.borders}

    def initialize(self) -> None:
        volume_shape = self._discover_volume_shape()
        self.config.volume_shape = volume_shape
        self.root.mkdir(parents=True, exist_ok=True)
        config_path = self._config_path()
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                existing = LargeDecodeConfig.from_dict(json.load(handle))
            if existing.to_dict() != self.config.to_dict():
                raise ValueError(
                    f"Workflow at {self.root} already exists with a different configuration."
                )
        else:
            self._write_config()
        if self._use_overlap_pipeline:
            tasks = build_large_decode_tasks_overlap(
                self.chunks,
                self.borders,
                write_output=self.config.write_output,
            )
        else:
            tasks = build_large_decode_tasks(
                self.chunks, write_output=self.config.write_output
            )
        self.orchestrator.register(tasks)

    def handlers(self) -> Dict[str, Any]:
        h = {
            "decode_chunk": self.handle_decode_chunk,
            "compute_offsets": self.handle_compute_offsets,
            "connect_border": self.handle_connect_border,
            "reduce_relabel": self.handle_reduce_relabel,
            "apply_relabel": self.handle_apply_relabel,
            "assemble_output": self.handle_assemble_output,
        }
        if self._use_overlap_pipeline:
            h.update(
                {
                    "fragment_chunk": self.handle_fragment_chunk,
                    "stitch_overlap": self.handle_stitch_overlap,
                    "build_rg_chunk": self.handle_build_rg_chunk,
                    "merge_rg": self.handle_merge_rg,
                    "agglomerate": self.handle_agglomerate,
                }
            )
        return h

    def run_serial(
        self,
        *,
        worker_id: str = "serial",
        poll_interval: float = 0.1,
    ) -> int:
        return self.orchestrator.run_worker(
            self.handlers(),
            worker_id=worker_id,
            poll_interval=poll_interval,
        )

    def run_worker(
        self,
        *,
        worker_id: Optional[str] = None,
        max_tasks: Optional[int] = None,
        poll_interval: float = 1.0,
        idle_timeout: Optional[float] = None,
        allowed_names: Optional[Sequence[str]] = None,
        allowed_stages: Optional[Sequence[str]] = None,
        job_id: Optional[str] = None,
    ) -> int:
        return self.orchestrator.run_worker(
            self.handlers(),
            worker_id=worker_id,
            max_tasks=max_tasks,
            poll_interval=poll_interval,
            idle_timeout=idle_timeout,
            allowed_names=allowed_names,
            allowed_stages=allowed_stages,
            job_id=job_id,
        )

    def wait(
        self,
        *,
        stage: Optional[str] = None,
        task_ids: Optional[Sequence[str]] = None,
        poll_interval: float = 1.0,
        timeout: Optional[float] = None,
        fail_fast: bool = True,
    ) -> Dict[str, TaskRecord]:
        return self.orchestrator.wait_for_completion(
            stage=stage,
            task_ids=task_ids,
            poll_interval=poll_interval,
            timeout=timeout,
            fail_fast=fail_fast,
        )

    def handle_decode_chunk(self, record: TaskRecord) -> Dict[str, Any]:
        os.environ.setdefault("CCACHE_DISABLE", "1")
        from ._uint8 import scale_aff_threshold, scale_thresholds

        chunk = self.chunk_map[record.spec.key]
        affs = self._read_affinity_chunk(chunk)

        is_uint8 = affs.dtype == np.uint8
        if not is_uint8:
            affs = affs.astype(np.float32, copy=False)

        # Scale float [0,1] parameters to [0,255] for uint8, matching decode_waterz
        thresholds = scale_thresholds(self.config.thresholds, is_uint8)
        aff_low, aff_high = scale_aff_threshold(
            (self.config.aff_threshold_low, self.config.aff_threshold_high),
            is_uint8,
        )

        waterz_kwargs = dict(
            thresholds=thresholds,
            aff_threshold_low=aff_low,
            aff_threshold_high=aff_high,
            scoring_function=self.config.scoring_function,
            seg_dtype="uint32" if self.config.use_seg_uint32 else "uint64",
            force_rebuild=self.config.force_rebuild,
        )
        if self.config.compute_fragments:
            waterz_kwargs["compute_fragments"] = True
            waterz_kwargs["seed_method"] = self.config.seed_method

        do_dust = self.config.dust_merge and self.config.dust_merge_size > 0
        waterz_kwargs["return_region_graph"] = do_dust
        waterz_kwargs["rescore_region_graph"] = False  # fast: use cached scores

        seg_results = _run_waterz(affs, **waterz_kwargs)
        if do_dust:
            seg, region_graph = seg_results[-1]
        else:
            seg = seg_results[-1]

        seg = np.asarray(seg, dtype=np.uint64)

        # Strip weak-boundary voxels
        if self.config.border_threshold > 0:
            xy_mean = affs[1:3].astype(np.float32, copy=False).mean(axis=0)
            if is_uint8:
                xy_mean /= 255.0
            seg[xy_mean < self.config.border_threshold] = 0

        # Dust merge using agglomeration's region graph (cached scores, no re-scoring)
        if do_dust:
            from ._merge import dust_merge_from_region_graph

            dust_merge_from_region_graph(
                seg,
                region_graph,
                is_uint8=is_uint8,
                size_th=self.config.dust_merge_size,
                weight_th=self.config.dust_merge_affinity,
                dust_th=self.config.dust_remove_size,
            )

        path = self._raw_chunk_path(chunk.key)
        self._write_chunk_seg(path, seg)

        # Pre-extract boundary faces as .npy so connect_border avoids HDF5 re-reads
        for axis, dim, side_idx in [("z", 0, -1), ("y", 1, -1), ("x", 2, -1)]:
            for side, idx in [("src", side_idx), ("dst", 0)]:
                fp = self._face_path(chunk.key, axis, side)
                fp.parent.mkdir(parents=True, exist_ok=True)
                if dim == 0:
                    face = seg[idx, :, :]
                elif dim == 1:
                    face = seg[:, idx, :]
                else:
                    face = seg[:, :, idx]
                np.save(fp, face)

        return {"chunk_path": str(path), "max_id": int(seg.max())}

    def handle_compute_offsets(self, record: TaskRecord) -> Dict[str, Any]:
        cursor = 0
        chunk_offsets: Dict[str, int] = {}
        chunk_max_ids: Dict[str, int] = {}
        for chunk in self.chunks:
            max_id = self._read_chunk_max(self._raw_chunk_path(chunk.key))
            chunk_offsets[chunk.key] = cursor
            chunk_max_ids[chunk.key] = max_id
            cursor += max_id

        data = {
            "chunk_offsets": chunk_offsets,
            "chunk_max_ids": chunk_max_ids,
            "global_max_id": cursor,
        }
        self._write_json(self._offsets_path(), data)
        return data

    def handle_connect_border(self, record: TaskRecord) -> Dict[str, Any]:
        border = self.border_map[record.spec.key]
        offsets = self._read_json(self._offsets_path())

        # Read pre-extracted .npy faces (fast), fall back to HDF5
        src_npy = self._face_path(border.src.key, border.axis, "src")
        dst_npy = self._face_path(border.dst.key, border.axis, "dst")
        if src_npy.exists() and dst_npy.exists():
            src_face = np.load(src_npy).astype(np.uint64, copy=False)
            dst_face = np.load(dst_npy).astype(np.uint64, copy=False)
        else:
            src_face = self._read_chunk_face(
                self._raw_chunk_path(border.src.key), border.axis, side="src"
            )
            dst_face = self._read_chunk_face(
                self._raw_chunk_path(border.dst.key), border.axis, side="dst"
            )

        src_offset = int(offsets["chunk_offsets"][border.src.key])
        dst_offset = int(offsets["chunk_offsets"][border.dst.key])
        if src_offset:
            src_mask = src_face > 0
            src_face[src_mask] += src_offset
        if dst_offset:
            dst_mask = dst_face > 0
            dst_face[dst_mask] += dst_offset

        aff = self._read_boundary_affinity(border)
        pairs = face_merge_pairs(
            src_face,
            dst_face,
            aff,
            min_overlap=self.config.min_overlap,
            iou_threshold=float(self.config.iou_threshold),
            one_sided_threshold=float(self.config.one_sided_threshold),
            one_sided_min_size=int(self.config.one_sided_min_size),
            affinity_threshold=float(self.config.affinity_threshold),
        )
        path = self._connect_path(border.key)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, pairs)
        return {"pair_path": str(path), "num_pairs": int(len(pairs))}

    def handle_reduce_relabel(self, record: TaskRecord) -> Dict[str, Any]:
        offsets = self._read_json(self._offsets_path())
        global_max_id = int(offsets["global_max_id"])
        pair_arrays = []
        for border in self.borders:
            path = self._connect_path(border.key)
            if path.exists():
                arr = np.load(path)
                if arr.size > 0:
                    pair_arrays.append(arr.astype(np.uint64, copy=False))

        mapping = self._build_relabel_array(global_max_id, pair_arrays)
        relabel_path = self._relabel_path()
        relabel_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(relabel_path, mapping)
        return {
            "relabel_path": str(relabel_path),
            "global_max_id": global_max_id,
            "final_max_id": int(mapping.max()) if mapping.size else 0,
        }

    def handle_apply_relabel(self, record: TaskRecord) -> Dict[str, Any]:
        chunk = self.chunk_map[record.spec.key]
        offsets = self._read_json(self._offsets_path())
        relabel = np.load(self._relabel_path())
        seg = self._read_chunk_seg(self._raw_chunk_path(chunk.key))
        offset = int(offsets["chunk_offsets"][chunk.key])
        if offset:
            mask = seg > 0
            seg[mask] += offset
        seg = relabel[seg]
        if self._use_overlap_pipeline:
            ov_chunk = self.overlap_chunk_map[chunk.key]
            base_local = tuple(
                slice(
                    chunk.start[i] - ov_chunk.start[i],
                    chunk.stop[i] - ov_chunk.start[i],
                )
                for i in range(3)
            )
            seg = seg[base_local]
        path = self._final_chunk_path(chunk.key)
        self._write_chunk_seg(path, seg)
        return {"chunk_path": str(path), "max_id": int(seg.max())}

    def handle_assemble_output(self, record: TaskRecord) -> Dict[str, Any]:
        output_path = self.config.resolved_output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        h5py = _require_h5py()
        waterz_shape = tuple(int(v) for v in self.config.volume_shape or ())
        waterz_chunk_shape = tuple(
            min(waterz_shape[i], self.config.chunk_shape[i]) for i in range(3)
        )
        out_layout = self.config.output_layout
        if out_layout == "xyz":
            # ZYX -> XYZ for both the dataset shape and the chunk hints.
            out_shape = (waterz_shape[2], waterz_shape[1], waterz_shape[0])
            out_chunks = (
                waterz_chunk_shape[2],
                waterz_chunk_shape[1],
                waterz_chunk_shape[0],
            )
        else:
            out_shape = waterz_shape
            out_chunks = waterz_chunk_shape
        kwargs = _compression_kwargs(
            self.config.compression, self.config.compression_level
        )
        with h5py.File(output_path, "w") as handle:
            dataset = handle.create_dataset(
                self.config.output_dataset,
                shape=out_shape,
                dtype=np.uint64,
                chunks=out_chunks,
                **kwargs,
            )
            for chunk in self.chunks:
                seg = self._read_chunk_seg(self._final_chunk_path(chunk.key))
                if out_layout == "xyz":
                    seg_xyz = np.transpose(seg, (2, 1, 0))
                    dataset[
                        chunk.start[2] : chunk.stop[2],
                        chunk.start[1] : chunk.stop[1],
                        chunk.start[0] : chunk.stop[0],
                    ] = seg_xyz
                else:
                    dataset[
                        chunk.start[0] : chunk.stop[0],
                        chunk.start[1] : chunk.stop[1],
                        chunk.start[2] : chunk.stop[2],
                    ] = seg
        return {"output_path": str(output_path)}

    # ------------------------------------------------------------------
    # Overlap pipeline handlers
    # ------------------------------------------------------------------

    def handle_fragment_chunk(self, record: TaskRecord) -> Dict[str, Any]:
        """Initialize fragments per chunk, then optional strong-edge premerge."""
        os.environ.setdefault("CCACHE_DISABLE", "1")

        chunk_key = record.spec.key
        ov_chunk = self.overlap_chunk_map[chunk_key]
        affs = self._read_affinity_chunk(ov_chunk)

        is_uint8 = affs.dtype == np.uint8
        if not is_uint8:
            affs = affs.astype(np.float32, copy=False)

        if self.config.fragment_init == "2d":
            from .seg_init import compute_fragments

            seg = compute_fragments(
                affs,
                seed_method=self.config.seed_method,
                aff_threshold_low=self.config.aff_threshold_low,
            )
            n_fragments = int(seg.max()) if seg.size else 0
        else:
            from ._agglomerate import initialize_fragments_3d
            from ._uint8 import scale_aff_threshold

            aff_low, aff_high = scale_aff_threshold(
                (self.config.aff_threshold_low, self.config.aff_threshold_high),
                is_uint8,
            )
            seg, n_fragments = initialize_fragments_3d(
                affs,
                aff_threshold_low=aff_low,
                aff_threshold_high=aff_high,
                seg_dtype="uint32" if self.config.use_seg_uint32 else "uint64",
                force_rebuild=self.config.force_rebuild,
            )
        seg = np.asarray(seg, dtype=np.uint64)
        premerge_result = self._premerge_chunk_fragments(seg, affs)
        path = self._raw_chunk_path(chunk_key)
        self._write_chunk_seg(path, seg)
        return {
            "chunk_path": str(path),
            "max_id": int(seg.max()) if seg.size else 0,
            "fragment_init": self.config.fragment_init,
            "initial_fragments": int(n_fragments),
            **premerge_result,
        }

    def _premerge_chunk_fragments(
        self,
        seg: np.ndarray,
        affs: np.ndarray,
    ) -> Dict[str, Any]:
        """Conservatively merge obvious intra-chunk fragment edges."""
        threshold = float(self.config.chunk_affinity_threshold)
        if threshold >= 1.0:
            return {
                "chunk_affinity_threshold": threshold,
                "premerge_edges": 0,
                "premerge_segments": int(seg.max()) if seg.size else 0,
            }

        rg_affs, id1, id2 = get_region_graph(
            seg,
            affs,
            scoring_function=self.config.affinity_scoring_function,
        )
        if len(rg_affs) == 0:
            return {
                "chunk_affinity_threshold": threshold,
                "premerge_edges": 0,
                "premerge_segments": int(seg.max()) if seg.size else 0,
            }

        ids, cnts = np.unique(seg, return_counts=True)
        max_id = int(ids.max()) if len(ids) else 0
        counts = np.zeros(max_id + 1, dtype=np.uint64)
        counts[ids] = cnts.astype(np.uint64, copy=False)

        # size_th greater than the chunk volume makes merge_segments behave as
        # a pure affinity-threshold merge while preserving its compact relabel.
        n_segments = merge_segments(
            seg,
            rg_affs,
            id1,
            id2,
            counts,
            size_th=int(seg.size) + 1,
            weight_th=threshold,
            dust_th=0,
        )
        return {
            "chunk_affinity_threshold": threshold,
            "premerge_edges": int(np.count_nonzero(rg_affs > threshold)),
            "premerge_segments": int(n_segments),
        }

    def handle_stitch_overlap(self, record: TaskRecord) -> Dict[str, Any]:
        """Write global ID equivalence pairs for one overlap zone."""
        border_key = record.spec.key
        border = self.border_map[border_key]
        src_ov = self.overlap_chunk_map[border.src.key]
        dst_ov = self.overlap_chunk_map[border.dst.key]

        src_seg = self._read_chunk_seg(self._raw_chunk_path(border.src.key))
        dst_seg = self._read_chunk_seg(self._raw_chunk_path(border.dst.key))

        # Compute overlap region in volume coordinates
        ovl_start = tuple(max(src_ov.start[i], dst_ov.start[i]) for i in range(3))
        ovl_stop = tuple(min(src_ov.stop[i], dst_ov.stop[i]) for i in range(3))

        # Convert to local chunk coordinates
        src_local = tuple(
            slice(ovl_start[i] - src_ov.start[i], ovl_stop[i] - src_ov.start[i])
            for i in range(3)
        )
        dst_local = tuple(
            slice(ovl_start[i] - dst_ov.start[i], ovl_stop[i] - dst_ov.start[i])
            for i in range(3)
        )

        overlap_src = src_seg[src_local]
        overlap_dst = dst_seg[dst_local]

        offsets = self._read_json(self._offsets_path())
        src_offset = int(offsets["chunk_offsets"][border.src.key])
        dst_offset = int(offsets["chunk_offsets"][border.dst.key])
        if src_offset:
            overlap_src = overlap_src.copy()
            src_mask = overlap_src > 0
            overlap_src[src_mask] += src_offset
        if dst_offset:
            overlap_dst = overlap_dst.copy()
            dst_mask = overlap_dst > 0
            overlap_dst[dst_mask] += dst_offset

        remap = build_overlap_remap(overlap_src, overlap_dst)
        if remap:
            pairs = np.asarray(list(remap.items()), dtype=np.uint64)
        else:
            pairs = np.empty((0, 2), dtype=np.uint64)

        path = self._stitch_path(border_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, pairs)
        return {
            "border_key": border_key,
            "pair_path": str(path),
            "num_pairs": len(pairs),
        }

    def handle_build_rg_chunk(self, record: TaskRecord) -> Dict[str, Any]:
        """Build scored region graph with contact areas for one chunk."""
        os.environ.setdefault("CCACHE_DISABLE", "1")
        chunk_key = record.spec.key
        ov_chunk = self.overlap_chunk_map[chunk_key]
        affs = self._read_affinity_chunk(ov_chunk)
        if affs.dtype != np.uint8:
            affs = affs.astype(np.float32, copy=False)
        seg = self._read_chunk_seg(self._raw_chunk_path(chunk_key))

        # Apply global offset so IDs are unique across chunks
        offsets = self._read_json(self._offsets_path())
        offset = int(offsets["chunk_offsets"][chunk_key])
        if offset:
            mask = seg > 0
            seg[mask] += offset

        from ._merge import get_region_graph_rich

        rg_affs, id1, id2, contact_areas = get_region_graph_rich(
            seg,
            affs,
            scoring_function=self.config.affinity_scoring_function,
        )

        path = self._rg_chunk_path(chunk_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, rg_affs=rg_affs, id1=id1, id2=id2, contact_areas=contact_areas)
        return {"rg_path": str(path), "num_edges": len(rg_affs)}

    def handle_merge_rg(self, record: TaskRecord) -> Dict[str, Any]:
        """Merge per-chunk region graphs into a single global region graph."""
        rg_list = []
        for chunk in self.chunks:
            path = self._rg_chunk_path(chunk.key)
            data = np.load(path)
            rg_list.append(
                (
                    data["rg_affs"],
                    data["id1"],
                    data["id2"],
                    data["contact_areas"],
                )
            )

        merged_affs, merged_id1, merged_id2, merged_areas = merge_region_graphs(rg_list)

        path = self._merged_rg_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            rg_affs=merged_affs,
            id1=merged_id1,
            id2=merged_id2,
            contact_areas=merged_areas,
        )
        return {"merged_rg_path": str(path), "num_edges": len(merged_affs)}

    def handle_agglomerate(self, record: TaskRecord) -> Dict[str, Any]:
        """Threshold merge on the global region graph plus overlap pairs."""
        data = np.load(self._merged_rg_path())
        rg_affs = data["rg_affs"]
        id1 = data["id1"]
        id2 = data["id2"]

        offsets = self._read_json(self._offsets_path())
        global_max_id = int(offsets["global_max_id"])

        # Use the highest threshold
        threshold = max(self.config.thresholds)

        merge_id1: list[np.ndarray] = []
        merge_id2: list[np.ndarray] = []

        # Merge region-graph edges with affinity >= threshold.
        qualify = rg_affs >= threshold
        if qualify.any():
            merge_id1.append(id1[qualify])
            merge_id2.append(id2[qualify])

        n_stitch_pairs = 0
        for border in self.borders:
            path = self._stitch_path(border.key)
            if not path.exists():
                continue
            pairs = np.load(path).astype(np.uint64, copy=False)
            if pairs.size == 0:
                continue
            pairs = pairs.reshape(-1, 2)
            n_stitch_pairs += int(len(pairs))
            merge_id1.append(pairs[:, 0])
            merge_id2.append(pairs[:, 1])

        if merge_id1:
            sentinel = np.array([global_max_id], dtype=np.uint64)
            all_id1 = np.concatenate([*merge_id1, sentinel])
            all_id2 = np.concatenate([*merge_id2, sentinel])
            roots = merge_id(all_id1, all_id2)
            if len(roots) < global_max_id + 1:
                padded = np.arange(global_max_id + 1, dtype=np.uint64)
                padded[: len(roots)] = roots
                roots = padded
        else:
            roots = np.arange(global_max_id + 1, dtype=np.uint64)

        # Compact the mapping
        mapping = np.zeros(global_max_id + 1, dtype=np.uint64)
        _, inverse = np.unique(roots[1:], return_inverse=True)
        mapping[1:] = inverse.astype(np.uint64) + 1

        relabel_path = self._relabel_path()
        relabel_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(relabel_path, mapping)
        return {
            "relabel_path": str(relabel_path),
            "global_max_id": global_max_id,
            "final_max_id": int(mapping.max()) if mapping.size else 0,
            "num_rg_edges": int(np.count_nonzero(qualify)),
            "num_stitch_pairs": n_stitch_pairs,
        }

    def _discover_volume_shape(self) -> tuple[int, int, int]:
        h5py = _require_h5py()
        with h5py.File(self.config.affinity_path, "r") as handle:
            dataset = handle[self.config.affinity_dataset]
            if dataset.ndim != 4 or dataset.shape[0] < 3:
                raise ValueError(
                    f"Expected affinity dataset with shape (C>=3, Z, Y, X), got {dataset.shape}."
                )
            h5_spatial = tuple(int(v) for v in dataset.shape[1:])
        # Internal volume_shape is always in waterz (Z, Y, X) order. For an
        # XYZ-stored H5 the spatial dims are (X, Y, Z) — swap to (Z, Y, X).
        if self.config.affinity_layout == "xyz":
            return (h5_spatial[2], h5_spatial[1], h5_spatial[0])
        return h5_spatial

    def _config_path(self) -> Path:
        return self.root / self.CONFIG_FILENAME

    def _write_config(self) -> None:
        self._write_json(self._config_path(), self.config.to_dict())

    def _raw_chunk_path(self, chunk_key: str) -> Path:
        return self.root / "chunks" / "raw" / f"{chunk_key}.h5"

    def _final_chunk_path(self, chunk_key: str) -> Path:
        return self.root / "chunks" / "final" / f"{chunk_key}.h5"

    def _face_path(self, chunk_key: str, axis: str, side: str) -> Path:
        return self.root / "chunks" / "faces" / f"{chunk_key}_{axis}_{side}.npy"

    def _connect_path(self, border_key: str) -> Path:
        return self.root / "connect" / f"{_safe_name(border_key)}.npy"

    def _stitch_path(self, border_key: str) -> Path:
        return self.root / "stitch" / f"{_safe_name(border_key)}.npy"

    def _offsets_path(self) -> Path:
        return self.root / "artifacts" / "offsets.json"

    def _relabel_path(self) -> Path:
        return self.root / "artifacts" / "relabel.npy"

    def _rg_chunk_path(self, chunk_key: str) -> Path:
        return self.root / "rg" / f"{chunk_key}.npz"

    def _merged_rg_path(self) -> Path:
        return self.root / "artifacts" / "merged_rg.npz"

    def _write_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _read_json(self, path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return dict(json.load(handle))

    # ------------------------------------------------------------------
    # Axis-order helpers (affinity_layout)
    # ------------------------------------------------------------------
    #
    # waterz internally interprets array axes as (Z, Y, X); chunk.start/stop
    # and YAML volume_shape/chunk_shape/overlap are also in this waterz ZYX
    # order regardless of how the H5 is stored. The helpers below convert
    # waterz coordinates into the H5's physical spatial axes when the H5 is
    # XYZ-stored (e.g. raw_x1_ch0-1-2.h5 from BANIS, shape (C, X, Y, Z)).

    def _waterz_to_h5_indices(
        self, start: Sequence[int], stop: Sequence[int]
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Map a waterz (Z, Y, X) start/stop pair into H5 physical indices.

        For ``affinity_layout="xyz"`` the H5 stores (X, Y, Z), so H5 axis 0
        sees waterz X (start[2]), axis 1 sees Y (start[1]), axis 2 sees Z
        (start[0]).
        """
        s = tuple(int(v) for v in start)
        e = tuple(int(v) for v in stop)
        if self.config.affinity_layout == "zyx":
            return s, e
        return (s[2], s[1], s[0]), (e[2], e[1], e[0])

    def _h5_to_waterz_affs(self, arr: np.ndarray) -> np.ndarray:
        """Transpose a 4D (C, h5_ax0, h5_ax1, h5_ax2) array to (C, Z, Y, X)."""
        if self.config.affinity_layout == "zyx":
            return arr
        # H5 (C, X, Y, Z) -> waterz (C, Z, Y, X): spatial axes 0,1,2 -> 2,1,0
        return np.transpose(arr, (0, 3, 2, 1))

    def _h5_to_waterz_mask(self, mask: np.ndarray) -> np.ndarray:
        """Transpose a 3D H5 mask block into waterz (Z, Y, X) order."""
        if self.config.affinity_layout == "zyx":
            return mask
        return np.transpose(mask, (2, 1, 0))

    def _channels_to_waterz_order(self, affs: np.ndarray) -> np.ndarray:
        """Return affinities with channels ordered as waterz (Z, Y, X)."""
        if self.config.channel_order == "xyz":
            return affs[[2, 1, 0]]
        return affs

    def _waterz_to_h5_axis(self, waterz_axis: int) -> int:
        """Which H5 spatial axis carries data for a given waterz logical axis."""
        if self.config.affinity_layout == "zyx":
            return int(waterz_axis)
        return (2, 1, 0)[int(waterz_axis)]

    def _read_affinity_chunk(self, chunk: ChunkRef) -> np.ndarray:
        h5py = _require_h5py()
        edge_offset = int(self.config.edge_offset)
        chunk_dims = tuple(int(chunk.stop[i] - chunk.start[i]) for i in range(3))
        with h5py.File(self.config.affinity_path, "r") as handle:
            dataset = handle[self.config.affinity_dataset]
            if edge_offset == 1:
                h5_start, h5_stop = self._waterz_to_h5_indices(chunk.start, chunk.stop)
                affs = np.array(
                    dataset[
                        0:3,
                        h5_start[0] : h5_stop[0],
                        h5_start[1] : h5_stop[1],
                        h5_start[2] : h5_stop[2],
                    ],
                )
                affs = self._h5_to_waterz_affs(affs)
                self._apply_affinity_mask(
                    affs,
                    start=chunk.start,
                    stop=chunk.stop,
                )
                affs = self._channels_to_waterz_order(affs)
            else:
                # BANIS source-stored -> waterz destination-stored: for each
                # channel c, dst[c, v] = src[c, v - 1] along spatial axis c
                # (zero at global v=0). Read one extra voxel on the low side
                # per *waterz* axis when available, then per-channel shift.
                # The extra voxels are then translated into H5 physical axes.
                extra = tuple(1 if chunk.start[i] > 0 else 0 for i in range(3))
                wz_start_ext = tuple(chunk.start[i] - extra[i] for i in range(3))
                h5_start_ext, h5_stop = self._waterz_to_h5_indices(
                    wz_start_ext, chunk.stop
                )
                ext_affs = np.array(
                    dataset[
                        0:3,
                        h5_start_ext[0] : h5_stop[0],
                        h5_start_ext[1] : h5_stop[1],
                        h5_start_ext[2] : h5_stop[2],
                    ],
                )
                ext_affs = self._h5_to_waterz_affs(ext_affs)
                self._apply_affinity_mask(
                    ext_affs,
                    start=wz_start_ext,
                    stop=chunk.stop,
                )
                ext_affs = self._channels_to_waterz_order(ext_affs)
                affs = np.zeros((3, *chunk_dims), dtype=ext_affs.dtype)
                for c in range(3):
                    # Slice channel c, with axis c+1 picking the shifted range
                    # (when extra[c]==1) and other axes trimming off the extra
                    # voxels from the low side so the result is chunk_dims.
                    slicer = [c]
                    for i in range(3):
                        if i == c and extra[c] == 1:
                            slicer.append(slice(0, chunk_dims[i]))
                        else:
                            slicer.append(slice(extra[i], extra[i] + chunk_dims[i]))
                    chan_3d = ext_affs[tuple(slicer)]
                    if extra[c] == 0:
                        # Global low boundary along axis c: roll +1, zero v=0.
                        chan_3d = np.roll(chan_3d, shift=1, axis=c)
                        zero_slicer = [slice(None)] * 3
                        zero_slicer[c] = 0
                        chan_3d[tuple(zero_slicer)] = 0
                    affs[c] = chan_3d
        return np.ascontiguousarray(affs)

    def _read_mask_dataset(self):
        h5py = _require_h5py()
        mask_path = getattr(self.config, "affinity_mask_path", "") or ""
        if not mask_path:
            return None, None
        handle = h5py.File(mask_path, "r")
        dataset_name = getattr(self.config, "affinity_mask_dataset", "main") or "main"
        try:
            if dataset_name in handle:
                dataset = handle[dataset_name]
            elif "main" in handle:
                dataset = handle["main"]
            else:
                keys = list(handle.keys())
                if len(keys) != 1:
                    raise ValueError(
                        f"affinity_mask_path={mask_path}: expected dataset {dataset_name!r}, "
                        f"'main', or a single dataset, got {keys}"
                    )
                dataset = handle[keys[0]]
            return handle, dataset
        except Exception:
            handle.close()
            raise

    def _apply_affinity_mask(
        self,
        affs: np.ndarray,
        *,
        start: Sequence[int],
        stop: Sequence[int],
    ) -> None:
        """Apply mask to ``affs`` (already in waterz ZYX). ``start``/``stop`` are waterz."""
        if not (getattr(self.config, "affinity_mask_path", "") or ""):
            return
        mask_handle, mask_dataset = self._read_mask_dataset()
        if mask_handle is None or mask_dataset is None:
            return
        try:
            # volume_shape is in waterz ZYX. The mask H5 follows affinity_layout.
            expected_waterz_shape = tuple(
                int(v) for v in (self.config.volume_shape or ())
            )
            expected_h5_shape: tuple[int, ...]
            if expected_waterz_shape:
                if self.config.affinity_layout == "zyx":
                    expected_h5_shape = expected_waterz_shape
                else:
                    expected_h5_shape = (
                        expected_waterz_shape[2],
                        expected_waterz_shape[1],
                        expected_waterz_shape[0],
                    )
                if tuple(mask_dataset.shape) != expected_h5_shape:
                    raise ValueError(
                        f"affinity mask shape {tuple(mask_dataset.shape)} != expected "
                        f"H5 shape {expected_h5_shape} (volume_shape={expected_waterz_shape} "
                        f"in waterz ZYX, affinity_layout={self.config.affinity_layout})"
                    )
            h5_start, h5_stop = self._waterz_to_h5_indices(start, stop)
            mask_block = np.array(
                mask_dataset[
                    h5_start[0] : h5_stop[0],
                    h5_start[1] : h5_stop[1],
                    h5_start[2] : h5_stop[2],
                ],
            )
        finally:
            mask_handle.close()

        mask = self._h5_to_waterz_mask(mask_block)
        if mask.shape != affs.shape[1:]:
            raise ValueError(
                f"affinity mask chunk shape {mask.shape} != affinity chunk spatial shape "
                f"{affs.shape[1:]}"
            )
        zero_idx = ~mask if mask.dtype == np.bool_ else (mask == 0)
        if zero_idx.any():
            affs[:, zero_idx] = 0

    def _read_boundary_affinity(self, border: BorderRef) -> np.ndarray:
        h5py = _require_h5py()
        channel_index = self.config.channel_index(border.axis)
        # Source-stored BANIS affinities: the edge between the src chunk's
        # last voxel and the dst chunk's first voxel is at the SRC voxel
        # (dst.start - 1). Destination-stored waterz native: at dst.start.
        edge_offset = int(self.config.edge_offset)
        waterz_axis_index = {"z": 0, "y": 1, "x": 2}[border.axis]
        boundary_pos = int(border.dst.start[waterz_axis_index]) - (
            1 if edge_offset == 0 else 0
        )

        # Build the H5 slicer (after the channel axis) for the 2D boundary face,
        # using waterz->H5 axis mapping. The boundary is at ``boundary_pos`` on
        # the H5 axis that corresponds to the waterz boundary axis; the other
        # two H5 axes get range slices derived from src.start/stop after
        # remapping.
        h5_boundary_axis = self._waterz_to_h5_axis(waterz_axis_index)
        h5_slicer: list = [None, None, None]
        h5_slicer[h5_boundary_axis] = int(boundary_pos)
        for waterz_ax in range(3):
            if waterz_ax == waterz_axis_index:
                continue
            h5_ax = self._waterz_to_h5_axis(waterz_ax)
            h5_slicer[h5_ax] = slice(
                int(border.src.start[waterz_ax]),
                int(border.src.stop[waterz_ax]),
            )

        with h5py.File(self.config.affinity_path, "r") as handle:
            dataset = handle[self.config.affinity_dataset]
            data = dataset[(channel_index, *h5_slicer)]
            arr = np.array(data)
            arr = self._h5_face_to_waterz(arr, waterz_axis_index)
            mask_handle, mask_dataset = self._read_mask_dataset()
            if mask_handle is not None and mask_dataset is not None:
                try:
                    mask_data = np.array(mask_dataset[tuple(h5_slicer)])
                    mask_data = self._h5_face_to_waterz(mask_data, waterz_axis_index)
                    zero_idx = (
                        ~mask_data if mask_data.dtype == np.bool_ else (mask_data == 0)
                    )
                    if np.asarray(zero_idx).any():
                        arr[zero_idx] = 0
                finally:
                    mask_handle.close()
            if arr.dtype == np.uint8:
                return arr.astype(np.float32) / 255.0
            return np.asarray(arr, dtype=np.float32)

    def _h5_face_to_waterz(
        self, face: np.ndarray, waterz_axis_index: int
    ) -> np.ndarray:
        """Transpose a 2D H5 boundary face into waterz order.

        The face has axes ``(h5_axis_a, h5_axis_b)`` where ``a, b`` are the two
        H5 spatial axes that are NOT the boundary axis, in increasing numeric
        order. waterz expects axes ``(other_waterz_axis_0, other_waterz_axis_1)``
        in increasing waterz-axis order. For ``affinity_layout="zyx"`` these
        coincide; for ``"xyz"`` the two remaining axes are reversed, so the
        face needs a single transpose.
        """
        if self.config.affinity_layout == "zyx":
            return face
        return np.ascontiguousarray(face.T)

    def _write_chunk_seg(
        self, path: Path, seg: np.ndarray, *, compress: bool = False
    ) -> None:
        h5py = _require_h5py()
        path.parent.mkdir(parents=True, exist_ok=True)
        kwargs = (
            _compression_kwargs(self.config.compression, self.config.compression_level)
            if compress
            else {}
        )
        tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        try:
            with h5py.File(tmp_path, "w") as handle:
                handle.create_dataset("main", data=seg, dtype=seg.dtype, **kwargs)
                handle.create_dataset(
                    "max", data=np.asarray(int(seg.max()), dtype=np.uint64)
                )
            os.replace(tmp_path, path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    def _read_chunk_seg(self, path: Path) -> np.ndarray:
        h5py = _require_h5py()
        with h5py.File(path, "r") as handle:
            return np.array(handle["main"], dtype=np.uint64)

    def _read_chunk_max(self, path: Path) -> int:
        h5py = _require_h5py()
        with h5py.File(path, "r") as handle:
            return int(np.array(handle["max"]))

    def _read_chunk_face(self, path: Path, axis: str, *, side: str) -> np.ndarray:
        h5py = _require_h5py()
        with h5py.File(path, "r") as handle:
            dataset = handle["main"]
            if axis == "z":
                index = dataset.shape[0] - 1 if side == "src" else 0
                return np.array(dataset[index, :, :], dtype=np.uint64)
            if axis == "y":
                index = dataset.shape[1] - 1 if side == "src" else 0
                return np.array(dataset[:, index, :], dtype=np.uint64)
            index = dataset.shape[2] - 1 if side == "src" else 0
            return np.array(dataset[:, :, index], dtype=np.uint64)

    def _build_relabel_array(
        self,
        global_max_id: int,
        pair_arrays: Sequence[np.ndarray],
    ) -> np.ndarray:
        if global_max_id <= 0:
            return np.zeros(1, dtype=np.uint64)

        if pair_arrays:
            pairs = np.concatenate(pair_arrays, axis=0)
            pairs = pairs[(pairs[:, 0] > 0) & (pairs[:, 1] > 0)]
        else:
            pairs = np.zeros((0, 2), dtype=np.uint64)

        if len(pairs) == 0:
            roots = np.arange(global_max_id + 1, dtype=np.uint64)
        else:
            sentinel = np.array([global_max_id], dtype=np.uint64)
            id1 = np.concatenate([pairs[:, 0].astype(np.uint64, copy=False), sentinel])
            id2 = np.concatenate([pairs[:, 1].astype(np.uint64, copy=False), sentinel])
            roots = merge_id(id1, id2)
            if len(roots) < global_max_id + 1:
                padded = np.arange(global_max_id + 1, dtype=np.uint64)
                padded[: len(roots)] = roots
                roots = padded

        mapping = np.zeros(global_max_id + 1, dtype=np.uint64)
        _, inverse = np.unique(roots[1:], return_inverse=True)
        mapping[1:] = inverse.astype(np.uint64) + 1
        return mapping


def decode_large(
    *,
    affinity_path: str,
    workflow_root: str,
    chunk_shape: Sequence[int],
    run: bool = True,
    wait: bool = True,
    **kwargs: Any,
) -> LargeDecodeRunner:
    runner = LargeDecodeRunner.create(
        affinity_path=affinity_path,
        workflow_root=workflow_root,
        chunk_shape=chunk_shape,
        **kwargs,
    )
    if run:
        runner.run_serial()
        if wait:
            runner.wait()
    return runner

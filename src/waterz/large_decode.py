"""File-backed extra-large WaterZ decoding.

This module wires the workflow planner and orchestrator into a runnable
large-volume pipeline over HDF5 affinity tensors.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np

from ._waterz import waterz as _run_waterz
from .large_workflow import BorderRef, ChunkRef, build_border_adjacency, build_chunk_grid, build_large_decode_tasks
from .orchestrator import TaskRecord, WorkflowOrchestrator
from .region_graph import merge_id

__all__ = [
    "LargeDecodeConfig",
    "LargeDecodeRunner",
    "decode_large",
]


def _merge_function_to_scoring(shorthand: str) -> str:
    parts = {tok[:3]: tok[3:] for tok in shorthand.split("_")}
    use_255 = parts.get("ran") == "255"
    wrapper = "One255Minus" if use_255 else "OneMinus"

    if "aff" in parts:
        quantile = parts["aff"]
        his_bins = parts.get("his", "0")
        if his_bins and his_bins != "0":
            inner = f"HistogramQuantileAffinity<RegionGraphType, {quantile}, ScoreValue, {his_bins}>"
        else:
            inner = f"QuantileAffinity<RegionGraphType, {quantile}, ScoreValue>"
        return f"{wrapper}<{inner}>"

    if "max" in parts:
        inner = f"MeanMaxKAffinity<RegionGraphType, {parts['max']}, ScoreValue>"
        return f"{wrapper}<{inner}>"

    if "<" in shorthand:
        return shorthand

    raise ValueError(
        f"Unknown merge_function shorthand: {shorthand!r}. "
        "Expected values like 'aff50_his256', 'aff85_his256', or 'max10'."
    )


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-") or "artifact"


def _normalize_thresholds(thresholds: Union[float, Sequence[float]]) -> tuple[float, ...]:
    if isinstance(thresholds, (int, float)):
        return (float(thresholds),)
    return tuple(sorted(float(value) for value in thresholds))


def _compression_kwargs(compression: Optional[str], compression_level: int) -> Dict[str, Any]:
    if compression in (None, "", "none"):
        return {}
    kwargs: Dict[str, Any] = {"compression": compression}
    if compression == "gzip":
        kwargs["compression_opts"] = int(compression_level)
    return kwargs


def _require_h5py():
    import h5py

    return h5py


@dataclass
class LargeDecodeConfig:
    affinity_path: str
    workflow_root: str
    chunk_shape: tuple[int, int, int]
    affinity_dataset: str = "main"
    thresholds: tuple[float, ...] = (0.3,)
    merge_function: str = "aff50_his256"
    aff_threshold_low: float = 0.0001
    aff_threshold_high: float = 0.9999
    channel_order: str = "zyx"
    write_output: bool = False
    output_path: Optional[str] = None
    output_dataset: str = "main"
    border_min_overlap: int = 1
    border_one_sided_threshold: float = 0.9
    border_iou_threshold: float = 0.0
    border_affinity_threshold: float = 0.0
    compression: Optional[str] = "gzip"
    compression_level: int = 4
    force_rebuild: bool = False
    volume_shape: Optional[tuple[int, int, int]] = None

    def __post_init__(self) -> None:
        self.affinity_path = str(Path(self.affinity_path))
        self.workflow_root = str(Path(self.workflow_root))
        self.chunk_shape = tuple(int(v) for v in self.chunk_shape)
        self.thresholds = _normalize_thresholds(self.thresholds)
        self.channel_order = str(self.channel_order).lower()
        if self.channel_order not in {"zyx", "xyz"}:
            raise ValueError("channel_order must be 'zyx' or 'xyz'.")
        if self.output_path is not None:
            self.output_path = str(Path(self.output_path))
        if self.volume_shape is not None:
            self.volume_shape = tuple(int(v) for v in self.volume_shape)

    @property
    def scoring_function(self) -> str:
        return _merge_function_to_scoring(self.merge_function)

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
            thresholds=tuple(float(v) for v in thresholds),
            merge_function=str(data.get("merge_function", "aff50_his256")),
            aff_threshold_low=float(data.get("aff_threshold_low", 0.0001)),
            aff_threshold_high=float(data.get("aff_threshold_high", 0.9999)),
            channel_order=str(data.get("channel_order", "zyx")),
            write_output=bool(data.get("write_output", False)),
            output_path=data.get("output_path"),
            output_dataset=str(data.get("output_dataset", "main")),
            border_min_overlap=int(data.get("border_min_overlap", 1)),
            border_one_sided_threshold=float(data.get("border_one_sided_threshold", 0.9)),
            border_iou_threshold=float(data.get("border_iou_threshold", 0.0)),
            border_affinity_threshold=float(data.get("border_affinity_threshold", 0.0)),
            compression=data.get("compression", "gzip"),
            compression_level=int(data.get("compression_level", 4)),
            force_rebuild=bool(data.get("force_rebuild", False)),
            volume_shape=(tuple(int(v) for v in data["volume_shape"]) if data.get("volume_shape") is not None else None),
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
        affinity_dataset: str = "main",
        aff_threshold_low: float = 0.0001,
        aff_threshold_high: float = 0.9999,
        channel_order: str = "zyx",
        write_output: bool = False,
        output_path: Optional[str] = None,
        output_dataset: str = "main",
        border_min_overlap: int = 1,
        border_one_sided_threshold: float = 0.9,
        border_iou_threshold: float = 0.0,
        border_affinity_threshold: float = 0.0,
        compression: Optional[str] = "gzip",
        compression_level: int = 4,
        force_rebuild: bool = False,
    ) -> "LargeDecodeRunner":
        config = LargeDecodeConfig(
            affinity_path=str(affinity_path),
            workflow_root=str(workflow_root),
            chunk_shape=tuple(int(v) for v in chunk_shape),
            affinity_dataset=str(affinity_dataset),
            thresholds=_normalize_thresholds(thresholds),
            merge_function=str(merge_function),
            aff_threshold_low=float(aff_threshold_low),
            aff_threshold_high=float(aff_threshold_high),
            channel_order=str(channel_order),
            write_output=bool(write_output),
            output_path=str(output_path) if output_path is not None else None,
            output_dataset=str(output_dataset),
            border_min_overlap=int(border_min_overlap),
            border_one_sided_threshold=float(border_one_sided_threshold),
            border_iou_threshold=float(border_iou_threshold),
            border_affinity_threshold=float(border_affinity_threshold),
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
    def chunks(self) -> list[ChunkRef]:
        if self.config.volume_shape is None:
            raise RuntimeError("LargeDecodeRunner is not initialized with a volume shape.")
        return build_chunk_grid(self.config.volume_shape, self.config.chunk_shape)

    @property
    def chunk_map(self) -> Dict[str, ChunkRef]:
        return {chunk.key: chunk for chunk in self.chunks}

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
        tasks = build_large_decode_tasks(self.chunks, write_output=self.config.write_output)
        self.orchestrator.register(tasks)

    def handlers(self) -> Dict[str, Any]:
        return {
            "decode_chunk": self.handle_decode_chunk,
            "compute_offsets": self.handle_compute_offsets,
            "connect_border": self.handle_connect_border,
            "reduce_relabel": self.handle_reduce_relabel,
            "apply_relabel": self.handle_apply_relabel,
            "assemble_output": self.handle_assemble_output,
        }

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
        chunk = self.chunk_map[record.spec.key]
        affs = self._read_affinity_chunk(chunk)
        seg_results = _run_waterz(
            affs,
            thresholds=self.config.thresholds,
            aff_threshold_low=self.config.aff_threshold_low,
            aff_threshold_high=self.config.aff_threshold_high,
            scoring_function=self.config.scoring_function,
            force_rebuild=self.config.force_rebuild,
        )
        seg = np.asarray(seg_results[-1], dtype=np.uint64)
        path = self._raw_chunk_path(chunk.key)
        self._write_chunk_seg(path, seg)
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
        src_face = self._read_chunk_face(self._raw_chunk_path(border.src.key), border.axis, side="src")
        dst_face = self._read_chunk_face(self._raw_chunk_path(border.dst.key), border.axis, side="dst")

        src_offset = int(offsets["chunk_offsets"][border.src.key])
        dst_offset = int(offsets["chunk_offsets"][border.dst.key])
        if src_offset:
            src_mask = src_face > 0
            src_face[src_mask] += src_offset
        if dst_offset:
            dst_mask = dst_face > 0
            dst_face[dst_mask] += dst_offset

        aff = self._read_boundary_affinity(border)
        pairs = self._compute_merge_pairs(src_face, dst_face, aff)
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
        path = self._final_chunk_path(chunk.key)
        self._write_chunk_seg(path, seg)
        return {"chunk_path": str(path), "max_id": int(seg.max())}

    def handle_assemble_output(self, record: TaskRecord) -> Dict[str, Any]:
        output_path = self.config.resolved_output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        h5py = _require_h5py()
        shape = tuple(int(v) for v in self.config.volume_shape or ())
        chunk_shape = tuple(min(shape[i], self.config.chunk_shape[i]) for i in range(3))
        kwargs = _compression_kwargs(self.config.compression, self.config.compression_level)
        with h5py.File(output_path, "w") as handle:
            dataset = handle.create_dataset(
                self.config.output_dataset,
                shape=shape,
                dtype=np.uint64,
                chunks=chunk_shape,
                **kwargs,
            )
            for chunk in self.chunks:
                seg = self._read_chunk_seg(self._final_chunk_path(chunk.key))
                dataset[
                    chunk.start[0]:chunk.stop[0],
                    chunk.start[1]:chunk.stop[1],
                    chunk.start[2]:chunk.stop[2],
                ] = seg
        return {"output_path": str(output_path)}

    def _discover_volume_shape(self) -> tuple[int, int, int]:
        h5py = _require_h5py()
        with h5py.File(self.config.affinity_path, "r") as handle:
            dataset = handle[self.config.affinity_dataset]
            if dataset.ndim != 4 or dataset.shape[0] < 3:
                raise ValueError(
                    f"Expected affinity dataset with shape (C>=3, Z, Y, X), got {dataset.shape}."
                )
            return tuple(int(v) for v in dataset.shape[1:])

    def _config_path(self) -> Path:
        return self.root / self.CONFIG_FILENAME

    def _write_config(self) -> None:
        self._write_json(self._config_path(), self.config.to_dict())

    def _raw_chunk_path(self, chunk_key: str) -> Path:
        return self.root / "chunks" / "raw" / f"{chunk_key}.h5"

    def _final_chunk_path(self, chunk_key: str) -> Path:
        return self.root / "chunks" / "final" / f"{chunk_key}.h5"

    def _connect_path(self, border_key: str) -> Path:
        return self.root / "connect" / f"{_safe_name(border_key)}.npy"

    def _offsets_path(self) -> Path:
        return self.root / "artifacts" / "offsets.json"

    def _relabel_path(self) -> Path:
        return self.root / "artifacts" / "relabel.npy"

    def _write_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _read_json(self, path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return dict(json.load(handle))

    def _read_affinity_chunk(self, chunk: ChunkRef) -> np.ndarray:
        h5py = _require_h5py()
        with h5py.File(self.config.affinity_path, "r") as handle:
            dataset = handle[self.config.affinity_dataset]
            affs = np.array(
                dataset[
                    0:3,
                    chunk.start[0]:chunk.stop[0],
                    chunk.start[1]:chunk.stop[1],
                    chunk.start[2]:chunk.stop[2],
                ],
                dtype=np.float32,
            )
        if self.config.channel_order == "xyz":
            affs = affs[[2, 1, 0]]
        return np.ascontiguousarray(affs, dtype=np.float32)

    def _read_boundary_affinity(self, border: BorderRef) -> np.ndarray:
        h5py = _require_h5py()
        channel_index = self.config.channel_index(border.axis)
        with h5py.File(self.config.affinity_path, "r") as handle:
            dataset = handle[self.config.affinity_dataset]
            if border.axis == "z":
                data = dataset[
                    channel_index,
                    border.dst.start[0],
                    border.src.start[1]:border.src.stop[1],
                    border.src.start[2]:border.src.stop[2],
                ]
            elif border.axis == "y":
                data = dataset[
                    channel_index,
                    border.src.start[0]:border.src.stop[0],
                    border.dst.start[1],
                    border.src.start[2]:border.src.stop[2],
                ]
            else:
                data = dataset[
                    channel_index,
                    border.src.start[0]:border.src.stop[0],
                    border.src.start[1]:border.src.stop[1],
                    border.dst.start[2],
                ]
            return np.array(data, dtype=np.float32)

    def _write_chunk_seg(self, path: Path, seg: np.ndarray) -> None:
        h5py = _require_h5py()
        path.parent.mkdir(parents=True, exist_ok=True)
        kwargs = _compression_kwargs(self.config.compression, self.config.compression_level)
        with h5py.File(path, "w") as handle:
            handle.create_dataset("main", data=seg, dtype=seg.dtype, **kwargs)
            handle.create_dataset("max", data=np.asarray(int(seg.max()), dtype=np.uint64))

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

    def _compute_merge_pairs(
        self,
        face0: np.ndarray,
        face1: np.ndarray,
        aff: Optional[np.ndarray],
    ) -> np.ndarray:
        fg = (face0 > 0) & (face1 > 0)
        if not fg.any():
            return np.zeros((0, 2), dtype=np.uint64)

        ids0 = face0[fg].astype(np.uint64, copy=False)
        ids1 = face1[fg].astype(np.uint64, copy=False)
        pair_ids, inverse, overlap = np.unique(
            np.stack([ids0, ids1], axis=1),
            axis=0,
            return_inverse=True,
            return_counts=True,
        )

        uniq0, cnt0 = np.unique(face0[face0 > 0], return_counts=True)
        uniq1, cnt1 = np.unique(face1[face1 > 0], return_counts=True)
        size0_map = dict(zip(uniq0.tolist(), cnt0.tolist()))
        size1_map = dict(zip(uniq1.tolist(), cnt1.tolist()))
        size0 = np.array([size0_map[int(value)] for value in pair_ids[:, 0]], dtype=np.float64)
        size1 = np.array([size1_map[int(value)] for value in pair_ids[:, 1]], dtype=np.float64)
        overlap = overlap.astype(np.float64)
        one_sided = overlap / np.maximum(np.minimum(size0, size1), 1.0)
        union = np.maximum(size0 + size1 - overlap, 1.0)
        iou = overlap / union

        keep = overlap >= float(self.config.border_min_overlap)
        if self.config.border_one_sided_threshold > 0:
            keep &= one_sided >= float(self.config.border_one_sided_threshold)
        if self.config.border_iou_threshold > 0:
            keep &= iou >= float(self.config.border_iou_threshold)

        if aff is not None and self.config.border_affinity_threshold > 0:
            aff_vals = aff[fg].astype(np.float64, copy=False)
            aff_sums = np.zeros(len(pair_ids), dtype=np.float64)
            np.add.at(aff_sums, inverse, aff_vals)
            mean_aff = aff_sums / overlap
            keep &= mean_aff >= float(self.config.border_affinity_threshold)

        return np.ascontiguousarray(pair_ids[keep], dtype=np.uint64)

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
                padded[:len(roots)] = roots
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

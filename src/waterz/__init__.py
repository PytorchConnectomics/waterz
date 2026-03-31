from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("waterz")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "uninstalled"

from ._agglomerate import agglomerate, build_region_graph_rich
from ._merge import dust_merge_from_region_graph, get_region_graph, get_region_graph_rich, merge_dust, merge_function_to_scoring, merge_region_graphs, merge_segments, strip_boundary
from ._uint8 import float_to_uint8, prepare_affinities, scale_aff_threshold, scale_thresholds
from ._waterz import waterz
from .evaluate import evaluate
from .large_decode import LargeDecodeConfig, LargeDecodeRunner, decode_large
from .large_workflow import (
    build_border_adjacency,
    build_chunk_grid,
    build_chunk_grid_overlap,
    build_large_decode_tasks,
    build_large_decode_tasks_overlap,
)
from .orchestrator import TaskRecord, TaskSpec, TaskState, WorkflowOrchestrator
from .overlap_stitch import apply_overlap_remap, build_overlap_remap
from .face_merge import face_merge_pairs, slice_overlaps
from .region_graph import merge_id
from .seg_init import compute_fragments

__all__ = [
    "LargeDecodeConfig",
    "LargeDecodeRunner",
    "TaskRecord",
    "TaskSpec",
    "TaskState",
    "WorkflowOrchestrator",
    "agglomerate",
    "apply_overlap_remap",
    "build_border_adjacency",
    "build_chunk_grid",
    "build_chunk_grid_overlap",
    "build_large_decode_tasks",
    "build_large_decode_tasks_overlap",
    "build_overlap_remap",
    "build_region_graph_rich",
    "compute_fragments",
    "decode_large",
    "dust_merge_from_region_graph",
    "evaluate",
    "face_merge_pairs",
    "float_to_uint8",
    "get_region_graph",
    "get_region_graph_rich",
    "merge_dust",
    "merge_function_to_scoring",
    "merge_id",
    "merge_region_graphs",
    "merge_segments",
    "prepare_affinities",
    "scale_aff_threshold",
    "scale_thresholds",
    "slice_overlaps",
    "strip_boundary",
    "waterz",
]

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("waterz")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "uninstalled"

from ._agglomerate import agglomerate
from ._merge import dust_merge_from_region_graph, get_region_graph, merge_dust, merge_function_to_scoring, merge_segments, strip_boundary
from ._waterz import waterz
from .evaluate import evaluate
from .large_decode import LargeDecodeConfig, LargeDecodeRunner, decode_large
from .large_workflow import (
    build_border_adjacency,
    build_chunk_grid,
    build_large_decode_tasks,
)
from .orchestrator import TaskRecord, TaskSpec, TaskState, WorkflowOrchestrator
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
    "build_border_adjacency",
    "dust_merge_from_region_graph",
    "build_chunk_grid",
    "build_large_decode_tasks",
    "compute_fragments",
    "decode_large",
    "evaluate",
    "face_merge_pairs",
    "get_region_graph",
    "merge_dust",
    "merge_function_to_scoring",
    "merge_id",
    "merge_segments",
    "slice_overlaps",
    "strip_boundary",
    "waterz",
]

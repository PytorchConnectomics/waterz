from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("psygnal")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "uninstalled"

from ._agglomerate import agglomerate
from ._merge import get_region_graph, merge_dust, merge_segments
from ._waterz import waterz
from .evaluate import evaluate
from .large_decode import LargeDecodeConfig, LargeDecodeRunner, decode_large
from .large_workflow import (
    build_border_adjacency,
    build_chunk_grid,
    build_large_decode_tasks,
)
from .orchestrator import TaskRecord, TaskSpec, TaskState, WorkflowOrchestrator
from .region_graph import merge_id

__all__ = [
    "LargeDecodeConfig",
    "LargeDecodeRunner",
    "TaskRecord",
    "TaskSpec",
    "TaskState",
    "WorkflowOrchestrator",
    "agglomerate",
    "build_border_adjacency",
    "build_chunk_grid",
    "build_large_decode_tasks",
    "decode_large",
    "evaluate",
    "get_region_graph",
    "merge_dust",
    "merge_id",
    "merge_segments",
    "waterz",
]

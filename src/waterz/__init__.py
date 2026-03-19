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
from .region_graph import merge_id

__all__ = [
    "agglomerate",
    "evaluate",
    "get_region_graph",
    "merge_dust",
    "merge_id",
    "merge_segments",
    "waterz",
]

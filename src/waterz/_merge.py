"""High-level Python API for region graph, merge, and dust removal.

Region graph uses waterz's JIT-compiled scoring functions via
``agglomerate()``, supporting any scoring function. Channel filtering
(z-only, xy-only) is done by zeroing out unwanted affinity channels.

Merge uses the standalone C++ ``merge`` extension for size+affinity
merge and dust removal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np

from .merge import merge as _c_merge

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "get_region_graph",
    "merge_segments",
    "merge_dust",
]


def _mask_channels(affs: np.ndarray, channels: str) -> np.ndarray:
    """Zero out affinity channels not in the selection.

    Edges with zero affinity still appear in the region graph but get
    score ~0, so they are effectively ignored by merge thresholds.
    """
    ch = channels.lower() if isinstance(channels, str) else str(channels)
    if ch in ("all", "zyx", "xyz", "7"):
        return affs
    affs = affs.copy()
    if ch in ("z", "z-only", "z_only", "1"):
        affs[1:] = 0
    elif ch in ("xy", "xy-only", "xy_only", "yx", "6"):
        affs[0] = 0
    else:
        raise ValueError(
            f"Unknown channels: {channels!r}. Use 'all', 'z', or 'xy'."
        )
    return affs


def get_region_graph(
    seg: NDArray[np.uint64],
    affs: NDArray[np.float32],
    scoring_function: str = "MeanAffinity<RegionGraphType, ScoreValue>",
    channels: str = "all",
) -> Tuple[NDArray[np.float32], NDArray[np.uint64], NDArray[np.uint64]]:
    """Build region graph using waterz's JIT-compiled scoring functions.

    Supports any waterz scoring function — max, mean, histogram quantiles,
    top-K affinities, composable operators, etc.

    Parameters
    ----------
    seg : ndarray, uint64, shape ``(Z, Y, X)``
        Segmentation (0 = background).
    affs : ndarray, float32, shape ``(3, Z, Y, X)``
        Affinities in z, y, x channel order.
    scoring_function : str
        C++ scoring function type string.  Common options:

        - ``"MeanAffinity<RegionGraphType, ScoreValue>"`` — mean (default)
        - ``"MaxAffinity<RegionGraphType, ScoreValue>"`` — max
        - ``"HistogramQuantileAffinity<RegionGraphType, 85, ScoreValue, 256>"`` — p85
        - ``"HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>"`` — median

        Note: do NOT wrap with ``OneMinus<...>`` — ``merge_segments``
        expects raw affinities sorted descending (high = strong connection).

        Use ``connectomics.decoding.decoders.waterz._merge_function_to_scoring()``
        to convert shorthands like ``"aff85_his256"``.
    channels : str
        Which affinity directions to include: ``"all"`` (default),
        ``"z"`` (z-only), or ``"xy"`` (xy-only).

    Returns
    -------
    rg_affs : ndarray, float32, shape ``(E,)``
        Scored affinity per edge, sorted descending.
    id1, id2 : ndarray, uint64, shape ``(E,)``
        Edge endpoints.
    """
    from ._agglomerate import build_region_graph_only

    seg = np.ascontiguousarray(seg, dtype=np.uint64)
    affs = np.ascontiguousarray(affs, dtype=np.float32)
    affs = _mask_channels(affs, channels)

    rg_list = build_region_graph_only(affs, seg, scoring_function=scoring_function)

    if rg_list is None or len(rg_list) == 0:
        empty_f = np.empty(0, dtype=np.float32)
        empty_id = np.empty(0, dtype=np.uint64)
        return empty_f, empty_id, empty_id

    rg_affs = np.array([e["score"] for e in rg_list], dtype=np.float32)
    id1 = np.array([e["u"] for e in rg_list], dtype=np.uint64)
    id2 = np.array([e["v"] for e in rg_list], dtype=np.uint64)

    order = np.argsort(-rg_affs)
    return rg_affs[order], id1[order], id2[order]


def merge_segments(
    seg: NDArray[np.uint64],
    rg_affs: NDArray[np.float32],
    id1: NDArray[np.uint64],
    id2: NDArray[np.uint64],
    counts: NDArray[np.uint64],
    size_th: int,
    weight_th: float = 0.0,
    dust_th: int = 0,
) -> int:
    """Size+affinity merge followed by dust removal.

    Modifies *seg* in-place and returns the new segment count.

    Parameters
    ----------
    seg : ndarray, uint64, shape ``(Z, Y, X)``
    rg_affs : ndarray, float32, shape ``(E,)`` — sorted descending
    id1, id2 : ndarray, uint64, shape ``(E,)``
    counts : ndarray, uint64, shape ``(max_id + 1,)``
    size_th : int
        Merge if at least one segment < this many voxels.
    weight_th : float
        Minimum affinity for an edge to be merge-eligible.
    dust_th : int
        Remove segments smaller than this after merging.

    Returns
    -------
    int
        Number of segments remaining (excluding background).
    """
    seg = np.ascontiguousarray(seg, dtype=np.uint64)
    rg_affs = np.ascontiguousarray(rg_affs, dtype=np.float32)
    id1 = np.ascontiguousarray(id1, dtype=np.uint64)
    id2 = np.ascontiguousarray(id2, dtype=np.uint64)
    counts = np.ascontiguousarray(counts, dtype=np.uint64)
    return _c_merge(seg, rg_affs, id1, id2, counts, size_th, weight_th, dust_th)


def merge_dust(
    seg: NDArray[np.uint64],
    affs: NDArray[np.float32],
    size_th: int,
    weight_th: float = 0.0,
    dust_th: int = 0,
    scoring_function: str = "MeanAffinity<RegionGraphType, ScoreValue>",
    channels: str = "all",
) -> NDArray[np.uint64]:
    """Convenience: build region graph + merge in one call.

    Parameters
    ----------
    seg : ndarray, uint64, shape ``(Z, Y, X)``
        Segmentation (0 = background).  Modified in-place.
    affs : ndarray, float32, shape ``(3, Z, Y, X)``
        Affinities in z, y, x channel order.
    size_th : int
        Merge if at least one segment has fewer voxels than this.
    weight_th : float
        Minimum affinity for merge eligibility.
    dust_th : int
        Remove segments smaller than this after merging.
    scoring_function : str
        Scoring function for region graph construction.
    channels : str
        Which directions: ``"all"``, ``"z"``, or ``"xy"``.

    Returns
    -------
    seg : ndarray, uint64
        Cleaned segmentation (same array, modified in-place).
    """
    seg = np.ascontiguousarray(seg, dtype=np.uint64)
    affs = np.ascontiguousarray(affs, dtype=np.float32)

    rg_affs, id1, id2 = get_region_graph(
        seg, affs, scoring_function=scoring_function, channels=channels
    )

    ids, cnts = np.unique(seg, return_counts=True)
    max_id = int(ids.max()) if len(ids) else 0
    counts = np.zeros(max_id + 1, dtype=np.uint64)
    counts[ids] = cnts

    _c_merge(seg, rg_affs, id1, id2, counts, size_th, weight_th, dust_th)
    return seg

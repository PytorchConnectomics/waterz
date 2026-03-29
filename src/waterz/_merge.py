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
    "dust_merge_from_region_graph",
    "get_region_graph",
    "merge_function_to_scoring",
    "merge_segments",
    "merge_dust",
    "strip_boundary",
]


def strip_boundary(
    seg: "NDArray",
    affs: "NDArray",
    threshold: float = 0.1,
    channels: str = "xy",
) -> int:
    """Zero out segmentation voxels at weak affinity boundaries.

    Strips noisy boundary voxels from segments so that subsequent dust
    merge sees true core sizes rather than inflated sizes.  Segments
    that shrink to size 0 are naturally handled by ``dust_remove_size``.

    Parameters
    ----------
    seg : ndarray, shape ``(Z, Y, X)``
        Segmentation (modified in-place).
    affs : ndarray, shape ``(3, Z, Y, X)``
        Affinities in **z, y, x** channel order.  Values in [0, 1] or
        [0, 255] for uint8.
    threshold : float
        Voxels with mean affinity below this value are set to 0.
        Specified in [0, 1] range regardless of dtype (auto-scaled
        for uint8).  Default: 0.1
    channels : str
        Which affinity channels to average: ``"xy"`` (default, channels
        1+2), ``"all"`` (channels 0+1+2), or ``"z"`` (channel 0 only).

    Returns
    -------
    int
        Number of voxels removed.
    """
    affs = np.asarray(affs)
    is_uint8 = affs.dtype == np.uint8

    if channels == "xy":
        mean_aff = affs[1:3].astype(np.float32, copy=False).mean(axis=0)
    elif channels == "all":
        mean_aff = affs[0:3].astype(np.float32, copy=False).mean(axis=0)
    elif channels == "z":
        mean_aff = affs[0].astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unknown channels: {channels!r}. Expected 'xy', 'all', or 'z'.")

    if is_uint8:
        mean_aff /= 255.0

    boundary_mask = mean_aff < threshold
    n_removed = int(boundary_mask.sum())
    seg[boundary_mask] = 0
    return n_removed


# ---------------------------------------------------------------------------
# Shorthand -> C++ scoring function conversion
# ---------------------------------------------------------------------------

_RG = "RegionGraphType"
_SV = "ScoreValue"


def merge_function_to_scoring(shorthand: str) -> str:
    """Convert a shorthand merge function name to a C++ scoring type string.

    Supported shorthands (examples)::

        affmean       -> OneMinus<MeanAffinity<RG, SV>>
        aff50_his256  -> OneMinus<HistogramQuantileAffinity<RG, 50, SV, 256>>
        aff85_his256  -> OneMinus<HistogramQuantileAffinity<RG, 85, SV, 256>>
        aff50_his0    -> OneMinus<QuantileAffinity<RG, 50, SV>>
        max10         -> OneMinus<MeanMaxKAffinity<RG, 10, SV>>
        *_ran255      -> One255Minus<...> instead of OneMinus<...>
    """
    parts = {tok[:3]: tok[3:] for tok in shorthand.split("_")}
    use_255 = parts.get("ran") == "255"
    wrapper = "One255Minus" if use_255 else "OneMinus"

    if shorthand in {"affmean", "mean", "mean_affinity"}:
        inner = f"MeanAffinity<{_RG}, {_SV}>"
        return f"{wrapper}<{inner}>"

    if "aff" in parts:
        quantile = parts["aff"]
        his_bins = parts.get("his", "0")
        if his_bins and his_bins != "0":
            inner = f"HistogramQuantileAffinity<{_RG}, {quantile}, {_SV}, {his_bins}>"
        else:
            inner = f"QuantileAffinity<{_RG}, {quantile}, {_SV}>"
        return f"{wrapper}<{inner}>"

    if "max" in parts:
        k = parts["max"]
        inner = f"MeanMaxKAffinity<{_RG}, {k}, {_SV}>"
        return f"{wrapper}<{inner}>"

    # If it already looks like a C++ type string, pass through
    if "<" in shorthand:
        return shorthand

    raise ValueError(
        f"Unknown merge_function shorthand: {shorthand!r}. "
        "Expected format like 'affmean', 'aff50_his256', 'aff85_his256', 'max10', etc."
    )


def _prepare_affinities(affs: np.ndarray) -> np.ndarray:
    """Preserve float32/uint8 affinity semantics for region-graph scoring."""
    affs = np.ascontiguousarray(affs)
    if affs.dtype == np.float64:
        affs = affs.astype(np.float32)
    if affs.dtype not in (np.dtype("float32"), np.dtype("uint8")):
        raise TypeError(f"affs.dtype must be float32 or uint8, got {affs.dtype}")
    return affs


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
    affs: NDArray,
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
    affs : ndarray, float32 or uint8, shape ``(3, Z, Y, X)``
        Affinities in z, y, x channel order.
    scoring_function : str
        C++ scoring function type string.  Common options:

        - ``"MeanAffinity<RegionGraphType, ScoreValue>"`` — mean (default)
        - ``"MaxAffinity<RegionGraphType, ScoreValue>"`` — max
        - ``"HistogramQuantileAffinity<RegionGraphType, 85, ScoreValue, 256>"`` — p85
        - ``"HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>"`` — median

        Note: do NOT wrap with ``OneMinus<...>`` — ``merge_segments``
        expects raw affinities sorted descending (high = strong connection).

        Use :func:`waterz.merge_function_to_scoring` to convert shorthands
        like ``"aff85_his256"``.
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
    affs = _prepare_affinities(affs)
    affs = _mask_channels(affs, channels)
    aff_dtype = affs.dtype

    rg_list = build_region_graph_only(affs, seg, scoring_function=scoring_function)

    if rg_list is None or len(rg_list) == 0:
        empty_f = np.empty(0, dtype=np.float32)
        empty_id = np.empty(0, dtype=np.uint64)
        return empty_f, empty_id, empty_id

    rg_affs = np.array([e["score"] for e in rg_list], dtype=np.float32)
    if aff_dtype == np.uint8:
        rg_affs /= 255.0
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
    affs: NDArray,
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
    affs : ndarray, float32 or uint8, shape ``(3, Z, Y, X)``
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
    affs = _prepare_affinities(affs)

    rg_affs, id1, id2 = get_region_graph(
        seg, affs, scoring_function=scoring_function, channels=channels
    )

    ids, cnts = np.unique(seg, return_counts=True)
    max_id = int(ids.max()) if len(ids) else 0
    counts = np.zeros(max_id + 1, dtype=np.uint64)
    counts[ids] = cnts

    _c_merge(seg, rg_affs, id1, id2, counts, size_th, weight_th, dust_th)
    return seg


def _build_segment_counts(seg: np.ndarray) -> np.ndarray:
    """Build a dense counts array indexed by segment id."""
    ids, cnts = np.unique(seg, return_counts=True)
    max_id = int(ids.max()) if len(ids) else 0
    counts = np.zeros(max_id + 1, dtype=np.uint64)
    counts[ids] = cnts
    return counts


def dust_merge_from_region_graph(
    seg: np.ndarray,
    region_graph: list,
    *,
    is_uint8: bool = False,
    size_th: int,
    weight_th: float = 0.0,
    dust_th: int = 0,
) -> None:
    """Invert OneMinus/One255Minus scores and merge dust segments.

    Extracts edges from a waterz region graph (list of dicts with
    ``"u"``, ``"v"``, ``"score"`` keys), inverts the OneMinus or
    One255Minus scoring back to raw affinities, sorts descending,
    builds segment counts, and calls :func:`merge_segments`.

    Modifies *seg* in-place.

    Parameters
    ----------
    seg : ndarray, uint64, shape ``(Z, Y, X)``
        Segmentation to clean.  Modified in-place.
    region_graph : list[dict]
        Region graph edges as returned by ``waterz.waterz()`` with
        ``return_region_graph=True``.  Each dict has keys
        ``"u"``, ``"v"``, ``"score"``.
    is_uint8 : bool
        If True, scores are in [0, 255] range (One255Minus) instead
        of [0, 1] (OneMinus).
    size_th : int
        Merge if at least one segment has fewer voxels than this.
    weight_th : float
        Minimum affinity for an edge to be merge-eligible.
    dust_th : int
        Remove segments smaller than this after merging.
    """
    seg = np.ascontiguousarray(seg, dtype=np.uint64)
    n_edges = len(region_graph)
    rg_affs = np.empty(n_edges, dtype=np.float32)
    id1 = np.empty(n_edges, dtype=np.uint64)
    id2 = np.empty(n_edges, dtype=np.uint64)
    score_max = 255.0 if is_uint8 else 1.0
    for idx, edge in enumerate(region_graph):
        rg_affs[idx] = score_max - float(edge["score"])
        id1[idx] = int(edge["u"])
        id2[idx] = int(edge["v"])
    if n_edges:
        np.clip(rg_affs, 0.0, score_max, out=rg_affs)
        order = np.argsort(rg_affs)[::-1]
        rg_affs = np.ascontiguousarray(rg_affs[order])
        id1 = np.ascontiguousarray(id1[order])
        id2 = np.ascontiguousarray(id2[order])
    counts = _build_segment_counts(seg)
    merge_segments(
        seg, rg_affs, id1, id2, counts,
        size_th=size_th,
        weight_th=weight_th,
        dust_th=dust_th,
    )

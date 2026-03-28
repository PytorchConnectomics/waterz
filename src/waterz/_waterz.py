"""High-level convenience wrapper around :func:`agglomerate`.

Returns all segmentations at once (copied) instead of yielding them from
a generator.  This makes it safe to hold references to multiple threshold
results simultaneously — which is critical for batch parameter sweeps
(e.g. Optuna tuning) where watershed + region-graph construction should
happen only once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Union

import numpy as np

from ._agglomerate import agglomerate

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _copy_agglomerate_result(
    result: Any,
    dtype: np.dtype = np.dtype("uint64"),
    *,
    copy_segmentation: bool = True,
) -> Any:
    """Copy the segmentation while preserving agglomerate's tuple layout."""
    if not copy_segmentation:
        return result
    if isinstance(result, tuple):
        if not result:
            return result
        seg = np.array(result[0], dtype=dtype, copy=True)
        return (seg, *result[1:])
    return np.array(result, dtype=dtype, copy=True)


def waterz(
    affs: NDArray[np.float32],
    thresholds: Union[float, Sequence[float]],
    *,
    gt: NDArray[np.uint32] | None = None,
    fragments: NDArray[np.uint64] | None = None,
    compute_fragments: bool = False,
    seed_method: str = "maxima_distance",
    aff_threshold_low: float = 0.0001,
    aff_threshold_high: float = 0.9999,
    scoring_function: str = "OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>",
    discretize_queue: int = 0,
    force_rebuild: bool = False,
    return_region_graph: bool = False,
    seg_dtype: str = "uint64",
    as_dict: bool = False,
) -> Union[List[Any], Dict[float, Any]]:
    """Run watershed + agglomeration and return all segmentations.

    Unlike :func:`agglomerate` (a generator whose output array is reused
    in-place), this function copies each result so callers can safely keep
    references to every threshold's segmentation at once.

    The underlying C++ engine performs watershed and region-graph extraction
    **once**, then incrementally merges for each threshold in ascending
    order — so passing *N* thresholds is nearly as fast as passing one.

    Parameters
    ----------
    affs : ndarray, float32 or uint8, shape ``(3, Z, Y, X)``
        Affinity predictions.
    thresholds : float or sequence of float
        One or more agglomeration thresholds.
    gt : ndarray, optional
        Ground-truth for inline Rand/VOI evaluation.
    fragments : ndarray, optional
        Pre-computed over-segmentation to skip the watershed step.
        Takes precedence over *compute_fragments*.
    compute_fragments : bool
        If True and *fragments* is None, run 2D slice-by-slice watershed
        via ``mahotas.cwatershed`` instead of waterz's built-in C++
        watershed.  Useful for anisotropic EM data.  Default: False.
    seed_method : str
        Seed placement for 2D watershed: ``"maxima_distance"`` (default),
        ``"minima"``, ``"grid"``, or ``"grid-N"`` (grid with spacing N).
        Only used when *compute_fragments* is True.
    aff_threshold_low, aff_threshold_high : float
        Affinity thresholds for the built-in C++ watershed.
        When *compute_fragments* is True, *aff_threshold_low* also
        controls border removal (zero out voxels with mean xy aff below it).
    scoring_function : str
        C++ type string for the merge scoring function.
    discretize_queue : int
        Use approximate bin-queue if > 0.
    force_rebuild : bool
        Force recompilation of the Cython module.
    return_region_graph : bool
        If True, each result includes the region graph after the copied
        segmentation, matching :func:`agglomerate`.
    as_dict : bool
        If True return ``{threshold: result, ...}`` instead of a list.

    Returns
    -------
    list or dict
        If neither *gt* nor *return_region_graph* is requested, returns one
        copied uint64 segmentation per threshold.

        Otherwise each item matches :func:`agglomerate`'s tuple ordering, with
        the first element replaced by a copied segmentation array.
    """
    # 2D slice-by-slice watershed via mahotas (alternative to C++ built-in)
    if fragments is None and compute_fragments:
        from .seg_init import compute_fragments as _compute_fragments

        fragments = _compute_fragments(
            affs,
            seed_method=seed_method,
            aff_threshold_low=aff_threshold_low,
        )

    if isinstance(thresholds, (int, float)):
        thresholds_list = [float(thresholds)]
    else:
        thresholds_list = [float(t) for t in thresholds]

    out_dtype = np.dtype(seg_dtype)

    results: List[Any] = []
    copy_segmentation = len(thresholds_list) > 1
    for result in agglomerate(
        affs,
        thresholds=thresholds_list,
        gt=gt,
        fragments=fragments,
        aff_threshold_low=aff_threshold_low,
        aff_threshold_high=aff_threshold_high,
        return_region_graph=return_region_graph,
        scoring_function=scoring_function,
        discretize_queue=discretize_queue,
        seg_dtype=seg_dtype,
        force_rebuild=force_rebuild,
    ):
        results.append(
            _copy_agglomerate_result(
                result,
                out_dtype,
                copy_segmentation=copy_segmentation,
            )
        )

    if as_dict:
        return {round(t, 10): s for t, s in zip(thresholds_list, results)}
    return results

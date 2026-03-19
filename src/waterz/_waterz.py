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


def _copy_agglomerate_result(result: Any) -> Any:
    """Copy the segmentation while preserving agglomerate's tuple layout."""
    if isinstance(result, tuple):
        if not result:
            return result
        return (np.array(result[0], copy=True), *result[1:])
    return np.array(result, copy=True)


def waterz(
    affs: NDArray[np.float32],
    thresholds: Union[float, Sequence[float]],
    *,
    gt: NDArray[np.uint32] | None = None,
    fragments: NDArray[np.uint64] | None = None,
    aff_threshold_low: float = 0.0001,
    aff_threshold_high: float = 0.9999,
    scoring_function: str = "OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>",
    discretize_queue: int = 0,
    force_rebuild: bool = False,
    return_region_graph: bool = False,
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
    affs : ndarray, float32, shape ``(3, Z, Y, X)``
        Affinity predictions in ``[0, 1]``.
    thresholds : float or sequence of float
        One or more agglomeration thresholds.
    gt : ndarray, optional
        Ground-truth for inline Rand/VOI evaluation. When provided, each result
        includes metrics after the copied segmentation, matching
        :func:`agglomerate`.
    fragments : ndarray, optional
        Pre-computed over-segmentation to skip the watershed step.
    aff_threshold_low, aff_threshold_high : float
        Affinity thresholds for the initial watershed.
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
    if isinstance(thresholds, (int, float)):
        thresholds_list = [float(thresholds)]
    else:
        thresholds_list = [float(t) for t in thresholds]

    results: List[Any] = []
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
        force_rebuild=force_rebuild,
    ):
        results.append(_copy_agglomerate_result(result))

    if as_dict:
        return {round(t, 10): s for t, s in zip(thresholds_list, results)}
    return results

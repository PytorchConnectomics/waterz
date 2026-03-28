"""Compute merge pairs between two adjacent 2D label faces.

Shared by:
- ``large_decode.py`` — border stitching between chunks
- ``branch_merge.py`` (pytc) — z-slice false-split resolution

All overlap/IOU/affinity logic lives here to avoid duplication.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["slice_overlaps", "face_merge_pairs"]


def slice_overlaps(
    s0: NDArray,
    s1: NDArray,
    aff: Optional[NDArray] = None,
) -> NDArray[np.float64]:
    """Overlap statistics between two adjacent 2D label maps.

    Parameters
    ----------
    s0, s1 : ndarray, 2D
        Label maps from adjacent faces (e.g. z-slices, chunk boundaries).
    aff : ndarray, 2D, optional
        Boundary affinity map (same shape as s0/s1).

    Returns
    -------
    ndarray, shape ``(N, K)`` float64
        Each row: ``[id0, id1, size0, size1, overlap, mean_aff?]``.
        K = 6 if *aff* is provided, else 5.
    """
    fg = (s0 > 0) & (s1 > 0)
    ncols = 6 if aff is not None else 5
    if not fg.any():
        return np.empty((0, ncols), dtype=np.float64)

    a = s0[fg].astype(np.int64)
    b = s1[fg].astype(np.int64)

    u0, c0 = np.unique(s0[s0 > 0], return_counts=True)
    u1, c1 = np.unique(s1[s1 > 0], return_counts=True)
    size0_map = dict(zip(u0.tolist(), c0.tolist()))
    size1_map = dict(zip(u1.tolist(), c1.tolist()))

    pairs = np.stack([a, b], axis=1)
    unique_pairs, inverse, counts = np.unique(
        pairs, axis=0, return_inverse=True, return_counts=True,
    )

    n = len(unique_pairs)
    result = np.zeros((n, ncols), dtype=np.float64)
    result[:, 0] = unique_pairs[:, 0]
    result[:, 1] = unique_pairs[:, 1]
    result[:, 2] = np.array([size0_map[int(i)] for i in unique_pairs[:, 0]])
    result[:, 3] = np.array([size1_map[int(i)] for i in unique_pairs[:, 1]])
    result[:, 4] = counts

    if aff is not None:
        aff_vals = aff[fg].astype(np.float64)
        aff_sums = np.zeros(n, dtype=np.float64)
        np.add.at(aff_sums, inverse, aff_vals)
        result[:, 5] = aff_sums / counts

    return result


def face_merge_pairs(
    face0: NDArray,
    face1: NDArray,
    aff: Optional[NDArray] = None,
    *,
    min_overlap: int = 1,
    iou_threshold: float = 0.0,
    one_sided_threshold: float = 0.9,
    one_sided_min_size: int = 0,
    affinity_threshold: float = 0.0,
) -> NDArray[np.uint64]:
    """Compute merge pairs between two adjacent 2D label faces.

    Returns segment ID pairs that should be merged based on overlap,
    IOU, one-sided containment, and optional affinity validation.

    Parameters
    ----------
    face0, face1 : ndarray, 2D
        Label maps from adjacent faces.
    aff : ndarray, 2D, optional
        Boundary affinity map.
    min_overlap : int
        Minimum overlap in pixels.  Default: 1.
    iou_threshold : float
        Full Jaccard IOU threshold.  0 disables.  Default: 0.0.
    one_sided_threshold : float
        One-sided IOU (overlap / min_size).  0 disables.  Default: 0.9.
    one_sided_min_size : int
        Minimum segment size in the face for one-sided merge.  Default: 0.
    affinity_threshold : float
        Minimum mean boundary affinity.  0 disables.  Default: 0.0.

    Returns
    -------
    ndarray, shape ``(M, 2)``, uint64
        Pairs of segment IDs to merge.
    """
    overlaps = slice_overlaps(face0, face1, aff)
    if len(overlaps) == 0:
        return np.zeros((0, 2), dtype=np.uint64)

    id0 = overlaps[:, 0]
    id1 = overlaps[:, 1]
    size0 = overlaps[:, 2]
    size1 = overlaps[:, 3]
    ovl = overlaps[:, 4]
    has_aff = overlaps.shape[1] > 5

    keep = ovl >= min_overlap

    if iou_threshold > 0:
        union = np.maximum(size0 + size1 - ovl, 1.0)
        iou = ovl / union
        keep &= iou >= iou_threshold

    if one_sided_threshold > 0:
        min_sz = np.minimum(size0, size1)
        one_sided = ovl / np.maximum(min_sz, 1.0)
        mask = one_sided >= one_sided_threshold
        if one_sided_min_size > 0:
            mask &= min_sz >= one_sided_min_size
        keep &= mask

    if has_aff and affinity_threshold > 0:
        keep &= overlaps[:, 5] >= affinity_threshold

    pairs = np.stack([id0[keep], id1[keep]], axis=1)
    return np.ascontiguousarray(pairs.astype(np.uint64))

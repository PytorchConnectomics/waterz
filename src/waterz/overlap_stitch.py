"""Fragment stitching in overlap zones between adjacent chunks.

Provides majority-vote matching to unify fragment IDs across chunk
boundaries when using overlapping chunks for large-volume segmentation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "apply_overlap_remap",
    "build_overlap_remap",
]


def build_overlap_remap(
    overlap_src: NDArray[np.uint64],
    overlap_dst: NDArray[np.uint64],
) -> dict[int, int]:
    """Map dst fragment IDs to src fragment IDs via majority-vote in overlap.

    For each nonzero dst fragment that overlaps a nonzero src fragment,
    the dst ID is mapped to the src ID that it overlaps most (by voxel
    count).  Dst fragments that don't overlap any src fragment, or that
    only overlap background (0), are not included in the mapping.

    Parameters
    ----------
    overlap_src : ndarray, uint64
        Segmentation from the source (lower-index) chunk in the overlap zone.
    overlap_dst : ndarray, uint64
        Segmentation from the destination (higher-index) chunk in the overlap zone.

    Returns
    -------
    dict[int, int]
        Mapping from dst fragment ID to src fragment ID.
    """
    overlap_src = np.asarray(overlap_src, dtype=np.uint64).ravel()
    overlap_dst = np.asarray(overlap_dst, dtype=np.uint64).ravel()

    # Only consider voxels where both src and dst are nonzero
    mask = (overlap_src > 0) & (overlap_dst > 0)
    src_vals = overlap_src[mask]
    dst_vals = overlap_dst[mask]

    if len(src_vals) == 0:
        return {}

    # Pack (dst, src) pairs and count occurrences
    pairs = np.empty(len(dst_vals), dtype=[("dst", np.uint64), ("src", np.uint64)])
    pairs["dst"] = dst_vals
    pairs["src"] = src_vals

    unique_pairs, pair_counts = np.unique(pairs, return_counts=True)

    # For each dst ID, find the src ID with the highest count
    remap: dict[int, int] = {}
    best_count: dict[int, int] = {}

    for i in range(len(unique_pairs)):
        dst_id = int(unique_pairs["dst"][i])
        src_id = int(unique_pairs["src"][i])
        count = int(pair_counts[i])
        if dst_id not in best_count or count > best_count[dst_id]:
            remap[dst_id] = src_id
            best_count[dst_id] = count

    return remap


def apply_overlap_remap(
    seg: NDArray[np.uint64],
    remap: dict[int, int],
) -> NDArray[np.uint64]:
    """Apply fragment ID remapping to a segmentation volume.

    Remaps segment IDs in *seg* according to *remap*.  IDs not in
    *remap* are left unchanged.  Operates in-place when possible
    via vectorized indexing.

    Parameters
    ----------
    seg : ndarray, uint64
        Segmentation volume (modified in-place).
    remap : dict[int, int]
        Mapping from old ID to new ID.

    Returns
    -------
    ndarray, uint64
        The same array, modified in-place.
    """
    if not remap:
        return seg

    seg = np.asarray(seg, dtype=np.uint64)

    # Build a lookup table for fast vectorized remapping
    max_id = int(seg.max())
    remap_max = max(max(remap.keys()), max(remap.values())) if remap else 0
    lut_size = max(max_id, remap_max) + 1

    lut = np.arange(lut_size, dtype=np.uint64)
    for old_id, new_id in remap.items():
        if old_id < lut_size:
            lut[old_id] = new_id

    # Apply via indexing
    flat = seg.ravel()
    flat[:] = lut[flat]
    return seg

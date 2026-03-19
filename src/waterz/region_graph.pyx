# cython: language_level=3
"""Cython bindings for union-find merge_id operations on region graphs.

Ported from waterz_dw (donglaiw/waterz). Four merge modes:
  1. merge_id(id1, id2)                    — merge by ID only
  2. merge_id(id1, id2, score=, aff_thres=)   — merge by ID + affinity
  3. merge_id(id1, id2, count=, count_thres=)  — merge by ID + size
  4. merge_id(id1, id2, score=, count=, ...)   — merge by ID + affinity + size
"""

from libc.stdint cimport uint64_t
import numpy as np
cimport numpy as np


cdef extern from "frontend_region_graph.h":
    ctypedef uint64_t SegID
    ctypedef float    AffValue

    void cpp_merge_id(
        const SegID* id1, const SegID* id2,
        SegID* mapping, SegID num_edge, SegID num_id)

    void cpp_merge_id_by_aff(
        const SegID* id1, const SegID* id2,
        SegID* mapping, const AffValue* score,
        SegID num_edge, SegID num_id, AffValue aff_thres)

    void cpp_merge_id_by_count(
        const SegID* id1, const SegID* id2,
        SegID* mapping, SegID* count,
        SegID num_edge, SegID num_id,
        SegID count_thres, SegID dust_thres)

    void cpp_merge_id_by_aff_count(
        const SegID* id1, const SegID* id2,
        SegID* mapping, const AffValue* score, SegID* count,
        SegID num_edge, SegID num_id,
        AffValue aff_thres, SegID count_thres, SegID dust_thres)

    void cpp_remove_small(
        SegID* mapping, const SegID* count,
        SegID num_id, SegID dust_thres)


def merge_id(id1, id2, score=None, count=None,
             aff_thres=1.0, count_thres=50, dust_thres=50):
    """Merge segment IDs via union-find on a region graph edge list.

    Parameters
    ----------
    id1, id2 : ndarray, uint64, shape (E,)
        Edge endpoint arrays.
    score : ndarray, float32, shape (E,), optional
        Affinity scores per edge.  Lower = more similar.
    count : ndarray, uint64, shape (M,), optional
        Voxel count per segment ID.  ``count[i]`` = size of segment *i*.
    aff_thres : float
        Max affinity for merge (edges with score > aff_thres are skipped).
    count_thres : int
        Size threshold: merge only if at least one side < count_thres.
    dust_thres : int
        Remove segments smaller than this after merging.

    Returns
    -------
    mapping : ndarray, uint64, shape (M,)
        ``mapping[old_id] = new_id``.  Apply with ``seg = mapping[seg]``.
    """
    cdef np.ndarray[uint64_t, ndim=1] _id1
    cdef np.ndarray[uint64_t, ndim=1] _id2
    cdef np.ndarray[uint64_t, ndim=1] _mapping
    cdef np.ndarray[float, ndim=1] _score
    cdef np.ndarray[uint64_t, ndim=1] _count

    _id1 = np.ascontiguousarray(id1, dtype=np.uint64)
    _id2 = np.ascontiguousarray(id2, dtype=np.uint64)

    cdef uint64_t num_edge = len(_id1)

    # Determine mapping size
    cdef uint64_t mid = 0
    if num_edge > 0:
        mid = max(int(_id1.max()), int(_id2.max())) + 1
    if count is not None:
        _count = np.ascontiguousarray(count, dtype=np.uint64)
        mid = max(mid, <uint64_t>len(_count))
    else:
        _count = np.empty(0, dtype=np.uint64)

    _mapping = np.arange(mid, dtype=np.uint64)

    if num_edge == 0 and count is not None:
        # No edges, just dust removal
        cpp_remove_small(&_mapping[0], &_count[0], mid, dust_thres)
        return _mapping

    if score is None and count is None:
        # Mode 1: merge by ID only
        cpp_merge_id(&_id1[0], &_id2[0], &_mapping[0], num_edge, mid)

    elif score is not None and count is None:
        # Mode 2: merge by ID + affinity
        _score = np.ascontiguousarray(score, dtype=np.float32)
        cpp_merge_id_by_aff(
            &_id1[0], &_id2[0], &_mapping[0], &_score[0],
            num_edge, mid, aff_thres)

    elif score is None and count is not None:
        # Mode 3: merge by ID + size
        cpp_merge_id_by_count(
            &_id1[0], &_id2[0], &_mapping[0], &_count[0],
            num_edge, mid, count_thres, dust_thres)

    else:
        # Mode 4: merge by ID + affinity + size
        _score = np.ascontiguousarray(score, dtype=np.float32)
        cpp_merge_id_by_aff_count(
            &_id1[0], &_id2[0], &_mapping[0], &_score[0], &_count[0],
            num_edge, mid, aff_thres, count_thres, dust_thres)

    return _mapping

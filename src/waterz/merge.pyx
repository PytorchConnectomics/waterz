# cython: language_level=3
"""Cython bindings for size+affinity segment merge."""

from libc.stdint cimport uint64_t
import numpy as np
cimport numpy as np


cdef extern from "frontend_merge.h":

    ctypedef uint64_t SegID
    ctypedef float    AffValue

    size_t merge_segments(
        size_t depth, size_t height, size_t width,
        SegID*          seg_data,
        size_t          n_edges,
        const AffValue* rg_affs,
        const SegID*    id1,
        const SegID*    id2,
        size_t          counts_len,
        const uint64_t* counts,
        size_t          size_th,
        AffValue        weight_th,
        size_t          dust_th)


def merge(
        np.ndarray[uint64_t, ndim=3] seg not None,
        np.ndarray[float, ndim=1] rg_affs not None,
        np.ndarray[uint64_t, ndim=1] id1 not None,
        np.ndarray[uint64_t, ndim=1] id2 not None,
        np.ndarray[uint64_t, ndim=1] counts not None,
        size_t size_th,
        float weight_th=0.0,
        size_t dust_th=0):
    """Size+affinity merge + dust removal.

    Modifies ``seg`` in-place.

    Parameters
    ----------
    seg : ndarray, uint64, shape (D, H, W)
    rg_affs : ndarray, float32, shape (E,) — sorted descending
    id1, id2 : ndarray, uint64, shape (E,)
    counts : ndarray, uint64, shape (max_id + 1,)
    size_th : int
        Merge if either segment < this many voxels.
    weight_th : float
        Minimum affinity for merge.
    dust_th : int
        Remove segments smaller than this after merge.

    Returns
    -------
    n_segments : int
        Number of segments after merge (excluding background).
    """
    assert rg_affs.shape[0] == id1.shape[0] == id2.shape[0], \
        "Region graph arrays must have same length"

    if not seg.flags['C_CONTIGUOUS']:
        seg = np.ascontiguousarray(seg)
    if not rg_affs.flags['C_CONTIGUOUS']:
        rg_affs = np.ascontiguousarray(rg_affs)
    if not id1.flags['C_CONTIGUOUS']:
        id1 = np.ascontiguousarray(id1)
    if not id2.flags['C_CONTIGUOUS']:
        id2 = np.ascontiguousarray(id2)
    if not counts.flags['C_CONTIGUOUS']:
        counts = np.ascontiguousarray(counts)

    cdef size_t D = seg.shape[0]
    cdef size_t H = seg.shape[1]
    cdef size_t W = seg.shape[2]
    cdef size_t n_edges = rg_affs.shape[0]
    cdef size_t counts_len = counts.shape[0]

    cdef size_t n_segs = merge_segments(
        D, H, W,
        &seg[0, 0, 0],
        n_edges,
        &rg_affs[0],
        &id1[0],
        &id2[0],
        counts_len,
        &counts[0],
        size_th,
        weight_th,
        dust_th)

    return n_segs

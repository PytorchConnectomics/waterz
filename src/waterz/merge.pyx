# cython: language_level=3
"""Cython bindings for size+affinity segment merge."""

from libc.stdint cimport uint8_t, uint32_t, uint64_t
import numpy as np
cimport numpy as np


cdef extern from "frontend_merge.h":

    size_t merge_segments_impl[SegID, AffValue](
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


def merge(seg not None, rg_affs not None, id1 not None, id2 not None,
          counts not None, size_t size_th, float weight_th=0.0, size_t dust_th=0):
    """Size+affinity merge + dust removal.

    Modifies ``seg`` in-place.  Accepts uint32 or uint64 segment IDs
    and float32 or uint8 affinities.

    Parameters
    ----------
    seg : ndarray, uint32 or uint64, shape (D, H, W)
    rg_affs : ndarray, float32 or uint8, shape (E,) — sorted descending
    id1, id2 : ndarray, same dtype as seg, shape (E,)
    counts : ndarray, uint64, shape (max_id + 1,)
    size_th : int
    weight_th : float
    dust_th : int

    Returns
    -------
    n_segments : int
    """
    assert rg_affs.shape[0] == id1.shape[0] == id2.shape[0], \
        "Region graph arrays must have same length"

    seg = np.ascontiguousarray(seg)
    rg_affs = np.ascontiguousarray(rg_affs)
    id1 = np.ascontiguousarray(id1)
    id2 = np.ascontiguousarray(id2)
    counts = np.ascontiguousarray(counts, dtype=np.uint64)

    cdef size_t D = seg.shape[0]
    cdef size_t H = seg.shape[1]
    cdef size_t W = seg.shape[2]
    cdef size_t n_edges = rg_affs.shape[0]
    cdef size_t counts_len = counts.shape[0]

    # Dispatch to typed C++ template instantiation
    seg_dtype = seg.dtype
    aff_dtype = rg_affs.dtype

    if seg_dtype == np.uint64 and aff_dtype == np.float32:
        return _merge_u64_f32(seg, rg_affs, id1, id2, counts,
                              D, H, W, n_edges, counts_len, size_th, weight_th, dust_th)
    elif seg_dtype == np.uint32 and aff_dtype == np.float32:
        return _merge_u32_f32(seg, rg_affs, id1, id2, counts,
                              D, H, W, n_edges, counts_len, size_th, weight_th, dust_th)
    elif seg_dtype == np.uint64 and aff_dtype == np.uint8:
        return _merge_u64_u8(seg, rg_affs, id1, id2, counts,
                             D, H, W, n_edges, counts_len, size_th, <uint8_t>(<int>weight_th), dust_th)
    elif seg_dtype == np.uint32 and aff_dtype == np.uint8:
        return _merge_u32_u8(seg, rg_affs, id1, id2, counts,
                             D, H, W, n_edges, counts_len, size_th, <uint8_t>(<int>weight_th), dust_th)
    else:
        raise TypeError(
            f"Unsupported dtypes: seg={seg_dtype}, rg_affs={aff_dtype}. "
            "Expected seg: uint32/uint64, rg_affs: float32/uint8."
        )


cdef size_t _merge_u64_f32(
    np.ndarray[uint64_t, ndim=3] seg,
    np.ndarray[float, ndim=1] rg_affs,
    np.ndarray[uint64_t, ndim=1] id1,
    np.ndarray[uint64_t, ndim=1] id2,
    np.ndarray[uint64_t, ndim=1] counts,
    size_t D, size_t H, size_t W, size_t n_edges, size_t counts_len,
    size_t size_th, float weight_th, size_t dust_th):
    return merge_segments_impl[uint64_t, float](
        D, H, W, &seg[0,0,0], n_edges,
        &rg_affs[0], &id1[0], &id2[0],
        counts_len, &counts[0], size_th, weight_th, dust_th)


cdef size_t _merge_u32_f32(
    np.ndarray[uint32_t, ndim=3] seg,
    np.ndarray[float, ndim=1] rg_affs,
    np.ndarray[uint32_t, ndim=1] id1,
    np.ndarray[uint32_t, ndim=1] id2,
    np.ndarray[uint64_t, ndim=1] counts,
    size_t D, size_t H, size_t W, size_t n_edges, size_t counts_len,
    size_t size_th, float weight_th, size_t dust_th):
    return merge_segments_impl[uint32_t, float](
        D, H, W, &seg[0,0,0], n_edges,
        &rg_affs[0], &id1[0], &id2[0],
        counts_len, &counts[0], size_th, weight_th, dust_th)


cdef size_t _merge_u64_u8(
    np.ndarray[uint64_t, ndim=3] seg,
    np.ndarray[uint8_t, ndim=1] rg_affs,
    np.ndarray[uint64_t, ndim=1] id1,
    np.ndarray[uint64_t, ndim=1] id2,
    np.ndarray[uint64_t, ndim=1] counts,
    size_t D, size_t H, size_t W, size_t n_edges, size_t counts_len,
    size_t size_th, uint8_t weight_th, size_t dust_th):
    return merge_segments_impl[uint64_t, uint8_t](
        D, H, W, &seg[0,0,0], n_edges,
        &rg_affs[0], &id1[0], &id2[0],
        counts_len, &counts[0], size_th, weight_th, dust_th)


cdef size_t _merge_u32_u8(
    np.ndarray[uint32_t, ndim=3] seg,
    np.ndarray[uint8_t, ndim=1] rg_affs,
    np.ndarray[uint32_t, ndim=1] id1,
    np.ndarray[uint32_t, ndim=1] id2,
    np.ndarray[uint64_t, ndim=1] counts,
    size_t D, size_t H, size_t W, size_t n_edges, size_t counts_len,
    size_t size_th, uint8_t weight_th, size_t dust_th):
    return merge_segments_impl[uint32_t, uint8_t](
        D, H, W, &seg[0,0,0], n_edges,
        &rg_affs[0], &id1[0], &id2[0],
        counts_len, &counts[0], size_th, weight_th, dust_th)

#ifndef FRONTEND_MERGE_H
#define FRONTEND_MERGE_H

#include <cstddef>
#include <cstdint>

/**
 * Size+affinity merge + dust removal (templated on SegID and AffValue).
 *
 * Pass 1: iterate edges descending by affinity.  Merge if
 *   aff > weight_th AND (size[s1] < size_th OR size[s2] < size_th).
 *
 * Pass 2: discard segments with < dust_th voxels (relabel to 0).
 *
 * Compact-relabels the segmentation volume in-place.
 */
template <typename SegID, typename AffValue>
std::size_t merge_segments_impl(
    std::size_t depth, std::size_t height, std::size_t width,
    SegID*          seg_data,
    std::size_t     n_edges,
    const AffValue* rg_affs,
    const SegID*    id1,
    const SegID*    id2,
    std::size_t     counts_len,
    const uint64_t* counts,
    std::size_t     size_th,
    AffValue        weight_th,
    std::size_t     dust_th);

// Explicit instantiations declared here, defined in .cpp
extern template std::size_t merge_segments_impl<uint64_t, float>(
    std::size_t, std::size_t, std::size_t, uint64_t*, std::size_t,
    const float*, const uint64_t*, const uint64_t*, std::size_t,
    const uint64_t*, std::size_t, float, std::size_t);

extern template std::size_t merge_segments_impl<uint32_t, float>(
    std::size_t, std::size_t, std::size_t, uint32_t*, std::size_t,
    const float*, const uint32_t*, const uint32_t*, std::size_t,
    const uint64_t*, std::size_t, float, std::size_t);

extern template std::size_t merge_segments_impl<uint64_t, uint8_t>(
    std::size_t, std::size_t, std::size_t, uint64_t*, std::size_t,
    const uint8_t*, const uint64_t*, const uint64_t*, std::size_t,
    const uint64_t*, std::size_t, uint8_t, std::size_t);

extern template std::size_t merge_segments_impl<uint32_t, uint8_t>(
    std::size_t, std::size_t, std::size_t, uint32_t*, std::size_t,
    const uint8_t*, const uint32_t*, const uint32_t*, std::size_t,
    const uint64_t*, std::size_t, uint8_t, std::size_t);

#endif

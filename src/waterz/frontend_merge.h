#ifndef FRONTEND_MERGE_H
#define FRONTEND_MERGE_H

#include <cstddef>
#include <cstdint>

typedef uint64_t SegID;
typedef float    AffValue;

/**
 * Size+affinity merge + dust removal.
 *
 * Pass 1: iterate edges descending by affinity.  Merge if
 *   aff > weight_th AND (size[s1] < size_th OR size[s2] < size_th).
 *
 * Pass 2: discard segments with < dust_th voxels (relabel to 0).
 *
 * Compact-relabels the segmentation volume in-place.
 *
 * @param counts_len  length of counts array (max_seg_id + 1)
 * @returns           new number of segments (excluding background)
 */
std::size_t merge_segments(
    std::size_t depth,
    std::size_t height,
    std::size_t width,
    SegID*          seg_data,     // modified in-place
    std::size_t     n_edges,
    const AffValue* rg_affs,      // sorted descending
    const SegID*    id1,
    const SegID*    id2,
    std::size_t     counts_len,
    const uint64_t* counts,       // counts[id] = voxel count
    std::size_t     size_th,
    AffValue        weight_th,
    std::size_t     dust_th);

#endif

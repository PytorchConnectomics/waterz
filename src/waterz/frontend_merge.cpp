/**
 * Size+affinity merge + dust removal.
 *
 * Ported from zwatershed's agglomeration.hpp.
 * Works on C-contiguous arrays (depth, height, width).
 */

#include "frontend_merge.h"
#include "backend/DisjointSets.hpp"

#include <cstdint>
#include <iostream>
#include <vector>

template <typename SegID, typename AffValue>
std::size_t merge_segments_impl(
    std::size_t D, std::size_t H, std::size_t W,
    SegID*          seg,
    std::size_t     n_edges,
    const AffValue* rg_affs,
    const SegID*    id1,
    const SegID*    id2,
    std::size_t     counts_len,
    const uint64_t* counts,
    std::size_t     size_th,
    AffValue        weight_th,
    std::size_t     dust_th)
{
    DisjointSets sets(counts_len);
    for (std::size_t i = 0; i < counts_len; ++i)
        sets.set_size(i, static_cast<std::size_t>(counts[i]));

    // --- Pass 1: size + affinity merge ---
    std::size_t merge_count = 0;
    for (std::size_t i = 0; i < n_edges; ++i) {
        AffValue w = rg_affs[i];
        if (w <= weight_th)
            break;  // sorted descending — rest are weaker

        std::size_t raw1 = static_cast<std::size_t>(id1[i]);
        std::size_t raw2 = static_cast<std::size_t>(id2[i]);
        if (raw1 >= counts_len || raw2 >= counts_len)
            continue;

        std::size_t s1 = sets.find(raw1);
        std::size_t s2 = sets.find(raw2);

        if (s1 == s2 || s1 == 0 || s2 == 0)
            continue;

        if (sets.size_of(s1) < size_th || sets.size_of(s2) < size_th) {
            sets.join(s1, s2);
            merge_count++;
        }
    }

    // --- Pass 2: compact relabel + dust removal ---
    std::vector<SegID> remap(counts_len, 0);
    SegID next_id = 1;

    for (std::size_t old_id = 1; old_id < counts_len; ++old_id) {
        std::size_t root = sets.find(old_id);
        if (root == 0) continue;
        if (remap[root] == 0) {
            std::size_t sz = sets.size_of(root);
            if (dust_th > 0 && sz < dust_th)
                continue;
            remap[root] = next_id;
            next_id++;
        }
        remap[old_id] = remap[root];
    }

    std::size_t vol = D * H * W;
    for (std::size_t i = 0; i < vol; ++i) {
        SegID sid = seg[i];
        seg[i] = (sid < counts_len) ? remap[sid] : 0;
    }

    std::size_t n_segs = static_cast<std::size_t>(next_id - 1);
    std::cout << "dust_merge: merged " << merge_count
              << " edges, " << n_segs << " segments remain" << std::endl;

    return n_segs;
}

// Explicit instantiations
template std::size_t merge_segments_impl<uint64_t, float>(
    std::size_t, std::size_t, std::size_t, uint64_t*, std::size_t,
    const float*, const uint64_t*, const uint64_t*, std::size_t,
    const uint64_t*, std::size_t, float, std::size_t);

template std::size_t merge_segments_impl<uint32_t, float>(
    std::size_t, std::size_t, std::size_t, uint32_t*, std::size_t,
    const float*, const uint32_t*, const uint32_t*, std::size_t,
    const uint64_t*, std::size_t, float, std::size_t);

template std::size_t merge_segments_impl<uint64_t, uint8_t>(
    std::size_t, std::size_t, std::size_t, uint64_t*, std::size_t,
    const uint8_t*, const uint64_t*, const uint64_t*, std::size_t,
    const uint64_t*, std::size_t, uint8_t, std::size_t);

template std::size_t merge_segments_impl<uint32_t, uint8_t>(
    std::size_t, std::size_t, std::size_t, uint32_t*, std::size_t,
    const uint8_t*, const uint32_t*, const uint32_t*, std::size_t,
    const uint64_t*, std::size_t, uint8_t, std::size_t);

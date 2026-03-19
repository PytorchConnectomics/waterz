/**
 * Union-find merge operations on region graph edge lists.
 *
 * Ported from waterz_dw (donglaiw/waterz) frontend_region_graph.cpp.
 * Adapted for uint64_t SegID and float AffValue (new waterz types).
 */

#include "frontend_region_graph.h"

// ------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------

static inline SegID find_root(SegID* mapping, SegID x) {
    SegID r = x;
    while (mapping[r] != r)
        r = mapping[r];
    // Path compression
    while (mapping[x] != r) {
        SegID nxt = mapping[x];
        mapping[x] = r;
        x = nxt;
    }
    return r;
}

void consolidate_mapping(SegID* mapping, SegID num_id) {
    for (SegID i = 1; i < num_id; ++i) {
        mapping[i] = find_root(mapping, i);
    }
}

void cpp_remove_small(
    SegID*       mapping,
    const SegID* count,
    SegID        num_id,
    SegID        dust_thres)
{
    for (SegID i = 1; i < num_id; ++i) {
        SegID s = mapping[i];
        if (count[s] < dust_thres) {
            mapping[i] = 0;
        }
    }
}

// ------------------------------------------------------------------
// Mode 1: merge by ID only
// ------------------------------------------------------------------

void cpp_merge_id(
    const SegID* id1,
    const SegID* id2,
    SegID*       mapping,
    SegID        num_edge,
    SegID        num_id)
{
    for (SegID i = 0; i < num_edge; ++i) {
        if (id1[i] == id2[i]) continue;

        SegID s1 = find_root(mapping, id1[i]);
        SegID s2 = find_root(mapping, id2[i]);
        if (s1 == s2) continue;

        // Merge larger ID into smaller
        if (s1 > s2)
            mapping[s1] = s2;
        else
            mapping[s2] = s1;
    }
    consolidate_mapping(mapping, num_id);
}

// ------------------------------------------------------------------
// Mode 2: merge by ID + affinity threshold
// ------------------------------------------------------------------

void cpp_merge_id_by_aff(
    const SegID*     id1,
    const SegID*     id2,
    SegID*           mapping,
    const AffValue*  score,
    SegID            num_edge,
    SegID            num_id,
    AffValue         aff_thres)
{
    for (SegID i = 0; i < num_edge; ++i) {
        if (id1[i] == id2[i] || score[i] > aff_thres) continue;

        SegID s1 = find_root(mapping, id1[i]);
        SegID s2 = find_root(mapping, id2[i]);
        if (s1 == s2) continue;

        if (s1 > s2)
            mapping[s1] = s2;
        else
            mapping[s2] = s1;
    }
    consolidate_mapping(mapping, num_id);
}

// ------------------------------------------------------------------
// Mode 3: merge by ID + size constraint
// ------------------------------------------------------------------

void cpp_merge_id_by_count(
    const SegID* id1,
    const SegID* id2,
    SegID*       mapping,
    SegID*       count,
    SegID        num_edge,
    SegID        num_id,
    SegID        count_thres,
    SegID        dust_thres)
{
    for (SegID i = 0; i < num_edge; ++i) {
        if (id1[i] == id2[i]) continue;

        SegID s1 = find_root(mapping, id1[i]);
        SegID s2 = find_root(mapping, id2[i]);
        // Skip if same root or both sides are large
        if (s1 == s2 || (count[s1] >= count_thres && count[s2] >= count_thres))
            continue;

        if (s1 > s2) {
            mapping[s1] = s2;
            count[s2] += count[s1];
            count[s1] = 0;
        } else {
            mapping[s2] = s1;
            count[s1] += count[s2];
            count[s2] = 0;
        }
    }
    consolidate_mapping(mapping, num_id);
    cpp_remove_small(mapping, count, num_id, dust_thres);
}

// ------------------------------------------------------------------
// Mode 4: merge by ID + affinity + size
// ------------------------------------------------------------------

void cpp_merge_id_by_aff_count(
    const SegID*     id1,
    const SegID*     id2,
    SegID*           mapping,
    const AffValue*  score,
    SegID*           count,
    SegID            num_edge,
    SegID            num_id,
    AffValue         aff_thres,
    SegID            count_thres,
    SegID            dust_thres)
{
    for (SegID i = 0; i < num_edge; ++i) {
        if (id1[i] == id2[i] || score[i] > aff_thres) continue;

        SegID s1 = find_root(mapping, id1[i]);
        SegID s2 = find_root(mapping, id2[i]);
        if (s1 == s2 || (count[s1] >= count_thres && count[s2] >= count_thres))
            continue;

        if (s1 > s2) {
            mapping[s1] = s2;
            count[s2] += count[s1];
            count[s1] = 0;
        } else {
            mapping[s2] = s1;
            count[s1] += count[s2];
            count[s2] = 0;
        }
    }
    consolidate_mapping(mapping, num_id);
    cpp_remove_small(mapping, count, num_id, dust_thres);
}

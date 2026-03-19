#ifndef FRONTEND_REGION_GRAPH_H
#define FRONTEND_REGION_GRAPH_H

#include <cstddef>
#include <cstdint>

/**
 * Union-find merge operations on region graph edge lists.
 *
 * Ported from waterz_dw (donglaiw/waterz). All functions operate on
 * flat arrays and produce a mapping array: mapping[old_id] = new_id.
 *
 * Four merge modes based on which optional arrays are provided:
 *   1. merge_id:             merge by ID only
 *   2. merge_id_by_aff:      merge by ID + affinity threshold
 *   3. merge_id_by_count:    merge by ID + size constraint
 *   4. merge_id_by_aff_count: merge by ID + affinity + size
 */

typedef uint64_t SegID;
typedef float    AffValue;

/* Consolidate mapping: chase parent pointers to root for every entry. */
void consolidate_mapping(SegID* mapping, SegID num_id);

/* Merge all edges unconditionally via union-find. */
void cpp_merge_id(
    const SegID* id1,
    const SegID* id2,
    SegID*       mapping,
    SegID        num_edge,
    SegID        num_id);

/* Merge edges where affinity <= aff_thres (lower = more similar). */
void cpp_merge_id_by_aff(
    const SegID*     id1,
    const SegID*     id2,
    SegID*           mapping,
    const AffValue*  score,
    SegID            num_edge,
    SegID            num_id,
    AffValue         aff_thres);

/* Merge edges where at least one side has count < count_thres.
   Then remove segments with count < dust_thres. */
void cpp_merge_id_by_count(
    const SegID* id1,
    const SegID* id2,
    SegID*       mapping,
    SegID*       count,
    SegID        num_edge,
    SegID        num_id,
    SegID        count_thres,
    SegID        dust_thres);

/* Merge edges where affinity <= aff_thres AND at least one side < count_thres.
   Then remove segments with count < dust_thres. */
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
    SegID            dust_thres);

/* Set mapping[i] = 0 for segments with count < dust_thres. */
void cpp_remove_small(
    SegID*  mapping,
    const SegID*  count,
    SegID   num_id,
    SegID   dust_thres);

#endif

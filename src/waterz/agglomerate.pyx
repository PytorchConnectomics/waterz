# cython: language_level=3
from libcpp.vector cimport vector
from libc.stdint cimport uint64_t, uint32_t, uint8_t
from libcpp cimport bool
import numpy as np
cimport numpy as np

# AffValue is typedef'd in AffType.h (JIT-generated: float or uint8_t).
# For the default float32 path, np.float32_t is used for typed arrays.
# The _agglomerate.py JIT helper patches these declarations to np.uint8_t
# when compiling the uint8 variant.

def agglomerate(
        affs,
        thresholds,
        gt=None,
        fragments=None,
        aff_threshold_low=0.0001,
        aff_threshold_high=0.9999,
        return_merge_history=False,
        return_region_graph=False,
        rescore_region_graph=True):

    if not affs.flags['C_CONTIGUOUS']:
        affs = np.ascontiguousarray(affs)
    if gt is not None and not gt.flags['C_CONTIGUOUS']:
        gt = np.ascontiguousarray(gt)
    if fragments is not None and not fragments.flags['C_CONTIGUOUS']:
        fragments = np.ascontiguousarray(fragments)

    print("Preparing segmentation volume...")

    if fragments is None:
        volume_shape = (affs.shape[1], affs.shape[2], affs.shape[3])
        segmentation = np.zeros(volume_shape, dtype=np.uint64)
        find_fragments = True
    else:
        segmentation = fragments
        find_fragments = False

    cdef WaterzState state = __initialize(affs, segmentation, gt, aff_threshold_low, aff_threshold_high, find_fragments)

    thresholds.sort()
    for threshold in thresholds:

        merge_history = mergeUntil(state, threshold)

        result = (segmentation,)

        if gt is not None:
            stats = {}
            stats['V_Rand_split'] = state.metrics.rand_split
            stats['V_Rand_merge'] = state.metrics.rand_merge
            stats['V_Info_split'] = state.metrics.voi_split
            stats['V_Info_merge'] = state.metrics.voi_merge
            result += (stats,)

        if return_merge_history:
            result += (merge_history,)

        if return_region_graph:
            result += (getRegionGraph(state, rescore_region_graph),)

        if len(result) == 1:
            yield result[0]
        else:
            yield result

    free(state)

def __initialize(
        np.ndarray[np.float32_t, ndim=4] affs,
        np.ndarray[uint64_t, ndim=3]     segmentation,
        np.ndarray[uint32_t, ndim=3]     gt = None,
        aff_threshold_low  = 0.0001,
        aff_threshold_high = 0.9999,
        find_fragments = True):

    cdef float*    aff_data
    cdef uint64_t* segmentation_data
    cdef uint32_t* gt_data = NULL

    aff_data = &affs[0,0,0,0]
    segmentation_data = &segmentation[0,0,0]
    if gt is not None:
        gt_data = &gt[0,0,0]

    return initialize(
        affs.shape[1], affs.shape[2], affs.shape[3],
        aff_data,
        segmentation_data,
        gt_data,
        aff_threshold_low,
        aff_threshold_high,
        find_fragments)

cdef extern from "frontend_agglomerate.h":

    struct Metrics:
        double voi_split
        double voi_merge
        double rand_split
        double rand_merge

    struct Merge:
        uint64_t a
        uint64_t b
        uint64_t c
        float score

    struct ScoredEdge:
        uint64_t u
        uint64_t v
        float score

    struct RichScoredEdge:
        uint64_t u
        uint64_t v
        float score
        uint64_t contact_area

    struct WaterzState:
        int     context
        Metrics metrics

    WaterzState initialize(
            size_t          width,
            size_t          height,
            size_t          depth,
            const float*    affinity_data,
            uint64_t*       segmentation_data,
            const uint32_t* groundtruth_data,
            float           affThresholdLow,
            float           affThresholdHigh,
            bool            findFragments);

    vector[Merge] mergeUntil(
            WaterzState& state,
            float        threshold)

    vector[ScoredEdge] getRegionGraph(WaterzState& state, bool rescore)

    vector[ScoredEdge] c_buildRegionGraphOnly "buildRegionGraphOnly" (
            size_t          width,
            size_t          height,
            size_t          depth,
            const float*    affinity_data,
            uint64_t*       segmentation_data)

    vector[RichScoredEdge] c_buildRegionGraphRich "buildRegionGraphRich" (
            size_t          width,
            size_t          height,
            size_t          depth,
            const float*    affinity_data,
            uint64_t*       segmentation_data)

    void free(WaterzState& state)


def buildRegionGraphOnly(
        np.ndarray[np.float32_t, ndim=4] affs,
        np.ndarray[uint64_t, ndim=3] seg):
    """Build scored region graph without RegionMerging overhead."""
    if not affs.flags['C_CONTIGUOUS']:
        affs = np.ascontiguousarray(affs)
    if not seg.flags['C_CONTIGUOUS']:
        seg = np.ascontiguousarray(seg)

    cdef size_t width = affs.shape[1]
    cdef size_t height = affs.shape[2]
    cdef size_t depth = affs.shape[3]
    cdef float* aff_data = &affs[0,0,0,0]
    cdef uint64_t* seg_data = &seg[0,0,0]

    cdef vector[ScoredEdge] edges = c_buildRegionGraphOnly(
        width, height, depth, aff_data, seg_data)

    result = []
    cdef size_t i
    for i in range(edges.size()):
        result.append({
            'u': edges[i].u,
            'v': edges[i].v,
            'score': edges[i].score,
        })
    return result


def buildRegionGraphRich(
        np.ndarray[np.float32_t, ndim=4] affs,
        np.ndarray[uint64_t, ndim=3] seg):
    """Build scored region graph with contact area per edge."""
    if not affs.flags['C_CONTIGUOUS']:
        affs = np.ascontiguousarray(affs)
    if not seg.flags['C_CONTIGUOUS']:
        seg = np.ascontiguousarray(seg)

    cdef size_t width = affs.shape[1]
    cdef size_t height = affs.shape[2]
    cdef size_t depth = affs.shape[3]
    cdef float* aff_data = &affs[0,0,0,0]
    cdef uint64_t* seg_data = &seg[0,0,0]

    cdef vector[RichScoredEdge] edges = c_buildRegionGraphRich(
        width, height, depth, aff_data, seg_data)

    result = []
    cdef size_t i
    for i in range(edges.size()):
        result.append({
            'u': edges[i].u,
            'v': edges[i].v,
            'score': edges[i].score,
            'contact_area': edges[i].contact_area,
        })
    return result

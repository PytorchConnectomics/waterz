from libc.stdint cimport uint8_t, uint64_t, uint32_t
from libcpp cimport bool
import numpy as np
cimport numpy as np

def merge_id(id1, id2, id_thres=0):
    # avoid wrong memory access
    if not id1.flags['C_CONTIGUOUS']:
        id1 = np.ascontiguousarray(id1)
    if not id2.flags['C_CONTIGUOUS']:
        id2 = np.ascontiguousarray(id2)
    if id_thres == 0:
        id_thres = max(id1.max(), id2.max()) + 1
    mapping = np.arange(id_thres).astype(np.uint64)
    __merge_id(id1, id2, mapping, id_thres)
    return mapping 

def __merge_id(np.ndarray[np.uint64_t, ndim=1] id1,
                 np.ndarray[np.uint64_t, ndim=1] id2,
                 np.ndarray[np.uint64_t, ndim=1] mapping, 
                 id_thres):
    '''Find the global mapping of IDs from the region graph without count constraints
    
    The region graph should be ordered by decreasing affinity and truncated
    at the affinity threshold.
    :param id1: a 1D array of the lefthand side of the two adjacent regions
    :param id2: a 1D array of the righthand side of the two adjacent regions
    :returns: a 1D array of the global IDs per local ID
    '''
    cdef uint64_t* id1_data;
    cdef uint64_t* id2_data;
    cdef uint64_t* mapping_data;
    id1_data = &id1[0];
    id2_data = &id2[0];
    mapping_data = &mapping[0];

    cpp_merge_id(id1_data, id2_data, mapping_data, len(id1), len(mapping), id_thres);

cdef extern from "frontend_region_graph.h":
    void cpp_merge_id(
        uint64_t*          id1,
        uint64_t*          id2,
        uint64_t*          mapping,
        uint64_t           num_edge,
        uint64_t           num_id,
        uint64_t           id_thres);

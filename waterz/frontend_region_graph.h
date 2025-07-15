#ifndef C_REGION_H
#define C_REGION_H

#include "frontend_shared.h"

void cpp_merge_id(
    SegID*          id1,
    SegID*          id2,
    SegID*          mapping,
    SegID           num_edge,
    SegID           num_id,
    SegID           id_thres);

#endif

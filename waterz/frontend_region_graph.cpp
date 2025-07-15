#include "frontend_region_graph.h"

void cpp_consolidate_mapping(
     SegID* mapping,
     SegID  num_id){

    SegID s1;
    // skip 0th element: bg seg
    for (size_t i=1; i<num_id; ++i) {
        s1 = mapping[i];
        // root node: mapping to itself
        // is it mapped to the root node
        if (mapping[s1] == s1) continue;

        s1 = mapping[s1];
        while (mapping[s1] != s1){
            s1 = mapping[s1];
        }
        mapping[i] = s1;
    }
}

void cpp_relabel_mapping(
     SegID* mapping,
     SegID  num_id,
     SegID  id_thres){

    SegID s1;
    // relabel ids >=id_thres
    if (id_thres>0){
        for (size_t i=id_thres; i<num_id; ++i) {
            s1 = mapping[i];
            if (s1 != i && s1 < id_thres && s1 != 0){
                 for (size_t j=1; j<num_id; ++j) {
                     if(mapping[j] == s1){
                        mapping[j] = i;
                     }
                 }
            }
        }
    }
}

void cpp_merge_id(
     SegID* id1,
     SegID* id2,
     SegID* mapping,
     SegID  num_edge,
     SegID  num_id,
     SegID  id_thres) {

    // merge root nodes
    SegID s1, s2;
    for (size_t i = 0; i < num_edge; ++i) {
        // in case redundant
        if (id1[i] == id2[i]) continue;
        // find root
        s1 = mapping[id1[i]];
        while (mapping[s1] != s1){
            s1 = mapping[s1];
        }
        s2 = mapping[id2[i]];
        while (mapping[s2] != s2){
            s2 = mapping[s2];
        }
        // compare
        if (s1 == s2) continue;
        if (s1 > s2){
            mapping[s1] = s2;
        } else {
            mapping[s2] = s1;
        }
    }
    cpp_consolidate_mapping(mapping, num_id);
    cpp_relabel_mapping(mapping, num_id, id_thres);
}


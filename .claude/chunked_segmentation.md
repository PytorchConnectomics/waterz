# Chunked Segmentation: Overlapping Region Graph Merge

## Problem

Large EM volumes (e.g. 1040×2066×2066) don't fit in memory for single-pass waterz agglomeration. The current `large_decode.py` splits into non-overlapping chunks, decodes each independently, then stitches borders via single-face IOU matching (`face_merge_pairs`). This is fragile:

- Single face = 1 slice of context for matching
- Segments that are large in 3D but small at the boundary face get poor overlap scores
- No affinity information crosses the chunk boundary during agglomeration

## Solution: Overlapping Chunks + Region Graph Merge

### Pipeline (8 stages)

```
1. fragment_chunk (parallel)  — watershed per chunk WITH overlap (e.g. 8-16 voxels)
2. compute_offsets (serial)   — assign globally unique fragment IDs
3. stitch_overlap (parallel)  — consensus-match fragment IDs in overlap zones
4. build_rg_chunk (parallel)  — build scored region graph per chunk (with contact areas)
5. merge_rg (serial)          — weighted-mean merge of per-chunk region graphs
6. agglomerate (serial)       — threshold merge on global region graph
7. apply_relabel (parallel)   — apply global ID mapping to each chunk
8. assemble_output (serial)   — write final volume
```

### Why This Works

- **Overlap zone** gives N slices for fragment matching instead of 1 face
- **Global RG** means boundary segments get the same scoring as interior ones
- **Weighted-mean merge** properly combines edge scores from overlapping chunks using contact area as weight
- Watershed is cheap (parallel, per-chunk). Agglomeration runs once on the merged RG.

## Implementation

### Phase 1: C++ — Return Contact Area Per Edge

The current `buildRegionGraphOnly` returns `vector<ScoredEdge>` with `(u, v, score)`. We need `contact_area` (number of affinity samples per edge) for proper weighted-mean merging.

**`frontend_agglomerate.h`** — add struct:
```cpp
struct RichScoredEdge {
    SegID u, v;
    ScoreValue score;
    uint64_t contact_area;
};
```

**`region_graph.hpp`** — add overload of `get_region_graph` that outputs edge counts. The existing code collects `vector<F> affinities` per edge pair before feeding to the statistics provider — the `.size()` of each vector is the contact area.

**`frontend_agglomerate.cpp`** — add `buildRegionGraphRich` using the new overload.

**`agglomerate.pyx`** — add Cython bindings for `RichScoredEdge` and `buildRegionGraphRich`.

### Phase 2: Python — Region Graph Merge

**`_merge.py`** — add:
```python
def merge_region_graphs(
    rg_list: list[tuple[NDArray, NDArray, NDArray, NDArray]],
    # Each: (rg_affs, id1, id2, contact_areas)
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Merge multiple region graphs via weighted-mean scoring."""
```

Algorithm:
1. Concatenate all edges from all chunks
2. Canonicalize keys: `(min(u,v), max(u,v))`
3. Group by key via `np.unique`
4. Weighted mean: `score = sum(score_i × area_i) / sum(area_i)`
5. Return merged arrays sorted descending by score

This is pure NumPy. For 1000 chunks × 100K edges each, the merged RG has ~100M edges — grouping via `np.unique` + `np.add.at` handles this.

### Phase 3: Fragment Stitching in Overlap Zone

**New file: `overlap_stitch.py`**

```python
def build_overlap_remap(overlap_src, overlap_dst) -> dict[int, int]:
    """Map dst fragment IDs → src fragment IDs via majority-vote in overlap."""

def apply_overlap_stitch(seg_dst, remap) -> NDArray:
    """Remap dst chunk's fragments to match src in overlap zone."""
```

Convention: the lower-index chunk ("src") owns the overlap. The higher-index chunk ("dst") remaps its fragment IDs to match src's. Fragments spanning from overlap into dst's non-overlap region are consistently remapped.

### Phase 4: Pipeline Integration

**`large_decode.py`** — new config fields:
```python
@dataclass
class LargeDecodeConfig:
    overlap: tuple[int, int, int] = (0, 0, 0)  # overlap per axis in voxels
    # ... existing fields ...
```

New task handlers:
- `handle_fragment_chunk` — watershed only (no agglomeration)
- `handle_stitch_overlap` — fragment consensus matching
- `handle_build_rg_chunk` — `get_region_graph_rich` on stitched fragments
- `handle_merge_rg` — `merge_region_graphs` across all chunks
- `handle_agglomerate` — `merge_segments` on global merged RG

**`large_workflow.py`** — add `build_chunk_grid_overlap` producing overlapping chunk refs.

## Design Decisions

### Why weighted-mean (not histogram merge)?

Three options for combining duplicate edges across chunks:

| Approach | Accuracy | C++ changes | Complexity |
|----------|----------|-------------|------------|
| **(a)** Return full histograms, merge in Python | Best | Major (serialize 256-bin histograms through Cython) | High |
| **(b)** Merge StatisticsProviders in C++ | Best | Major (new C++ entry point, template refactoring) | High |
| **(c)** Weighted-mean by contact area | Good | Minimal (add `contact_area` to ScoredEdge) | Low |

Approach (c) is chosen. The accuracy trade-off is acceptable because:
- Only cross-boundary edges are approximated (interior edges are exact)
- The overlap zone already provides both-sides context
- Edges with large contact areas (reliable scores) dominate the weighted mean

### Why majority-vote stitching (not watershed consensus)?

Running a joint watershed across the overlap zone would be more accurate but requires:
- Loading affinities from both chunks simultaneously
- A special "constrained watershed" that respects non-overlap boundaries

Majority-vote on independently-computed fragments is simpler and works well when overlap ≥ 8 voxels (fragments in the interior of the overlap zone converge to the same boundaries regardless of chunk origin).

## File Changes Summary

| File | Change |
|------|--------|
| `backend/region_graph.hpp` | Add overload with `edge_counts` output |
| `frontend_agglomerate.h` | Add `RichScoredEdge`, `buildRegionGraphRich` declaration |
| `frontend_agglomerate.cpp` | Implement `buildRegionGraphRich` |
| `agglomerate.pyx` | Add Cython bindings for rich RG |
| `_agglomerate.py` | Add `build_region_graph_rich` wrapper |
| `_merge.py` | Add `get_region_graph_rich`, `merge_region_graphs` |
| `overlap_stitch.py` | **New** — fragment consensus in overlap zone |
| `large_decode.py` | Extend config, add overlap pipeline handlers |
| `large_workflow.py` | Add `build_chunk_grid_overlap`, overlap task DAG |
| `__init__.py` | Export new public API |

# waterz

Pronounced *water-zed*. Watershed and region agglomeration for affinity graphs,
with region graph construction, union-find merge, and dust removal.

Based on the watershed implementation of [Aleksandar Zlateski](https://bitbucket.org/poozh/watershed)
and [Chandan Singh](https://github.com/TuragaLab/zwatershed).
Extended with region graph, merge_id, and merge_dust from
[zwatershed](https://github.com/PytorchConnectomics/zwatershed) and
[waterz_dw](https://github.com/donglaiw/waterz).

## Install

```sh
# requires boost headers
sudo apt install libboost-dev  # linux
brew install boost              # macos

pip install -e .
```

## API

### Agglomeration

```python
import waterz
import numpy as np

# affinities: [3, depth, height, width] float32, values in [0, 1]
affs = np.random.rand(3, 100, 256, 256).astype(np.float32)

# Generator — segmentation is modified in-place between yields (copy if needed)
for seg in waterz.agglomerate(affs, thresholds=[0.1, 0.3, 0.5]):
    seg = seg.copy()  # safe to keep

# Convenience wrapper — returns copied segmentations
seg_list = waterz.waterz(affs, thresholds=[0.1, 0.3, 0.5])
seg_dict = waterz.waterz(affs, thresholds=[0.1, 0.3, 0.5], as_dict=True)
```

Scoring functions are specified as C++ type strings:
- `OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>` (default)
- `OneMinus<HistogramQuantileAffinity<RegionGraphType, 85, ScoreValue, 256>>` (p85)
- `OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>` (median)

### uint8 Affinity Mode

Pass uint8 affinities [0, 255] instead of float32 [0, 1] for **4x less
memory** and **~2x better cache performance**.  The entire C++ pipeline
runs in integer arithmetic — no float conversion anywhere.

```python
# Convert once (lossless for 256-bin histogram scoring)
affs_u8 = (affs_f32 * 255).astype(np.uint8)

# Thresholds and aff_threshold in [0, 255] range
for seg in waterz.agglomerate(affs_u8, thresholds=[76],
        aff_threshold_low=1, aff_threshold_high=254):
    seg = seg.copy()
```

How it works:
- `AffValue` and `ScoreValue` are typedef'd as `uint8_t` (via JIT header)
- `discretize(uint8, 256)` is identity — bin = value, no multiply
- `OneMinus<uint8>` computes `255 - x` (not `1.0 - x`)
- Each (dtype, scoring_function) combo gets its own cached compiled module
- For `HistogramQuantileAffinity` with 256 bins, results are identical to float32

### Region Graph

Build a scored region graph from segmentation + affinities using any
waterz scoring function. Skips the agglomeration state machine for speed.

```python
# Any scoring function (JIT-compiled)
rg_affs, id1, id2 = waterz.get_region_graph(
    seg, affs,
    scoring_function="MeanAffinity<RegionGraphType, ScoreValue>",
    channels="all",  # "all", "z", or "xy"
)
# rg_affs: float32 (E,) sorted descending
# id1, id2: uint64 (E,) edge endpoints
```

### Size+Affinity Merge (Dust Cleanup)

Ported from zwatershed's `merge_segments_with_function`. Merges small
segments into their highest-affinity neighbor, then removes remaining dust.

```python
# Step-by-step
rg_affs, id1, id2 = waterz.get_region_graph(seg, affs)
waterz.merge_segments(seg, rg_affs, id1, id2, counts,
                      size_th=800, weight_th=0.3, dust_th=600)

# One-call convenience
waterz.merge_dust(seg, affs, size_th=800, weight_th=0.3, dust_th=600)
```

### Union-Find Merge (merge_id)

Ported from waterz_dw. Four modes based on which optional arrays are provided:

```python
# Mode 1: merge all edges unconditionally
mapping = waterz.merge_id(id1, id2)

# Mode 2: merge edges with affinity <= threshold
mapping = waterz.merge_id(id1, id2, score=affs, aff_thres=0.5)

# Mode 3: merge edges where at least one side < count_thres
mapping = waterz.merge_id(id1, id2, count=sizes, count_thres=100, dust_thres=50)

# Mode 4: merge by affinity + size
mapping = waterz.merge_id(id1, id2, score=affs, count=sizes,
                          aff_thres=0.5, count_thres=100, dust_thres=50)

# Apply: seg = mapping[seg]
```

### Evaluation

```python
metrics = waterz.evaluate(seg, gt)
# Returns dict: V_Rand_split, V_Rand_merge, V_Info_split, V_Info_merge
```

## Architecture

```
src/waterz/
├── _agglomerate.py          # JIT compilation via witty, agglomerate() + buildRegionGraphOnly()
├── _waterz.py               # waterz() convenience wrapper (copies results)
├── _merge.py                # get_region_graph(), merge_segments(), merge_dust()
├── agglomerate.pyx          # Cython bridge to C++ agglomeration engine
├── merge.pyx                # Cython bridge to C++ merge_segments
├── region_graph.pyx         # Cython bridge to C++ merge_id
├── frontend_agglomerate.cpp # C++ agglomeration + buildRegionGraphOnly
├── frontend_merge.cpp       # C++ size+affinity merge + dust removal
├── frontend_region_graph.cpp# C++ union-find merge_id (4 modes)
├── evaluate.pyx             # Cython bridge to C++ Rand/VOI
├── backend/                 # C++ template library
│   ├── basic_watershed.hpp  # BFS watershed on affinity graph
│   ├── region_graph.hpp     # Extract RAG from segmentation + affinities
│   ├── IterativeRegionMerging.hpp  # Priority-queue agglomeration
│   ├── MergeFunctions.hpp   # Scoring: MinSize, MaxSize, MeanAffinity, etc.
│   ├── Operators.hpp        # OneMinus, Multiply, Add, etc.
│   ├── HistogramQuantileProvider.hpp  # Histogram-based quantile scoring
│   ├── DisjointSets.hpp     # Union-find with path compression
│   └── ...
```

## Dependencies

- numpy >= 1.20
- witty[cython] >= 0.3.1 (JIT Cython compilation for scoring functions)
- Boost C++ headers (for multi_array)
- Cython >= 0.29 (build-time)

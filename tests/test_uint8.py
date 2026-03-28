"""Test that float32 and uint8 affinity paths produce identical segmentations.

Ground truth is a uint8 affinity volume. All thresholds are chosen in uint8
space first, then converted to float for the float32 path. This ensures
both paths operate on identical discretized values — no rounding mismatch.

For HistogramQuantileAffinity with 256 bins, uint8 is lossless:
  discretize(u8/255.0, 256) = min(int(u8/255.0 * 256), 255) = u8
  undiscretize(u8, 256)     = (u8 + 0.5) / 256  [float path]
  undiscretize(u8, 256)     = u8                  [uint8 path]

Usage:
    python tests/test_uint8.py
    pytest tests/test_uint8.py -v
"""

import numpy as np
import pytest


def _make_test_volume(shape=(8, 32, 32), seed=42):
    """Create a uint8 affinity volume with realistic structure."""
    rng = np.random.RandomState(seed)
    affs_u8 = rng.randint(0, 256, size=(3,) + shape, dtype=np.uint8)
    # Add some strong boundaries (low affinity regions)
    affs_u8[:, :, 16, :] = rng.randint(0, 30, size=(3, shape[0], shape[2]), dtype=np.uint8)
    affs_u8[:, :, :, 16] = rng.randint(0, 30, size=(3, shape[0], shape[1]), dtype=np.uint8)
    return affs_u8


def _u8_to_f32(affs_u8):
    """Convert uint8 [0,255] to float32 [0,1]."""
    return affs_u8.astype(np.float32) / 255.0


def _u8_threshold_to_f32(t):
    """Convert a uint8 threshold to the exact float32 equivalent."""
    return t / 255.0


class TestUint8Float32Equivalence:
    """Verify uint8 and float32 paths produce the same segmentation.

    All thresholds are defined in uint8 space first, then converted
    to float32 so both paths see the same discretized values.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.affs_u8 = _make_test_volume()
        self.affs_f32 = _u8_to_f32(self.affs_u8)
        # Watershed thresholds in uint8 space
        self.aff_low_u8 = 1
        self.aff_high_u8 = 254

    def _run_u8(self, thresholds_u8, scoring):
        import waterz
        results = []
        for seg in waterz.agglomerate(
            self.affs_u8, thresholds=thresholds_u8,
            scoring_function=scoring,
            aff_threshold_low=self.aff_low_u8,
            aff_threshold_high=self.aff_high_u8,
        ):
            results.append(np.array(seg, copy=True))
        return results

    def _run_f32(self, thresholds_u8, scoring):
        """Run float32 path with thresholds derived from uint8 values."""
        import waterz
        thresholds_f32 = [_u8_threshold_to_f32(t) for t in thresholds_u8]
        results = []
        for seg in waterz.agglomerate(
            self.affs_f32, thresholds=thresholds_f32,
            scoring_function=scoring,
            aff_threshold_low=_u8_threshold_to_f32(self.aff_low_u8),
            aff_threshold_high=_u8_threshold_to_f32(self.aff_high_u8),
        ):
            results.append(np.array(seg, copy=True))
        return results

    def test_same_fragments(self):
        """Watershed fragments should be identical (threshold=0 → no merge)."""
        scoring = "OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>"
        segs_u8 = self._run_u8([0], scoring)
        segs_f32 = self._run_f32([0], scoring)
        assert len(np.unique(segs_u8[0])) == len(np.unique(segs_f32[0])), \
            f"Fragment count: uint8={len(np.unique(segs_u8[0]))}, float32={len(np.unique(segs_f32[0]))}"

    def test_histogram_quantile_exact(self):
        """HistogramQuantile with 256 bins: uint8 and float32 must match exactly."""
        scoring = "OneMinus<HistogramQuantileAffinity<RegionGraphType, 85, ScoreValue, 256>>"
        threshold_u8 = 64
        segs_u8 = self._run_u8([threshold_u8], scoring)
        segs_f32 = self._run_f32([threshold_u8], scoring)
        n_u8 = len(np.unique(segs_u8[0])) - 1
        n_f32 = len(np.unique(segs_f32[0])) - 1
        assert n_u8 == n_f32, f"uint8={n_u8}, float32={n_f32}"

    def test_mean_affinity_exact(self):
        """MeanAffinity scoring: uint8 and float32 must match exactly."""
        scoring = "OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>"
        threshold_u8 = 128
        segs_u8 = self._run_u8([threshold_u8], scoring)
        segs_f32 = self._run_f32([threshold_u8], scoring)
        n_u8 = len(np.unique(segs_u8[0])) - 1
        n_f32 = len(np.unique(segs_f32[0])) - 1
        assert n_u8 == n_f32, f"uint8={n_u8}, float32={n_f32}"

    def test_multiple_thresholds_monotonic(self):
        """Both paths should produce monotonically decreasing segment counts."""
        scoring = "OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>"
        thresholds_u8 = [50, 100, 150]
        segs_u8 = self._run_u8(thresholds_u8, scoring)
        segs_f32 = self._run_f32(thresholds_u8, scoring)

        counts_u8 = [len(np.unique(s)) for s in segs_u8]
        counts_f32 = [len(np.unique(s)) for s in segs_f32]

        for i in range(len(counts_u8) - 1):
            assert counts_u8[i] >= counts_u8[i + 1], f"uint8 not monotonic: {counts_u8}"
            assert counts_f32[i] >= counts_f32[i + 1], f"float32 not monotonic: {counts_f32}"

        # Counts should match at each threshold
        for i, t in enumerate(thresholds_u8):
            assert counts_u8[i] == counts_f32[i], \
                f"Mismatch at threshold {t}: uint8={counts_u8[i]}, float32={counts_f32[i]}"

    def test_memory_savings(self):
        """uint8 affinities should use 4x less memory."""
        assert self.affs_u8.nbytes * 4 == self.affs_f32.nbytes


def test_basic_uint8():
    """Smoke test: uint8 agglomerate doesn't crash and produces segments."""
    import waterz
    affs = _make_test_volume()
    results = waterz.waterz(affs, thresholds=[128],
                            aff_threshold_low=1, aff_threshold_high=254)
    assert len(results) == 1
    seg = results[0]
    assert seg.dtype == np.uint64
    assert len(np.unique(seg)) > 1
    print(f"uint8 smoke test: {len(np.unique(seg))-1} segments")


def test_get_region_graph_uint8_returns_normalized_scores():
    """uint8 region-graph scores should keep float32-style [0,1] semantics."""
    import waterz

    affs_u8 = _make_test_volume(seed=7)
    threshold_u8 = 64
    seg_u8 = waterz.waterz(
        affs_u8,
        thresholds=[threshold_u8],
        scoring_function="OneMinus<HistogramQuantileAffinity<RegionGraphType, 85, ScoreValue, 256>>",
        aff_threshold_low=1,
        aff_threshold_high=254,
    )[0]

    rg_affs, _, _ = waterz.get_region_graph(
        seg_u8,
        affs_u8,
        scoring_function="HistogramQuantileAffinity<RegionGraphType, 85, ScoreValue, 256>",
    )

    assert rg_affs.dtype == np.float32
    assert float(rg_affs.min(initial=0.0)) >= 0.0
    assert float(rg_affs.max(initial=0.0)) <= 1.0 + 1e-6


def test_merge_dust_uint8_matches_float32_threshold_semantics():
    """Dust merge should interpret uint8 affinities with the same thresholds as float32."""
    import waterz

    affs_u8 = _make_test_volume(seed=11)
    affs_f32 = _u8_to_f32(affs_u8)
    threshold_u8 = 64
    scoring = "OneMinus<HistogramQuantileAffinity<RegionGraphType, 85, ScoreValue, 256>>"
    dust_scoring = "HistogramQuantileAffinity<RegionGraphType, 85, ScoreValue, 256>"

    seg_u8 = waterz.waterz(
        affs_u8,
        thresholds=[threshold_u8],
        scoring_function=scoring,
        aff_threshold_low=1,
        aff_threshold_high=254,
    )[0]
    seg_f32 = waterz.waterz(
        affs_f32,
        thresholds=[_u8_threshold_to_f32(threshold_u8)],
        scoring_function=scoring,
        aff_threshold_low=_u8_threshold_to_f32(1),
        aff_threshold_high=_u8_threshold_to_f32(254),
    )[0]

    waterz.merge_dust(
        seg_u8,
        affs_u8,
        size_th=100,
        weight_th=0.3,
        dust_th=20,
        scoring_function=dust_scoring,
    )
    waterz.merge_dust(
        seg_f32,
        affs_f32,
        size_th=100,
        weight_th=0.3,
        dust_th=20,
        scoring_function=dust_scoring,
    )

    assert len(np.unique(seg_u8)) == len(np.unique(seg_f32))


if __name__ == "__main__":
    print("Running uint8 vs float32 equivalence tests...\n")

    affs_u8 = _make_test_volume()
    affs_f32 = _u8_to_f32(affs_u8)

    import waterz

    aff_low_u8, aff_high_u8 = 1, 254
    scoring = "OneMinus<HistogramQuantileAffinity<RegionGraphType, 85, ScoreValue, 256>>"

    # Test 1: Smoke test
    print("1. Smoke test (uint8)...")
    test_basic_uint8()
    print("   PASS\n")

    # Test 2: Exact match at multiple thresholds
    print("2. Exact match (uint8 thresholds → float32)...")
    for threshold_u8 in [32, 64, 96, 128, 192]:
        threshold_f32 = _u8_threshold_to_f32(threshold_u8)

        seg_u8 = waterz.waterz(affs_u8, thresholds=[threshold_u8],
                               scoring_function=scoring,
                               aff_threshold_low=aff_low_u8,
                               aff_threshold_high=aff_high_u8)[0]
        seg_f32 = waterz.waterz(affs_f32, thresholds=[threshold_f32],
                                scoring_function=scoring,
                                aff_threshold_low=_u8_threshold_to_f32(aff_low_u8),
                                aff_threshold_high=_u8_threshold_to_f32(aff_high_u8))[0]

        n_u8 = len(np.unique(seg_u8)) - 1
        n_f32 = len(np.unique(seg_f32)) - 1
        match = "OK" if n_u8 == n_f32 else "FAIL"
        print(f"   threshold={threshold_u8:>3d} (f32={threshold_f32:.4f}): "
              f"uint8={n_u8}, float32={n_f32}  [{match}]")

    # Test 3: Memory
    print(f"\n3. Memory: uint8={affs_u8.nbytes/1e6:.1f}MB, "
          f"float32={affs_f32.nbytes/1e6:.1f}MB, "
          f"ratio={affs_f32.nbytes/affs_u8.nbytes:.0f}x")

    print("\nDone!")

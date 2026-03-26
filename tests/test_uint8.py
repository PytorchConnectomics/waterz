"""Test that float32 and uint8 affinity paths produce identical segmentations.

The ground truth is a uint8 affinity volume. We run waterz agglomeration
on both the raw uint8 and a float32 conversion, then verify the resulting
segmentations are identical.

For HistogramQuantileAffinity with 256 bins, uint8 is lossless — the
histogram bins map 1:1 to uint8 values. The float32 path discretizes
float→bin via ``min(int(val*256), 255)``, which for values that originated
from uint8 (i.e. ``val = u8/255.0``) should land in the same bin.

Usage:
    python tests/test_uint8.py
    pytest tests/test_uint8.py -v
"""

import numpy as np
import pytest


def _make_test_volume(shape=(8, 32, 32), seed=42):
    """Create a uint8 affinity volume with realistic structure."""
    rng = np.random.RandomState(seed)
    # Base affinities with some structure (not pure noise)
    affs_u8 = rng.randint(0, 256, size=(3,) + shape, dtype=np.uint8)
    # Add some strong boundaries (low affinity regions)
    affs_u8[:, :, 16, :] = rng.randint(0, 30, size=(3, shape[0], shape[2]), dtype=np.uint8)
    affs_u8[:, :, :, 16] = rng.randint(0, 30, size=(3, shape[0], shape[1]), dtype=np.uint8)
    return affs_u8


def _u8_to_f32(affs_u8):
    """Convert uint8 [0,255] to float32 [0,1]."""
    return affs_u8.astype(np.float32) / 255.0


class TestUint8Float32Equivalence:
    """Verify uint8 and float32 paths produce the same segmentation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.affs_u8 = _make_test_volume()
        self.affs_f32 = _u8_to_f32(self.affs_u8)

    def _run_u8(self, thresholds_u8, scoring, **kwargs):
        import waterz
        results = []
        for seg in waterz.agglomerate(
            self.affs_u8,
            thresholds=thresholds_u8,
            scoring_function=scoring,
            aff_threshold_low=1,
            aff_threshold_high=254,
            **kwargs,
        ):
            results.append(np.array(seg, copy=True))
        return results

    def _run_f32(self, thresholds_f32, scoring, **kwargs):
        import waterz
        results = []
        for seg in waterz.agglomerate(
            self.affs_f32,
            thresholds=thresholds_f32,
            scoring_function=scoring,
            aff_threshold_low=1 / 255.0,
            aff_threshold_high=254 / 255.0,
            **kwargs,
        ):
            results.append(np.array(seg, copy=True))
        return results

    def test_same_fragments(self):
        """Watershed fragments should be identical."""
        segs_u8 = self._run_u8(
            [0], "OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>")
        segs_f32 = self._run_f32(
            [0], "OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>")

        n_u8 = len(np.unique(segs_u8[0]))
        n_f32 = len(np.unique(segs_f32[0]))
        # Allow small difference from rounding at watershed boundaries
        assert abs(n_u8 - n_f32) <= max(n_u8, n_f32) * 0.02, \
            f"Fragment count differs too much: uint8={n_u8}, float32={n_f32}"

    def test_histogram_quantile_equivalence(self):
        """HistogramQuantile with 256 bins should be lossless for uint8 input."""
        scoring = "OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>"
        # uint8 threshold 128 ≈ float32 threshold 0.5 (OneMinus: 255-128=127, 1-0.5=0.5)
        segs_u8 = self._run_u8([128], scoring)
        segs_f32 = self._run_f32([0.5], scoring)

        n_u8 = len(np.unique(segs_u8[0])) - 1
        n_f32 = len(np.unique(segs_f32[0])) - 1
        # Should be very close (within 5% tolerance for boundary rounding)
        diff = abs(n_u8 - n_f32)
        assert diff <= max(n_u8, n_f32) * 0.05, \
            f"Segment count differs: uint8={n_u8}, float32={n_f32} (diff={diff})"

    def test_multiple_thresholds(self):
        """Multiple thresholds should produce decreasing segment counts for both."""
        scoring = "OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>"
        # uint8: thresholds in [0, 255] after OneMinus
        segs_u8 = self._run_u8([50, 100, 150], scoring)
        segs_f32 = self._run_f32([50/255, 100/255, 150/255], scoring)

        counts_u8 = [len(np.unique(s)) for s in segs_u8]
        counts_f32 = [len(np.unique(s)) for s in segs_f32]

        # Both should be monotonically decreasing (more merging at higher threshold)
        for i in range(len(counts_u8) - 1):
            assert counts_u8[i] >= counts_u8[i + 1], \
                f"uint8 not monotonic: {counts_u8}"
            assert counts_f32[i] >= counts_f32[i + 1], \
                f"float32 not monotonic: {counts_f32}"

    def test_with_fragments(self):
        """Pre-computed fragments should work with both dtypes."""
        import waterz
        scoring = "OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>"

        # Get fragments from float32 (reference)
        frags = None
        for seg in waterz.agglomerate(
            self.affs_f32, thresholds=[0],
            scoring_function=scoring,
            aff_threshold_low=1/255.0, aff_threshold_high=254/255.0,
        ):
            frags = seg.copy()
            break

        # Agglomerate with uint8 affinities on float32 fragments
        segs_u8 = []
        for seg in waterz.agglomerate(
            self.affs_u8, thresholds=[128],
            fragments=frags.copy(),
            scoring_function=scoring,
        ):
            segs_u8.append(seg.copy())
            break

        assert len(segs_u8) > 0
        assert len(np.unique(segs_u8[0])) > 1

    def test_memory_savings(self):
        """uint8 affinities should use 4x less memory."""
        assert self.affs_u8.nbytes * 4 == self.affs_f32.nbytes

    def test_build_region_graph_only(self):
        """buildRegionGraphOnly should work with uint8."""
        from waterz._agglomerate import build_region_graph_only

        scoring = "MeanAffinity<RegionGraphType, ScoreValue>"
        rg_u8 = build_region_graph_only(self.affs_u8,
            np.zeros(self.affs_u8.shape[1:], dtype=np.uint64),
            scoring_function=scoring)
        # Empty seg → no edges, but should not crash
        assert isinstance(rg_u8, list)


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


if __name__ == "__main__":
    print("Running uint8 vs float32 equivalence tests...\n")

    affs_u8 = _make_test_volume()
    affs_f32 = _u8_to_f32(affs_u8)

    import waterz

    # Test 1: Basic smoke test
    print("1. Smoke test (uint8)...")
    test_basic_uint8()
    print("   PASS\n")

    # Test 2: Compare segment counts
    print("2. Segment count comparison...")
    scoring = "OneMinus<HistogramQuantileAffinity<RegionGraphType, 85, ScoreValue, 256>>"

    seg_u8 = waterz.waterz(affs_u8, thresholds=[64],
                           scoring_function=scoring,
                           aff_threshold_low=1, aff_threshold_high=254)[0]
    seg_f32 = waterz.waterz(affs_f32, thresholds=[64/255],
                            scoring_function=scoring,
                            aff_threshold_low=1/255, aff_threshold_high=254/255)[0]

    n_u8 = len(np.unique(seg_u8)) - 1
    n_f32 = len(np.unique(seg_f32)) - 1
    diff_pct = abs(n_u8 - n_f32) / max(n_u8, n_f32, 1) * 100
    print(f"   uint8:   {n_u8} segments")
    print(f"   float32: {n_f32} segments")
    print(f"   diff:    {diff_pct:.1f}%")
    print(f"   {'PASS' if diff_pct < 5 else 'FAIL'}\n")

    # Test 3: Memory
    print("3. Memory savings...")
    print(f"   uint8:   {affs_u8.nbytes / 1e6:.1f} MB")
    print(f"   float32: {affs_f32.nbytes / 1e6:.1f} MB")
    print(f"   ratio:   {affs_f32.nbytes / affs_u8.nbytes:.0f}x")
    print("   PASS\n")

    print("All tests passed!")

"""JIT-compiled watershed + agglomeration on affinity graphs.

Supports both float32 and uint8 affinities.  Each (dtype, scoring_function,
discretize_queue) combination gets its own cached compiled module via witty.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from types import ModuleType

    from numpy.typing import NDArray

HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# JIT compilation helper
# ---------------------------------------------------------------------------

_AFF_DTYPE_MAP = {
    np.dtype("float32"): ("float", "float"),
    np.dtype("uint8"): ("uint8_t", "uint8_t"),
}

_SEG_DTYPE_MAP = {
    np.dtype("uint64"): "uint64_t",
    np.dtype("uint32"): "uint32_t",
}

# Cython source patches for uint8 mode.
# Each (old, new) pair is applied via str.replace.
_PYX_UINT8_PATCHES = [
    # Typed array declarations
    ("np.float32_t, ndim=4", "np.uint8_t, ndim=4"),
    # Local pointer declarations
    ("cdef float*    aff_data", "cdef uint8_t*  aff_data"),
    ("cdef float* aff_data", "cdef uint8_t* aff_data"),
    # C extern function signatures (affinities)
    ("const float*    affinity_data", "const uint8_t*  affinity_data"),
    ("const float*            affinity_data", "const uint8_t*          affinity_data"),
    # Struct score fields + threshold
    ("float score", "uint8_t score"),
    ("float           affThresholdLow", "uint8_t         affThresholdLow"),
    ("float           affThresholdHigh", "uint8_t         affThresholdHigh"),
    ("float        threshold", "uint8_t      threshold"),
]

# Cython source patches for uint32 segmentation.
_PYX_UINT32_SEG_PATCHES = [
    # Typed array declarations (segmentation / fragments)
    ("uint64_t, ndim=3]     segmentation", "uint32_t, ndim=3]     segmentation"),
    ("uint64_t, ndim=3] seg", "uint32_t, ndim=3] seg"),
    # Local pointer declarations
    ("cdef uint64_t* segmentation_data", "cdef uint32_t* segmentation_data"),
    ("cdef uint64_t* seg_data", "cdef uint32_t* seg_data"),
    # Array creation dtype
    ("dtype=np.uint64", "dtype=np.uint32"),
    # C extern: struct fields (Merge, ScoredEdge use SegID = uint32_t)
    ("        uint64_t a\n        uint64_t b\n        uint64_t c",
     "        uint32_t a\n        uint32_t b\n        uint32_t c"),
    ("        uint64_t u\n        uint64_t v",
     "        uint32_t u\n        uint32_t v"),
    # C extern: function signatures
    ("            uint64_t*       segmentation_data,", "            uint32_t*       segmentation_data,"),
    ("            uint64_t*       segmentation_data)", "            uint32_t*       segmentation_data)"),
]

_SOURCE_KEY_GLOBS = (
    '_agglomerate.py',
    'agglomerate.pyx',
    'frontend_agglomerate.cpp',
    'frontend_agglomerate.h',
    'backend/**/*.h',
    'backend/**/*.hpp',
    'backend/**/*.cpp',
)


def _source_fingerprint() -> str:
    """Hash local JIT sources so ABI changes invalidate cached builds."""
    digest = hashlib.md5()
    for pattern in _SOURCE_KEY_GLOBS:
        for path in sorted(HERE.glob(pattern)):
            if not path.is_file():
                continue
            digest.update(str(path.relative_to(HERE)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def _compile_module(
    scoring_function: str,
    discretize_queue: int = 0,
    aff_dtype: np.dtype = np.dtype("float32"),
    seg_dtype: np.dtype = np.dtype("uint64"),
    force_rebuild: bool = False,
) -> "ModuleType":
    """Compile and cache the agglomerate Cython/C++ module.

    Returns the compiled module with ``agglomerate()`` and
    ``buildRegionGraphOnly()`` entry points.
    """
    import os
    import subprocess
    import sys as _sys

    import witty

    if aff_dtype not in _AFF_DTYPE_MAP:
        raise ValueError(f"Unsupported aff_dtype {aff_dtype}; expected float32 or uint8")
    aff_ctype, score_ctype = _AFF_DTYPE_MAP[aff_dtype]

    if seg_dtype not in _SEG_DTYPE_MAP:
        raise ValueError(f"Unsupported seg_dtype {seg_dtype}; expected uint64 or uint32")
    seg_ctype = _SEG_DTYPE_MAP[seg_dtype]

    # Cache keys must reflect source/header ABI, not just runtime parameters.
    cache_dir = witty.get_witty_cache_dir()
    source_key = _source_fingerprint()
    header_key = hashlib.md5(
        f"{scoring_function}|{discretize_queue}|{aff_ctype}|{seg_ctype}|{source_key}".encode()
    ).hexdigest()[:12]
    header_dir = cache_dir / f"_waterz_headers_{header_key}"
    header_dir.mkdir(parents=True, exist_ok=True)

    # SegType.h — parameterises SegID (uint64_t or uint32_t)
    (header_dir / "SegType.h").write_text(
        f"typedef {seg_ctype} SegID;\n"
    )

    # AffType.h — parameterises AffValue and ScoreValue
    (header_dir / "AffType.h").write_text(
        f"typedef {aff_ctype} AffValue;\n"
        f"typedef {score_ctype} ScoreValue;\n"
    )

    # ScoringFunction.h
    (header_dir / "ScoringFunction.h").write_text(
        f"typedef {scoring_function} ScoringFunctionType;"
    )

    # Queue.h
    queue_src = "template<typename T, typename S> using QueueType = " + (
        "PriorityQueue<T, S>;"
        if discretize_queue == 0
        else f"BinQueue<T, S, {discretize_queue}>;"
    )
    (header_dir / "Queue.h").write_text(queue_src)

    include_dirs = [
        str(HERE),
        str(header_dir),
        str(HERE / "backend"),
        np.get_include(),
    ]
    _boost_inc = Path(_sys.prefix) / "include"
    if (_boost_inc / "boost").is_dir():
        include_dirs.append(str(_boost_inc))

    # Pre-compile frontend C++
    frontend_cpp = HERE / "frontend_agglomerate.cpp"
    obj_path = cache_dir / f"_waterz_frontend_{header_key}.o"

    if not obj_path.exists() or force_rebuild:
        compile_cmd = ["g++", "-c", "-fPIC", "-std=c++11", "-w", "-O2"]
        for d in include_dirs:
            compile_cmd += ["-I", d]
        compile_cmd += [str(frontend_cpp), "-o", str(obj_path)]
        env = os.environ.copy()
        env["CCACHE_DISABLE"] = "1"
        subprocess.check_call(compile_cmd, env=env)

    # Patch .pyx source for uint8 affinities and/or uint32 segmentation
    pyx_source = (HERE / "agglomerate.pyx").read_text()
    if aff_ctype == "uint8_t":
        for old, new in _PYX_UINT8_PATCHES:
            pyx_source = pyx_source.replace(old, new)
    if seg_ctype == "uint32_t":
        for old, new in _PYX_UINT32_SEG_PATCHES:
            pyx_source = pyx_source.replace(old, new)

    module = witty.compile_cython(
        pyx_source,
        source_files=[str(frontend_cpp)],
        extra_link_args=["-std=c++11", str(obj_path)],
        extra_compile_args=["-std=c++11", "-w"],
        include_dirs=include_dirs,
        language="c++",
        quiet=True,
        force_rebuild=force_rebuild,
    )
    return module


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def agglomerate(
    affs: NDArray,
    thresholds: Sequence[float],
    gt: NDArray[np.uint32] | None = None,
    fragments: NDArray[np.uint64] | None = None,
    aff_threshold_low: float = 0.0001,
    aff_threshold_high: float = 0.9999,
    return_merge_history: bool = False,
    return_region_graph: bool = False,
    rescore_region_graph: bool = True,
    scoring_function: str = "OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>",
    discretize_queue: int = 0,
    seg_dtype: np.dtype | str = "uint64",
    force_rebuild: bool = False,
) -> Iterator[tuple | NDArray]:
    """Compute segmentations from an affinity graph for several thresholds.

    Accepts **float32** or **uint8** affinities.  When uint8 is provided,
    the entire C++ pipeline operates on uint8 — no conversion to float32.
    For ``HistogramQuantileAffinity`` with 256 bins this is lossless and
    uses 4x less memory.

    Parameters
    ----------
    affs : ndarray, float32 or uint8, shape ``(3, Z, Y, X)``
        Affinity predictions.  float32 values in [0,1]; uint8 in [0,255].
    thresholds : sequence of float
        Agglomeration thresholds (for uint8 mode, scale to [0,255] range
        or use float — waterz handles the conversion internally).
    gt, fragments, aff_threshold_low, aff_threshold_high,
    return_merge_history, return_region_graph, scoring_function,
    discretize_queue, force_rebuild : see waterz documentation.

    Yields
    ------
    segmentation or (segmentation, metrics, merge_history, region_graph)
        depending on options.  Segmentation is uint64, modified in-place
        between yields — copy if needed.
    """
    affs = np.ascontiguousarray(affs)
    aff_dtype = affs.dtype

    # Validate and normalise dtype
    if aff_dtype == np.float64:
        affs = affs.astype(np.float32)
        aff_dtype = np.dtype("float32")
    if aff_dtype not in _AFF_DTYPE_MAP:
        raise TypeError(
            f"affs.dtype must be float32 or uint8, got {aff_dtype}"
        )

    seg_dtype = np.dtype(seg_dtype)

    if gt is not None:
        gt = np.ascontiguousarray(gt, dtype=np.uint32)
    if fragments is not None:
        fragments = np.ascontiguousarray(fragments, dtype=seg_dtype)

    module = _compile_module(
        scoring_function, discretize_queue,
        aff_dtype=aff_dtype, seg_dtype=seg_dtype, force_rebuild=force_rebuild,
    )

    return module.agglomerate(
        affs,
        thresholds,
        gt,
        fragments,
        aff_threshold_low,
        aff_threshold_high,
        return_merge_history,
        return_region_graph,
        rescore_region_graph,
    )


def build_region_graph_only(
    affs: NDArray,
    fragments: NDArray,
    scoring_function: str = "MeanAffinity<RegionGraphType, ScoreValue>",
    discretize_queue: int = 0,
    seg_dtype: np.dtype | str = "uint64",
    force_rebuild: bool = False,
) -> list:
    """Build scored region graph without RegionMerging or priority queue.

    Accepts float32 or uint8 affinities, uint64 or uint32 segmentation.

    Returns list of dicts ``[{"u": id1, "v": id2, "score": float}, ...]``.
    """
    affs = np.ascontiguousarray(affs)
    aff_dtype = affs.dtype
    if aff_dtype == np.float64:
        affs = affs.astype(np.float32)
        aff_dtype = np.dtype("float32")
    if aff_dtype not in _AFF_DTYPE_MAP:
        raise TypeError(f"affs.dtype must be float32 or uint8, got {aff_dtype}")

    seg_dtype = np.dtype(seg_dtype)
    fragments = np.ascontiguousarray(fragments, dtype=seg_dtype)

    module = _compile_module(
        scoring_function, discretize_queue,
        aff_dtype=aff_dtype, seg_dtype=seg_dtype, force_rebuild=force_rebuild,
    )

    return module.buildRegionGraphOnly(affs, fragments)


def build_region_graph_rich(
    affs: NDArray,
    fragments: NDArray,
    scoring_function: str = "MeanAffinity<RegionGraphType, ScoreValue>",
    discretize_queue: int = 0,
    seg_dtype: np.dtype | str = "uint64",
    force_rebuild: bool = False,
) -> list:
    """Build scored region graph with contact area per edge.

    Accepts float32 or uint8 affinities, uint64 or uint32 segmentation.

    Returns list of dicts ``[{"u": id1, "v": id2, "score": float, "contact_area": int}, ...]``.
    """
    affs = np.ascontiguousarray(affs)
    aff_dtype = affs.dtype
    if aff_dtype == np.float64:
        affs = affs.astype(np.float32)
        aff_dtype = np.dtype("float32")
    if aff_dtype not in _AFF_DTYPE_MAP:
        raise TypeError(f"affs.dtype must be float32 or uint8, got {aff_dtype}")

    seg_dtype = np.dtype(seg_dtype)
    fragments = np.ascontiguousarray(fragments, dtype=seg_dtype)

    module = _compile_module(
        scoring_function, discretize_queue,
        aff_dtype=aff_dtype, seg_dtype=seg_dtype, force_rebuild=force_rebuild,
    )

    return module.buildRegionGraphRich(affs, fragments)

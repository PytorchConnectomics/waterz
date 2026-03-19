from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from numpy.typing import NDArray

HERE = Path(__file__).parent


def agglomerate(
    affs: NDArray[np.float32],
    thresholds: Sequence[float],
    gt: NDArray[np.uint32] | None = None,
    fragments: NDArray[np.uint64] | None = None,
    aff_threshold_low: float = 0.0001,
    aff_threshold_high: float = 0.9999,
    return_merge_history: bool = False,
    return_region_graph: bool = False,
    scoring_function: str = "OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>",
    discretize_queue: int = 0,
    force_rebuild: bool = False,
) -> Iterator[tuple | NDArray[np.uint64]]:
    """Compute segmentations from an affinity graph for several thresholds.

    Passed volumes need to be converted into contiguous memory arrays. This will
    be done for you if needed, but you can save memory by making sure your
    volumes are already C_CONTIGUOUS.

    Parameters
    ----------

        affs: numpy array, float32, 4 dimensional

            The affinities as an array with affs[channel][z][y][x].

        thresholds: list of float32

            The thresholds to compute segmentations for. For each threshold, one
            segmentation is returned.

        gt: numpy array, uint32, 3 dimensional (optional)

            An optional ground-truth segmentation as an array with gt[z][y][x].
            If given, metrics

        fragments: numpy array, uint64, 3 dimensional (optional)

            An optional volume of fragments to use, instead of the build-in
            zwatershed.

        aff_threshold_low: float, default 0.0001
        aff_threshold_high: float, default 0.9999,

            Thresholds on the affinities for the initial segmentation step.

        return_merge_history: bool

            If set to True, the returning tuple will contain a merge history,
            relative to the previous segmentation.

        return_region_graph: bool

            If set to True, the returning tuple will contain the region graph
            for the returned segmentation.

        scoring_function: string, default 'OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>'

            A C++ type string specifying the edge scoring function to use. See

                https://github.com/funkey/waterz/blob/master/waterz/backend/MergeFunctions.hpp

            for available functions, and

                https://github.com/funkey/waterz/blob/master/waterz/backend/Operators.hpp

            for operators to combine them.

        discretize_queue: int

            If set to non-zero, a bin queue with that many bins will be used to
            approximate the priority queue for merge operations.

        force_rebuild: bool

            Force the rebuild of the module. Only needed for development.

    Returns
    -------

        Results are returned as tuples from a generator object, and only
        computed on-the-fly when iterated over. This way, you can ask for
        hundreds of thresholds while at any point only one segmentation is
        stored in memory.

        Depending on the given parameters, the returned values are a subset of
        the following items (in that order):

        segmentation

            The current segmentation (numpy array, uint64, 3 dimensional).

        metrics (only if ground truth was provided)

            A  dictionary with the keys 'V_Rand_split', 'V_Rand_merge',
            'V_Info_split', and 'V_Info_merge'.

        merge_history (only if return_merge_history is True)

            A list of dictionaries with keys 'a', 'b', 'c', and 'score',
            indicating that region a got merged with b into c with the given
            score.

        region_graph (only if return_region_graph is True)

            A list of dictionaries with keys 'u', 'v', and 'score', indicating
            an edge between u and v with the given score.

    Examples
    --------

        affs = ...
        gt   = ...

        # only segmentation
        for segmentation in agglomerate(affs, range(100,10000,100)):
            # ...

        # segmentation with merge history
        for segmentation, merge_history in agglomerate(
            affs, range(100,10000,100), return_merge_history = True):
            # ...

        # segmentation with merge history and metrics compared to gt
        for segmentation, metrics, merge_history in agglomerate(
            affs, range(100,10000,100), gt, return_merge_history = True):
            # ...
    """
    import subprocess
    import hashlib

    import witty

    # --- Persistent header directory ---
    # ScoringFunction.h and Queue.h must survive past the witty call because
    # witty caches the compiled .so — but the headers are baked into the
    # object at compile time so we only need them during the first build.
    # We store them under the witty cache keyed by scoring_function +
    # discretize_queue so different configs don't collide.
    cache_dir = witty.get_witty_cache_dir()
    header_key = hashlib.md5(
        f"{scoring_function}|{discretize_queue}".encode()
    ).hexdigest()[:12]
    header_dir = cache_dir / f"_waterz_headers_{header_key}"
    header_dir.mkdir(parents=True, exist_ok=True)

    scoredef = f"typedef {scoring_function} ScoringFunctionType;"
    (header_dir / "ScoringFunction.h").write_text(scoredef)

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

    # Add conda/system boost headers if available
    import sys as _sys
    _conda_prefix = Path(_sys.prefix)
    _boost_inc = _conda_prefix / "include"
    if (_boost_inc / "boost").is_dir():
        include_dirs.append(str(_boost_inc))

    # --- Pre-compile frontend_agglomerate.cpp into an object file ---
    # witty's source_files parameter is only used for cache-key hashing,
    # not for compilation.  We compile the C++ frontend ourselves and
    # pass the resulting .o via extra_link_args.
    frontend_cpp = HERE / "frontend_agglomerate.cpp"
    obj_path = cache_dir / f"_waterz_frontend_{header_key}.o"

    if not obj_path.exists() or force_rebuild:
        compile_cmd = [
            "g++", "-c", "-fPIC",
            "-std=c++11", "-w", "-O2",
        ]
        for d in include_dirs:
            compile_cmd += ["-I", d]
        compile_cmd += [str(frontend_cpp), "-o", str(obj_path)]
        subprocess.check_call(compile_cmd)

    # compile module
    module = witty.compile_cython(
        (HERE / "agglomerate.pyx").read_text(),
        source_files=[str(frontend_cpp)],
        extra_link_args=["-std=c++11", str(obj_path)],
        extra_compile_args=["-std=c++11", "-w"],
        include_dirs=include_dirs,
        language="c++",
        quiet=True,
        force_rebuild=force_rebuild,
    )

    # call compiled function
    return module.agglomerate(
        affs,
        thresholds,
        gt,
        fragments,
        aff_threshold_low,
        aff_threshold_high,
        return_merge_history,
        return_region_graph,
    )


def build_region_graph_only(
    affs: NDArray[np.float32],
    fragments: NDArray[np.uint64],
    scoring_function: str = "MeanAffinity<RegionGraphType, ScoreValue>",
    discretize_queue: int = 0,
    force_rebuild: bool = False,
) -> list:
    """Build scored region graph without RegionMerging or priority queue.

    Skips the full agglomeration state machine — only builds the region
    graph, collects statistics, and scores each edge.  ~2-3x faster than
    ``agglomerate(thresholds=[0], return_region_graph=True)``.

    Returns list of dicts ``[{"u": id1, "v": id2, "score": float}, ...]``.
    """
    import subprocess
    import hashlib

    import witty

    affs = np.ascontiguousarray(affs, dtype=np.float32)
    fragments = np.ascontiguousarray(fragments, dtype=np.uint64)

    cache_dir = witty.get_witty_cache_dir()
    header_key = hashlib.md5(
        f"{scoring_function}|{discretize_queue}".encode()
    ).hexdigest()[:12]
    header_dir = cache_dir / f"_waterz_headers_{header_key}"
    header_dir.mkdir(parents=True, exist_ok=True)

    (header_dir / "ScoringFunction.h").write_text(
        f"typedef {scoring_function} ScoringFunctionType;"
    )
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
    import sys as _sys
    _boost_inc = Path(_sys.prefix) / "include"
    if (_boost_inc / "boost").is_dir():
        include_dirs.append(str(_boost_inc))

    frontend_cpp = HERE / "frontend_agglomerate.cpp"
    obj_path = cache_dir / f"_waterz_frontend_{header_key}.o"

    if not obj_path.exists() or force_rebuild:
        compile_cmd = ["g++", "-c", "-fPIC", "-std=c++11", "-w", "-O2"]
        for d in include_dirs:
            compile_cmd += ["-I", d]
        compile_cmd += [str(frontend_cpp), "-o", str(obj_path)]
        subprocess.check_call(compile_cmd)

    module = witty.compile_cython(
        (HERE / "agglomerate.pyx").read_text(),
        source_files=[str(frontend_cpp)],
        extra_link_args=["-std=c++11", str(obj_path)],
        extra_compile_args=["-std=c++11", "-w"],
        include_dirs=include_dirs,
        language="c++",
        quiet=True,
        force_rebuild=force_rebuild,
    )

    return module.buildRegionGraphOnly(affs, fragments)

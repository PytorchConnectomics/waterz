"""Large-volume parallel waterz decoding using file-backed orchestrator.

Usage:
    # Serial (single process, all stages)
    waterz_decode_large --config tutorials/waterz_decoding_large.yaml

    # Parallel (N workers on one machine)
    waterz_decode_large --config tutorials/waterz_decoding_large.yaml --parallel 4

    # Initialize workflow only (for SLURM parallel launch)
    waterz_decode_large --config tutorials/waterz_decoding_large.yaml --init-only

    # Run as a worker (claims tasks from shared workflow dir)
    waterz_decode_large --config tutorials/waterz_decoding_large.yaml --worker

    # Wait for all workers to finish, then assemble output
    waterz_decode_large --config tutorials/waterz_decoding_large.yaml --wait --assemble

The `waterz_decode_large` console script is installed via
`pip install -e lib/waterz/`. The same module is also runnable directly via
`python -m waterz.cli.decode_large` (equivalent behavior).
"""

import argparse
import os
import shlex
import sys

import yaml

_STAGE_ORDER = [
    "decode",
    "fragment",
    "offsets",
    "stitch",
    "connect",
    "build_rg",
    "merge_rg",
    "agglomerate",
    "relabel",
    "apply",
    "assemble",
]

_STAGE_LABELS = {
    "decode": "chunk_waterz",
    "fragment": "chunk_waterz",
    "offsets": "id_offsets",
    "stitch": "bd_stitch",
    "connect": "bd_connect",
    "build_rg": "chunk_rg",
    "merge_rg": "global_rg",
    "agglomerate": "global_waterz",
    "relabel": "global_relabel",
    "apply": "apply_relabel",
    "assemble": "assemble",
}


def _ordered_stages(counts):
    """Order stage_counts() keys by pipeline position."""
    order = {s: i for i, s in enumerate(_STAGE_ORDER)}
    return sorted(counts.keys(), key=lambda s: (order.get(s, 999), s))


def _format_progress(counts):
    """Format stage_counts() in pipeline order."""
    parts = []
    for stage in _ordered_stages(counts):
        sc = counts[stage]
        done = sc.get("succeeded", 0)
        total = sum(sc.values())
        pending = sc.get("pending", 0)
        running = sc.get("running", 0)
        failed = sc.get("failed", 0)
        extras = []
        if running:
            extras.append(f"{running} running")
        if pending:
            extras.append(f"{pending} pending")
        if failed:
            extras.append(f"{failed} failed")
        status = f"{_STAGE_LABELS.get(stage, stage)}: {done}/{total}"
        if extras:
            status += f" ({', '.join(extras)})"
        parts.append(status)
    return f"  Progress: {' -> '.join(parts)}"


def _pipeline_steps(config, runner):
    """Human-readable workflow stages matching the task DAG."""
    if runner._use_overlap_pipeline:
        chunk_method = "waterz" if config.fragment_init == "waterz" else "2d fragments"
        threshold = float(config.chunk_affinity_threshold)
        if threshold >= 1.0:
            premerge = "no chunk premerge"
        else:
            premerge = f"relaxed premerge >= {threshold:g}"
        return [
            (
                "chunk_waterz",
                len(runner.chunks),
                "parallel",
                f"{chunk_method} per overlapped chunk; {premerge}",
            ),
            ("id_offsets", 1, "serial", "assign globally unique chunk label ids"),
            (
                "bd_stitch",
                len(runner.borders),
                "parallel",
                "write overlap equivalence pairs; can run with chunk_rg",
            ),
            (
                "chunk_rg",
                len(runner.chunks),
                "parallel",
                "build per-chunk region graphs; can run with bd_stitch",
            ),
            ("global_rg", 1, "serial", "merge per-chunk region graphs"),
            (
                "global_waterz",
                1,
                "serial",
                "threshold merge on global_rg plus bd_stitch pairs",
            ),
            (
                "apply_relabel",
                len(runner.chunks),
                "parallel",
                "write final chunk labels",
            ),
            ("assemble", 1 if config.write_output else 0, "serial", "write output H5"),
        ]
    return [
        ("chunk_waterz", len(runner.chunks), "parallel", "run waterz per chunk"),
        ("id_offsets", 1, "serial", "assign globally unique chunk ids"),
        (
            "bd_connect",
            len(runner.borders),
            "parallel",
            "connect labels across chunk faces",
        ),
        (
            "global_relabel",
            1,
            "serial",
            "reduce boundary links into a global relabel map",
        ),
        ("apply_relabel", len(runner.chunks), "parallel", "write final chunk labels"),
        ("assemble", 1 if config.write_output else 0, "serial", "write output H5"),
    ]


def _print_workflow_summary(config, runner):
    chunks = runner.chunks
    borders = runner.borders
    print(f"Volume shape: {config.volume_shape}")
    print(f"Chunk shape:  {config.chunk_shape}")
    print(f"Overlap:      {config.overlap}")
    print(f"Chunks:       {len(chunks)}")
    print(f"Borders:      {len(borders)}")
    print(f"Workflow:     {config.workflow_root}")
    print("Pipeline:")
    for i, (name, count, mode, detail) in enumerate(
        _pipeline_steps(config, runner), start=1
    ):
        print(f"  {i}. {name}: {count} {mode} - {detail}")


def _worker_fn(args_tuple):
    """Worker function for parallel decode (takes all args as tuple for spawn compatibility)."""
    worker_idx, workflow_root, idle_timeout, max_tasks = args_tuple
    os.environ["CCACHE_DISABLE"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    from waterz import LargeDecodeRunner as _LDR

    w = _LDR.load(workflow_root)
    return w.run_worker(
        worker_id=f"local-{worker_idx}",
        idle_timeout=idle_timeout,
        max_tasks=max_tasks,
    )


def main():
    parser = argparse.ArgumentParser(description="Large-volume waterz decoding")
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument(
        "--init-only", action="store_true", help="Initialize workflow and exit"
    )
    parser.add_argument(
        "--worker", action="store_true", help="Run as a worker (claim tasks)"
    )
    parser.add_argument(
        "--chunk-index",
        type=int,
        default=None,
        help="Decode a specific chunk by index (for sbatch --array)",
    )
    parser.add_argument(
        "--chunk-range",
        type=str,
        default=None,
        help="Decode chunk range 'start-end' (inclusive)",
    )
    parser.add_argument(
        "--wait", action="store_true", help="Wait for all tasks to complete"
    )
    parser.add_argument(
        "--assemble", action="store_true", help="Assemble final output volume"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=None,
        help="Run N worker processes on this machine",
    )
    parser.add_argument(
        "--sbatch",
        action="store_true",
        help="Force SLURM submission (overrides backend config)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Force local multiprocess (overrides backend config)",
    )
    parser.add_argument(
        "--max-tasks", type=int, default=None, help="Max tasks per worker"
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=60.0,
        help="Worker idle timeout (seconds)",
    )
    parser.add_argument("--worker-id", type=str, default=None, help="Worker identifier")
    parser.add_argument("--job-id", type=str, default=None, help="SLURM job ID")
    parser.add_argument(
        "--stale-timeout",
        type=float,
        default=600,
        help="Reset RUNNING tasks older than this many seconds (default: 600)",
    )
    parser.add_argument(
        "--no-reset-stale",
        action="store_true",
        help="Skip resetting stale RUNNING tasks",
    )
    parser.add_argument("overrides", nargs="*", help="Config overrides (key=value)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    large_cfg = cfg.get("large_decode", {})

    # Apply CLI overrides
    for override in args.overrides:
        if "=" not in override:
            print(
                f"Warning: skipping invalid override '{override}' (expected key=value)"
            )
            continue
        key, value = override.split("=", 1)
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        large_cfg[key] = value

    if not large_cfg.get("affinity_path"):
        print("Error: large_decode.affinity_path is required")
        sys.exit(1)
    if not large_cfg.get("workflow_root"):
        print("Error: large_decode.workflow_root is required")
        sys.exit(1)

    os.environ.setdefault("CCACHE_DISABLE", "1")

    from waterz import LargeDecodeConfig, LargeDecodeRunner

    # Build config from yaml dict — from_dict handles all field mapping
    config = LargeDecodeConfig.from_dict(large_cfg)
    runner = LargeDecodeRunner(config)
    runner.initialize()

    # Reset stale/failed tasks so re-runs recover from crashed workers. Wait mode
    # is monitor-only; resetting there can race with long-running SLURM workers.
    if not args.no_reset_stale and not args.wait:
        n_stale = runner.orchestrator.reset_stale_tasks(
            max_age_seconds=args.stale_timeout
        )
        runner.orchestrator.reset_failed_tasks()
        if n_stale:
            print(
                f"Reset {n_stale} stale RUNNING tasks (older than {args.stale_timeout}s)."
            )

    # Recover direct decode tasks that completed outside the orchestrator
    # (e.g. --chunk-index). The overlap pipeline stores fragment outputs under
    # the same raw chunk paths but has fragment:* task ids, so this recovery is
    # intentionally limited to the non-overlap decode workflow.
    if not runner._use_overlap_pipeline:
        n_recovered = 0
        for chunk in runner.chunks:
            output_path = runner._raw_chunk_path(chunk.key)
            if not output_path.exists():
                continue
            task_id = f"decode:{chunk.key}"
            try:
                record = runner.orchestrator.get_record(task_id)
                if record.state.value == "succeeded":
                    continue
                max_id = runner._read_chunk_max(output_path)
                runner.orchestrator.force_complete(
                    task_id,
                    result={"chunk_path": str(output_path), "max_id": max_id},
                )
                n_recovered += 1
            except Exception as e:
                print(f"  Warning: {chunk.key}: corrupt output ({e}), deleting")
                output_path.unlink(missing_ok=True)
        if n_recovered:
            print(f"Recovered {n_recovered} decode tasks from existing chunk files.")

    chunks = runner.chunks
    if not args.worker:
        _print_workflow_summary(config, runner)

    if args.init_only:
        print("Workflow initialized. Launch workers to execute tasks.")
        return

    # Determine execution backend: CLI flags override YAML config
    if args.sbatch:
        backend = "slurm"
    elif args.local:
        backend = "multiprocess"
    else:
        backend = large_cfg.get("backend", "multiprocess")

    if backend == "slurm":
        import subprocess
        import tempfile
        import textwrap

        slurm_cfg = large_cfg.get("slurm", {})
        partition = slurm_cfg.get("partition", "weilab")
        mem = slurm_cfg.get("mem", "64G")
        cpus = slurm_cfg.get("cpus_per_task", 2)
        time_limit = slurm_cfg.get("time", "12:00:00")
        worker_idle_timeout = float(slurm_cfg.get("idle_timeout", args.idle_timeout))
        n_chunks = len(chunks)

        config_path = os.path.abspath(args.config)
        work_dir = os.getcwd()
        output_dir = os.path.join(work_dir, "slurm_outputs")
        os.makedirs(output_dir, exist_ok=True)
        waterz_src_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        waterz_src_for_display = os.path.relpath(waterz_src_path, work_dir)
        monitor_cmd = (
            f"PYTHONPATH={shlex.quote(waterz_src_for_display)} "
            f"python -m waterz.cli.decode_large --config {shlex.quote(config_path)} --wait"
        )

        # Invoke via `python -m waterz.cli.decode_large` so the sbatch worker
        # does not depend on the location of the entry-point binary.
        sbatch_script = textwrap.dedent(f"""\
            #!/bin/bash
            #SBATCH --job-name=waterz_worker
            #SBATCH --partition={partition}
            #SBATCH --mem={mem}
            #SBATCH --cpus-per-task={cpus}
            #SBATCH --time={time_limit}
            #SBATCH --array=0-{n_chunks - 1}
            #SBATCH --output={output_dir}/waterz_worker_%A_%a.out
            #SBATCH --error={output_dir}/waterz_worker_%A_%a.err

            source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
            cd {work_dir}
            export CCACHE_DISABLE=1
            export OMP_NUM_THREADS=1
            export OPENBLAS_NUM_THREADS=1
            export MKL_NUM_THREADS=1
            export PYTHONUNBUFFERED=1
            export PYTHONPATH={waterz_src_path}:${{PYTHONPATH:-}}

            python -u -m waterz.cli.decode_large --config {config_path} --worker --no-reset-stale \\
                --idle-timeout {worker_idle_timeout:g} \\
                --worker-id "${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}" \\
                --job-id "${{SLURM_ARRAY_TASK_ID}}"
        """)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(sbatch_script)
            tmp_path = f.name

        result = subprocess.run(["sbatch", tmp_path], capture_output=True, text=True)
        os.unlink(tmp_path)
        print(result.stdout.strip())
        if result.returncode != 0:
            print(result.stderr.strip(), file=sys.stderr)
            sys.exit(result.returncode)
        print(f"Monitor: {monitor_cmd}")
        return

    if args.worker:
        worker_id = args.worker_id or os.environ.get("SLURM_JOB_ID", None)
        job_id = args.job_id or os.environ.get("SLURM_ARRAY_TASK_ID", None)
        print(
            f"Starting worker: {worker_id or 'auto'} (job={job_id or 'none'})",
            flush=True,
        )
        n = runner.run_worker(
            worker_id=worker_id,
            max_tasks=args.max_tasks,
            idle_timeout=args.idle_timeout,
            job_id=job_id,
        )
        print(f"Worker completed {n} tasks.", flush=True)
        return

    # Direct chunk assignment (no orchestrator competition)
    chunk_index = args.chunk_index
    if not args.wait and chunk_index is None and os.environ.get("SLURM_ARRAY_TASK_ID"):
        # Auto-detect from SLURM array index for direct chunk-array jobs only.
        chunk_index = int(os.environ["SLURM_ARRAY_TASK_ID"])

    if not args.wait and (chunk_index is not None or args.chunk_range is not None):
        if args.chunk_range:
            start, end = args.chunk_range.split("-")
            indices = list(range(int(start), int(end) + 1))
        else:
            indices = [chunk_index]
        from waterz.orchestrator import TaskRecord, TaskSpec

        for idx in indices:
            if idx >= len(chunks):
                print(f"Chunk index {idx} out of range (0-{len(chunks)-1}), skipping")
                continue
            chunk = chunks[idx]
            output_path = runner._raw_chunk_path(chunk.key)
            if output_path.exists():
                print(
                    f"Chunk {idx}/{len(chunks)} ({chunk.key}): already exists, skipping"
                )
                continue
            if runner._use_overlap_pipeline:
                print(f"Running chunk_waterz {idx}/{len(chunks)}: {chunk.key}")
                spec = TaskSpec(name="fragment_chunk", stage="fragment", key=chunk.key)
                result = runner.handle_fragment_chunk(TaskRecord(spec=spec))
            else:
                print(f"Decoding chunk {idx}/{len(chunks)}: {chunk.key}")
                spec = TaskSpec(
                    name=f"decode_{chunk.key}", stage="decode", key=chunk.key
                )
                result = runner.handle_decode_chunk(TaskRecord(spec=spec))
            print(f"  Done: {result}")
        return

    if args.wait:
        import time as _time

        print("Waiting for all tasks to complete...")
        last_print = 0
        while True:
            counts = runner.orchestrator.stage_counts()
            now = _time.monotonic()
            if now - last_print >= 10:
                print(_format_progress(counts), flush=True)
                last_print = now

            all_terminal = all(
                all(k in ("succeeded", "failed") for k in sc) for sc in counts.values()
            )
            if counts and all_terminal:
                break
            _time.sleep(1)

        print("All tasks completed.")
        # Check for failures
        for stage in _ordered_stages(counts):
            sc = counts[stage]
            failed = sc.get("failed", 0)
            if failed:
                label = _STAGE_LABELS.get(stage, stage)
                print(f"  WARNING: {label} has {failed} failed tasks")

        if args.assemble and config.write_output:
            print("Assembling output...")
            runner.handle_assemble_output(None)
            print(f"Output: {config.resolved_output_path}")
        return

    n_parallel = args.parallel or large_cfg.get("num_workers", 1)
    if n_parallel > 1:
        import multiprocessing as mp

        workflow_root = large_cfg["workflow_root"]
        idle_timeout = args.idle_timeout or 120
        max_tasks = args.max_tasks

        n_workers = n_parallel
        print(f"Running parallel decode with {n_workers} workers...")

        worker_args = [
            (i, workflow_root, idle_timeout, max_tasks) for i in range(n_workers)
        ]
        ctx = mp.get_context("spawn")
        with ctx.Pool(n_workers) as pool:
            counts = pool.map(_worker_fn, worker_args)
        n = sum(counts)
        print(f"Completed {n} tasks across {n_workers} workers.")
    else:
        n = runner.run_serial()
        print(f"Completed {n} tasks.")

    status = runner.orchestrator.stage_counts()
    for stage in _ordered_stages(status):
        counts = status[stage]
        print(f"  {_STAGE_LABELS.get(stage, stage)}: {counts}")


if __name__ == "__main__":
    main()

"""Timing driver: one runner, one scene, one JSON result file."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).parent))

from runners import make_runner  # noqa: E402
from scene import SceneConfig, build_scene, particle_stats  # noqa: E402


def percentile(samples, q):
    return float(np.percentile(np.asarray(samples, dtype=np.float64), q))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", required=True)
    ap.add_argument("--particle-count", type=int, default=32768)
    ap.add_argument("--spacing", type=float, default=0.006)
    ap.add_argument("--h-over-s", type=float, default=1.8)
    ap.add_argument("--substeps", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--viscosity", type=float, default=0.0)
    ap.add_argument("--cohesion", type=float, default=0.0)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--max-neighbors", type=int, default=96)
    ap.add_argument("--no-cuda-graph", action="store_true")
    ap.add_argument("--kernel-breakdown", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    wp.init()
    device = wp.get_device()

    cfg = SceneConfig(
        particle_count=args.particle_count,
        spacing=args.spacing,
        h_over_s=args.h_over_s,
        substeps=args.substeps,
        iterations=args.iterations,
        viscosity=args.viscosity,
        cohesion=args.cohesion,
        fps=args.fps,
    )
    # Newton needs real collision shapes for the tank; OmniSurg confines the
    # fluid with its analytic bounds and would double-collide against shapes.
    with_walls = args.runner.startswith("newton")
    scene = build_scene(cfg, with_walls=with_walls)
    runner = make_runner(args.runner, scene, max_neighbors=args.max_neighbors)

    dt = cfg.sim_dt

    def frame():
        for _ in range(cfg.substeps):
            runner.substep(dt)

    # ---- warm up (kernel compilation + module load) ----
    t_compile = time.perf_counter()
    frame()
    wp.synchronize()
    compile_s = time.perf_counter() - t_compile

    graph = None
    graph_error = None
    capture_ms = 0.0
    if device.is_cuda and not args.no_cuda_graph:
        try:
            t0 = time.perf_counter()
            with wp.ScopedCapture() as capture:
                frame()
            graph = capture.graph
            wp.synchronize()
            capture_ms = (time.perf_counter() - t0) * 1e3
        except Exception as exc:  # noqa: BLE001
            graph_error = f"{type(exc).__name__}: {exc}"
            graph = None

    def step_frame():
        if graph is not None:
            wp.capture_launch(graph)
        else:
            frame()

    for _ in range(args.warmup):
        step_frame()
    wp.synchronize()

    samples = []
    for _ in range(args.frames):
        t0 = time.perf_counter()
        step_frame()
        wp.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)

    stats = particle_stats(scene.state_0)

    # ---- optional per-kernel breakdown (uncaptured, CUDA event timing) ----
    breakdown = None
    if args.kernel_breakdown:
        wp.synchronize()
        reps = 4
        wp.timing_begin(cuda_filter=wp.TIMING_ALL)
        for _ in range(reps):
            frame()
        wp.synchronize()
        results = wp.timing_end()
        agg: dict[str, list[float]] = {}
        for r in results:
            agg.setdefault(r.name, []).append(r.elapsed)
        breakdown = sorted(
            (
                {"kernel": k, "total_ms": sum(v) / reps, "launches": len(v) // reps}
                for k, v in agg.items()
            ),
            key=lambda d: -d["total_ms"],
        )

    result = {
        "runner": args.runner,
        "label": runner.name,
        "device": str(device),
        "device_name": device.name if hasattr(device, "name") else str(device),
        "warp_version": wp.config.version,
        "particle_count": scene.particle_count,
        "dims": list(scene.dims),
        "substeps": cfg.substeps,
        "iterations": cfg.iterations,
        "spacing": cfg.spacing,
        "h_over_s": cfg.h_over_s,
        "smoothing_length": cfg.smoothing_length,
        "frame_dt": cfg.frame_dt,
        "sim_dt": dt,
        "frames": args.frames,
        "cuda_graph": graph is not None,
        "cuda_graph_error": graph_error,
        "graph_capture_ms": capture_ms,
        "first_frame_compile_s": compile_s,
        "solver_info": runner.info,
        "ms_median": statistics.median(samples),
        "ms_mean": statistics.fmean(samples),
        "ms_min": min(samples),
        "ms_p95": percentile(samples, 95),
        "ms_p99": percentile(samples, 99),
        "ms_stdev": statistics.pstdev(samples),
        "fps_median": 1000.0 / statistics.median(samples),
        "samples": samples,
        "final_state": stats,
        "kernel_breakdown": breakdown,
    }

    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(
        f"RESULT {args.runner:28s} N={scene.particle_count:7d} "
        f"median={result['ms_median']:8.3f} ms  p95={result['ms_p95']:8.3f} ms  "
        f"graph={result['cuda_graph']} finite={stats['finite']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

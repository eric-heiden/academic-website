"""Run the whole benchmark matrix in one process (amortizes module load).

Results are appended to results/sweep.jsonl as they complete, so a crash or
timeout still leaves everything measured so far on disk.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).parent))

from runners import make_runner  # noqa: E402
from scene import SceneConfig, build_scene, particle_stats  # noqa: E402

OUT = Path(__file__).parent / "results"


def measure(
    runner_spec: str,
    *,
    particle_count: int,
    substeps: int = 8,
    iterations: int = 3,
    h_over_s: float = 1.8,
    spacing: float = 0.006,
    viscosity: float = 0.0,
    frames: int = 100,
    warmup: int = 20,
    max_neighbors: int = 96,
    kernel_breakdown: bool = False,
    group: str = "",
) -> dict:
    cfg = SceneConfig(
        particle_count=particle_count,
        spacing=spacing,
        h_over_s=h_over_s,
        substeps=substeps,
        iterations=iterations,
        viscosity=viscosity,
    )
    with_walls = runner_spec.startswith("newton")
    scene = build_scene(cfg, with_walls=with_walls)
    runner = make_runner(runner_spec, scene, max_neighbors=max_neighbors)
    dt = cfg.sim_dt
    device = wp.get_device()

    def frame():
        for _ in range(cfg.substeps):
            runner.substep(dt)

    t0 = time.perf_counter()
    frame()
    wp.synchronize()
    compile_s = time.perf_counter() - t0

    graph, graph_error, capture_ms = None, None, 0.0
    if device.is_cuda:
        try:
            t0 = time.perf_counter()
            with wp.ScopedCapture() as cap:
                frame()
            graph = cap.graph
            wp.synchronize()
            capture_ms = (time.perf_counter() - t0) * 1e3
        except Exception as exc:  # noqa: BLE001
            graph_error = f"{type(exc).__name__}: {exc}"

    def step_frame():
        if graph is not None:
            wp.capture_launch(graph)
        else:
            frame()

    for _ in range(warmup):
        step_frame()
    wp.synchronize()

    samples = []
    for _ in range(frames):
        t = time.perf_counter()
        step_frame()
        wp.synchronize()
        samples.append((time.perf_counter() - t) * 1e3)

    stats = particle_stats(scene.state_0)

    breakdown = None
    if kernel_breakdown:
        reps = 4
        wp.synchronize()
        wp.timing_begin(cuda_filter=wp.TIMING_ALL)
        for _ in range(reps):
            frame()
        wp.synchronize()
        agg: dict[str, list[float]] = {}
        for r in wp.timing_end():
            agg.setdefault(r.name, []).append(r.elapsed)
        breakdown = sorted(
            ({"kernel": k, "total_ms": sum(v) / reps, "launches": len(v) // reps} for k, v in agg.items()),
            key=lambda d: -d["total_ms"],
        )

    res = {
        "group": group,
        "runner": runner_spec,
        "label": runner.name,
        "particle_count": scene.particle_count,
        "dims": list(scene.dims),
        "substeps": substeps,
        "iterations": iterations,
        "h_over_s": h_over_s,
        "spacing": spacing,
        "viscosity": viscosity,
        "frames": frames,
        "cuda_graph": graph is not None,
        "cuda_graph_error": graph_error,
        "graph_capture_ms": capture_ms,
        "first_frame_s": compile_s,
        "solver_info": runner.info,
        "ms_median": statistics.median(samples),
        "ms_mean": statistics.fmean(samples),
        "ms_min": min(samples),
        "ms_p95": float(np.percentile(samples, 95)),
        "ms_p99": float(np.percentile(samples, 99)),
        "ms_stdev": statistics.pstdev(samples),
        "final_state": stats,
        "kernel_breakdown": breakdown,
    }
    del runner, scene, graph
    gc.collect()
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT / "sweep.jsonl")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--only", default=None, help="comma-separated group filter")
    args = ap.parse_args()

    wp.init()
    OUT.mkdir(parents=True, exist_ok=True)
    fh = args.out.open("a")

    NEWTON = "newton"
    OMNI_BASE = "omnisurg:baseline"
    OMNI_ALL = "omnisurg:all"

    jobs: list[tuple[str, dict]] = []

    # 1. scaling: how does cost grow with particle count?
    for n in (8_000, 16_000, 32_768, 65_536, 131_072):
        for r in (NEWTON, OMNI_BASE, OMNI_ALL):
            jobs.append(("scaling", dict(runner_spec=r, particle_count=n, group="scaling")))

    # 2. ablation: which OmniSurg optimization buys what?
    for mode in (
        "baseline",
        "uniform-grid",
        "fused",
        "specialized",
        "skip-render",
        "sorted",
        "fused+specialized",
        "all",
        "all+uniform",
        "all+flex",
    ):
        jobs.append(
            ("ablation", dict(runner_spec=f"omnisurg:{mode}", particle_count=32_768, group="ablation"))
        )
    jobs.append(("ablation", dict(runner_spec=NEWTON, particle_count=32_768, group="ablation")))

    # 3. solver-iteration sensitivity (the axis that exposes neighbor reuse)
    for it in (1, 2, 3, 4, 6, 8):
        for r in (NEWTON, OMNI_ALL):
            jobs.append(
                ("iterations", dict(runner_spec=r, particle_count=32_768, iterations=it, group="iterations"))
            )

    # 4. neighborhood size sensitivity
    for hs in (1.5, 1.8, 2.2, 2.5):
        for r in (NEWTON, OMNI_ALL):
            jobs.append(
                ("neighborhood", dict(runner_spec=r, particle_count=32_768, h_over_s=hs, group="neighborhood"))
            )

    # 5. kernel breakdown at a realistic size
    for r in (NEWTON, OMNI_BASE, OMNI_ALL):
        jobs.append(
            ("breakdown", dict(runner_spec=r, particle_count=65_536, kernel_breakdown=True, group="breakdown"))
        )

    if args.only:
        keep = set(args.only.split(","))
        jobs = [j for j in jobs if j[0] in keep]

    print(f"{len(jobs)} jobs", file=sys.stderr, flush=True)
    for i, (_g, kw) in enumerate(jobs):
        kw.setdefault("frames", args.frames)
        tag = f"{kw['group']}/{kw['runner_spec']}/N={kw['particle_count']}/it={kw.get('iterations', 3)}/hs={kw.get('h_over_s', 1.8)}"
        try:
            res = measure(**kw)
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            print(
                f"[{i + 1:3d}/{len(jobs)}] {tag:70s} {res['ms_median']:8.3f} ms "
                f"graph={res['cuda_graph']} ok={res['final_state']['finite']}",
                file=sys.stderr,
                flush=True,
            )
        except Exception:  # noqa: BLE001
            print(f"[{i + 1:3d}/{len(jobs)}] {tag:70s} FAILED", file=sys.stderr, flush=True)
            traceback.print_exc()
            fh.write(json.dumps({**{k: str(v) for k, v in kw.items()}, "error": traceback.format_exc()}) + "\n")
            fh.flush()
    fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Record body poses for a configuration, then replay them through ViewerGL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import warp as wp

sys.path.insert(0, "/home/horde/repos/fpgs-study")
sys.path.insert(0, "/home/horde/repos/newton-tests/src")

import newton  # noqa: E402
from harness import make_env  # noqa: E402
from exp_gravcomp import run_gc  # noqa: E402

CASES = {
    "mujoco": dict(solver="mujoco", gain_scale=1.0, gc=False, solver_kw=None),
    "fpgs_default": dict(solver="fpgs", gain_scale=1.0, gc=False, solver_kw=None),
    "fpgs_100x": dict(solver="fpgs", gain_scale=100.0, gc=False, solver_kw=None),
    "fpgs_gravity": dict(solver="fpgs", gain_scale=1.0, gc=True, solver_kw=None),
    "fpgs_cap32": dict(solver="fpgs", gain_scale=1.0, gc=True,
                       solver_kw={"dense_max_constraints": 32}),
    "fpgs_velit4": dict(solver="fpgs", gain_scale=1.0, gc=True,
                        solver_kw={"pgs_velocity_iterations": 4}),
    "fpgs_dt16ms": dict(solver="fpgs", gain_scale=1.0, gc=True, solver_kw=None, substeps=1),
}


def record(name, frames, stride):
    cfg = dict(CASES[name])
    gc = cfg.pop("gc")
    solver = cfg.pop("solver")
    env = make_env(solver, **{k: v for k, v in cfg.items() if v is not None})
    res = run_gc(env, frames, gc=gc, stride=stride, record_poses=True)
    np.savez_compressed(Path(f"data/poses_{name}.npz"),
                        poses=res["poses"], t=res["t"], phase=res["phase"])
    np.savez_compressed(Path(f"data/{name}.npz"),
                        **{k: v for k, v in res.items()
                           if k not in ("failure", "poses")},
                        failure=np.asarray(res["failure"]))
    print(f"{name}: {len(res['poses'])} poses, failure={res['failure']!r}", flush=True)
    del env


def render(name, size, fps):
    d = np.load(f"data/poses_{name}.npz")
    poses, times = d["poses"], d["t"]
    env = make_env("mujoco", capture_sim_graph=False)
    viewer = newton.viewer.ViewerGL(width=size, height=size, vsync=False, headless=True)
    viewer.set_model(env.model)
    viewer.picking_enabled = False
    viewer.set_camera(pos=wp.vec3(0.54, -1.20, 0.60), pitch=-22.0, yaw=133.0)
    st = env.model.state()
    out = Path(f"video/{name}.mp4")
    out.parent.mkdir(exist_ok=True, parents=True)
    w = imageio.get_writer(out, fps=fps, codec="libx264", quality=8,
                           macro_block_size=None, ffmpeg_log_level="warning")
    try:
        for i, (p, t) in enumerate(zip(poses, times)):
            st.body_q.assign(p)
            viewer.begin_frame(float(t))
            viewer.log_state(st)
            viewer.end_frame()
            frame = viewer.get_frame(render_ui=False).numpy()
            if i == 0:
                imageio.imwrite(out.with_suffix(".jpg"), frame, quality=90)
            w.append_data(frame)
    finally:
        w.close()
        viewer.close()
    print(f"{name}: wrote {out} ({len(poses)} frames)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("record", "render"))
    ap.add_argument("names", nargs="+")
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--size", type=int, default=640)
    a = ap.parse_args()
    wp.init()
    wp.set_device("cuda:0")
    for n in a.names:
        if a.mode == "record":
            record(n, a.frames, a.stride)
        else:
            render(n, a.size, 60 // a.stride)

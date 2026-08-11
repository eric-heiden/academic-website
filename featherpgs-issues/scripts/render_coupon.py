"""Record and replay the coupon scenes through ViewerGL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import warp as wp

sys.path.insert(0, "/home/horde/repos/fpgs-study")

import newton  # noqa: E402
import coupon  # noqa: E402

CASES = {
    "coupon_static_ok": dict(kind="static", geom="box", kw=dict(overlap=5.0e-4),
                             frames=300),
    "coupon_static_eject": dict(kind="static", geom="box", kw=dict(overlap=2.0e-3),
                                frames=300),
    "coupon_driven_box": dict(kind="driven", geom="box", kw={}, frames=420),
    "coupon_driven_mesh": dict(kind="driven", geom="mesh", kw={}, frames=420),
}


def record(name, stride):
    c = CASES[name]
    env = coupon.make(c["kind"], c["geom"], **c["kw"])
    r = coupon.run(env, frames=c["frames"], record_poses=True)
    Path("data").mkdir(exist_ok=True)
    np.savez_compressed(f"data/poses_{name}.npz", poses=r["poses"][::stride],
                        t=r["t"][::stride])
    print(f"{name}: {len(r['poses'][::stride])} poses, failure={r['failure']!r}",
          flush=True)
    del env


def render(name, size, fps):
    c = CASES[name]
    d = np.load(f"data/poses_{name}.npz")
    poses, times = d["poses"], d["t"]
    env = coupon.make(c["kind"], c["geom"], **c["kw"])
    viewer = newton.viewer.ViewerGL(width=size, height=size, vsync=False, headless=True)
    viewer.set_model(env["model"])
    viewer.picking_enabled = False
    viewer.set_camera(pos=wp.vec3(0.06, -0.28, 0.13), pitch=-6.0, yaw=102.0)
    st = env["model"].state()
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
            f = viewer.get_frame(render_ui=False).numpy()
            if i == 0:
                imageio.imwrite(out.with_suffix(".jpg"), f, quality=90)
            w.append_data(f)
    finally:
        w.close()
        viewer.close()
    print(f"{name}: wrote {out} ({len(poses)} frames)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("record", "render"))
    ap.add_argument("names", nargs="+")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--size", type=int, default=640)
    a = ap.parse_args()
    wp.init()
    wp.set_device("cuda:0")
    for n in a.names:
        if a.mode == "record":
            record(n, a.stride)
        else:
            render(n, a.size, 60 // a.stride)

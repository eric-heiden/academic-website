"""Side-by-side visual comparison: same dam-break scene, two solvers.

Dumps particle positions/velocities at matched frames for each solver, then
renders a filmstrip so the two implementations can be compared qualitatively as
well as by wall-clock.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).parent))

import newton  # noqa: E402
from runners import make_runner  # noqa: E402
from scene import Scene, SceneConfig  # noqa: E402

OUT = Path(__file__).parent / "results"


def build_dam_break(cfg: SceneConfig, *, with_walls: bool) -> Scene:
    """A water column in the -x corner of a tank, released at t=0."""
    from scene import cube_dims

    dx, dy, dz = cube_dims(cfg.particle_count)
    # elongate in z, narrow in x -> a proper collapsing column
    dz = int(dz * 1.7)
    dx = max(2, int(round(cfg.particle_count / (dy * dz))))
    s, r = cfg.spacing, cfg.radius
    bx, by, bz = dx * s, dy * s, dz * s

    tank_x, tank_y, tank_z = 4.5 * bx, 1.15 * by, 1.5 * bz
    bounds_min = (0.0, -0.5 * tank_y, 0.0)
    bounds_max = (tank_x, 0.5 * tank_y, tank_z)

    builder = newton.ModelBuilder(up_axis="Z", gravity=cfg.gravity)
    builder.default_particle_radius = r
    builder.add_particle_grid(
        pos=wp.vec3(0.5 * s, -0.5 * by, 0.5 * s),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=dx,
        dim_y=dy,
        dim_z=dz,
        cell_x=s,
        cell_y=s,
        cell_z=s,
        mass=cfg.mass,
        jitter=0.05 * s,
        radius_mean=r,
        flags=newton.ParticleFlags.ACTIVE | newton.ParticleFlags.FLUID,
    )

    if with_walls:
        t = 0.2 * tank_x
        wcfg = newton.ModelBuilder.ShapeConfig(mu=0.0)

        def static_box(cx, cy, cz, hx, hy, hz):
            b = builder.add_body(xform=wp.transform(wp.vec3(cx, cy, cz), wp.quat_identity()))
            builder.add_shape_box(body=b, hx=hx, hy=hy, hz=hz, cfg=wcfg)
            builder.body_mass[b] = 0.0
            builder.body_inv_mass[b] = 0.0

        static_box(0.5 * tank_x, 0.0, -t, 0.5 * tank_x + t, 0.5 * tank_y + t, t)
        static_box(-t, 0.0, 0.5 * tank_z, t, 0.5 * tank_y + t, 0.5 * tank_z)
        static_box(tank_x + t, 0.0, 0.5 * tank_z, t, 0.5 * tank_y + t, 0.5 * tank_z)
        static_box(0.5 * tank_x, -0.5 * tank_y - t, 0.5 * tank_z, 0.5 * tank_x + t, t, 0.5 * tank_z)
        static_box(0.5 * tank_x, 0.5 * tank_y + t, 0.5 * tank_z, 0.5 * tank_x + t, t, 0.5 * tank_z)

    model = builder.finalize()
    model.particle_max_velocity = 0.5 * r / cfg.sim_dt
    model.soft_contact_mu = 0.0

    return Scene(
        model=model,
        state_0=model.state(),
        state_1=model.state(),
        contacts=model.contacts() if with_walls else None,
        cfg=cfg,
        dims=(dx, dy, dz),
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        particle_count=model.particle_count,
    )


def capture(runner_spec: str, cfg: SceneConfig, frames: int, snap_at: list[int]) -> dict:
    scene = build_dam_break(cfg, with_walls=runner_spec.startswith("newton"))
    runner = make_runner(runner_spec, scene, max_neighbors=96)
    dt = cfg.sim_dt
    snaps = {}
    if 0 in snap_at:
        snaps[0] = (
            scene.state_0.particle_q.numpy().copy(),
            scene.state_0.particle_qd.numpy().copy(),
        )
    for f in range(1, frames + 1):
        for _ in range(cfg.substeps):
            runner.substep(dt)
        if f in snap_at:
            snaps[f] = (
                scene.state_0.particle_q.numpy().copy(),
                scene.state_0.particle_qd.numpy().copy(),
            )
    return {
        "snaps": snaps,
        "bounds": (scene.bounds_min, scene.bounds_max),
        "label": runner.name,
        "dims": scene.dims,
        "n": scene.particle_count,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--particle-count", type=int, default=32768)
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--snaps", default="0,8,16,26,40,60,90")
    ap.add_argument("--runners", default="newton,omnisurg:all")
    args = ap.parse_args()

    wp.init()
    snap_at = [int(v) for v in args.snaps.split(",")]
    cfg = SceneConfig(particle_count=args.particle_count, substeps=8, iterations=3)

    out = {}
    for spec in args.runners.split(","):
        print(f"capturing {spec} ...", file=sys.stderr, flush=True)
        out[spec] = capture(spec, cfg, args.frames, snap_at)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "visual.npz",
        **{
            f"{spec}|{f}|{kind}": arr
            for spec, d in out.items()
            for f, (q, qd) in d["snaps"].items()
            for kind, arr in (("q", q), ("qd", qd))
        },
    )
    (OUT / "visual_meta.json").write_text(
        json.dumps(
            {
                spec: {
                    "label": d["label"],
                    "bounds": d["bounds"],
                    "dims": d["dims"],
                    "n": d["n"],
                    "frames": sorted(d["snaps"]),
                }
                for spec, d in out.items()
            },
            indent=2,
        )
    )
    print("wrote", OUT / "visual.npz", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

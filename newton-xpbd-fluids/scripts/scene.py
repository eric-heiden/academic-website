"""Shared benchmark scene for Newton XPBD-PBF vs OmniSurg PBF.

Both implementations are Newton ``SolverBase`` subclasses operating on the same
``Model``/``State`` pair, so the only thing that differs between runs is the
solver under test. The scene is a cube of fluid particles dropped into an
axis-aligned tank so the fluid (not the collision geometry) dominates the cost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

import newton


@dataclass
class SceneConfig:
    particle_count: int = 32768
    spacing: float = 0.006  # rest distance [m]
    h_over_s: float = 1.8  # smoothing length / rest distance
    rest_density: float = 1000.0
    gravity: float = -9.81
    fps: float = 60.0
    substeps: int = 8
    iterations: int = 3
    viscosity: float = 0.0
    cohesion: float = 0.0
    # tank is sized from the fluid block so the fluid stays resident
    tank_pad: float = 1.6

    @property
    def radius(self) -> float:
        return 0.5 * self.spacing

    @property
    def smoothing_length(self) -> float:
        return self.h_over_s * self.spacing

    @property
    def mass(self) -> float:
        return self.rest_density * self.spacing**3

    @property
    def frame_dt(self) -> float:
        return 1.0 / self.fps

    @property
    def sim_dt(self) -> float:
        return self.frame_dt / self.substeps


def cube_dims(target: int) -> tuple[int, int, int]:
    """Near-cubic (dim_x, dim_y, dim_z) whose product is closest to ``target``."""
    n = max(1, round(target ** (1.0 / 3.0)))
    best = None
    for dx in range(max(1, n - 6), n + 7):
        for dy in range(max(1, n - 6), n + 7):
            dz = max(1, round(target / (dx * dy)))
            count = dx * dy * dz
            err = abs(count - target)
            if best is None or err < best[0]:
                best = (err, (dx, dy, dz))
    return best[1]


@dataclass
class Scene:
    model: newton.Model
    state_0: newton.State
    state_1: newton.State
    contacts: object | None
    cfg: SceneConfig
    dims: tuple[int, int, int]
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    particle_count: int


def build_scene(cfg: SceneConfig, *, with_walls: bool) -> Scene:
    """Build the shared tank scene.

    Args:
        cfg: Scene configuration.
        with_walls: When ``True`` the tank is made of real Newton collision
            shapes (used by the Newton solver, which has no analytic bounds).
            When ``False`` no shapes are added and the caller is expected to
            confine the fluid analytically (OmniSurg's ``bounds_min/max``).
    """
    dx, dy, dz = cube_dims(cfg.particle_count)
    s = cfg.spacing
    r = cfg.radius

    # Fluid block sits in the middle of a tank with `tank_pad` extra room in x/y
    # and headroom in z, dropped from a height so it develops a real flow.
    bx, by, bz = dx * s, dy * s, dz * s
    tank_x = cfg.tank_pad * bx
    tank_y = cfg.tank_pad * by
    tank_z = 2.2 * bz

    bounds_min = (-0.5 * tank_x, -0.5 * tank_y, 0.0)
    bounds_max = (0.5 * tank_x, 0.5 * tank_y, tank_z)

    builder = newton.ModelBuilder(up_axis="Z", gravity=cfg.gravity)
    builder.default_particle_radius = r

    builder.add_particle_grid(
        pos=wp.vec3(-0.5 * bx, -0.5 * by, 0.35 * tank_z),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=dx,
        dim_y=dy,
        dim_z=dz,
        cell_x=s,
        cell_y=s,
        cell_z=s,
        mass=cfg.mass,
        jitter=0.1 * s,
        radius_mean=r,
        flags=newton.ParticleFlags.ACTIVE | newton.ParticleFlags.FLUID,
    )

    if with_walls:
        # Static tank: floor + 4 walls, as boxes (no ground plane, so the
        # analytic-bounds and shape-collision variants confine the same volume).
        t = 0.05 * max(tank_x, tank_y)
        wall_cfg = newton.ModelBuilder.ShapeConfig(mu=0.0)
        floor = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, -t), wp.quat_identity()))
        builder.add_shape_box(body=floor, hx=0.5 * tank_x + t, hy=0.5 * tank_y + t, hz=t, cfg=wall_cfg)
        builder.body_mass[floor] = 0.0
        builder.body_inv_mass[floor] = 0.0
        for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cx = sx * (0.5 * tank_x + t)
            cy = sy * (0.5 * tank_y + t)
            b = builder.add_body(xform=wp.transform(wp.vec3(cx, cy, 0.5 * tank_z), wp.quat_identity()))
            builder.add_shape_box(
                body=b,
                hx=t if sx else 0.5 * tank_x + t,
                hy=t if sy else 0.5 * tank_y + t,
                hz=0.5 * tank_z,
                cfg=wall_cfg,
            )
            builder.body_mass[b] = 0.0
            builder.body_inv_mass[b] = 0.0

    model = builder.finalize()
    # CFL-style clamp identical for both solvers
    model.particle_max_velocity = 0.5 * r / cfg.sim_dt
    model.soft_contact_mu = 0.0

    state_0 = model.state()
    state_1 = model.state()
    contacts = model.contacts() if with_walls else None

    return Scene(
        model=model,
        state_0=state_0,
        state_1=state_1,
        contacts=contacts,
        cfg=cfg,
        dims=(dx, dy, dz),
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        particle_count=model.particle_count,
    )


def analytic_rest_density(spacing: float, h: float, mass: float) -> float:
    """Poly6 lattice-sum rest density, matching SolverXPBD's auto-calibration."""
    n = int(math.ceil(h / spacing))
    total = 0.0
    for ix in range(-n, n + 1):
        for iy in range(-n, n + 1):
            for iz in range(-n, n + 1):
                r_sq = float(ix * ix + iy * iy + iz * iz) * spacing * spacing
                if r_sq < h * h:
                    total += (h * h - r_sq) ** 3
    return mass * total * 315.0 / (64.0 * math.pi * h**9)


def particle_stats(state: newton.State) -> dict:
    q = state.particle_q.numpy()
    qd = state.particle_qd.numpy()
    return {
        "finite": bool(np.all(np.isfinite(q)) and np.all(np.isfinite(qd))),
        "com": [float(v) for v in q.mean(axis=0)],
        "min": [float(v) for v in q.min(axis=0)],
        "max": [float(v) for v in q.max(axis=0)],
        "speed_mean": float(np.linalg.norm(qd, axis=1).mean()),
        "speed_max": float(np.linalg.norm(qd, axis=1).max()),
    }

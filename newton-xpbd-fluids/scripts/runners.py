"""Solver runners under test.

Every runner exposes the same interface so the timing loop is identical:

    runner.substep(dt)   # one physics substep, graph-capturable
    runner.name          # label
    runner.info          # dict of resolved solver parameters
"""

from __future__ import annotations

import newton
import warp as wp

from scene import Scene, analytic_rest_density

# --------------------------------------------------------------------------
# Newton: SolverXPBD position-based-fluid path (eric-heiden/flex-fluid)
# --------------------------------------------------------------------------


class NewtonRunner:
    key = "newton"
    label = "Newton SolverXPBD (PBF)"

    def __init__(self, scene: Scene, *, collide: bool = True, **_):
        cfg = scene.cfg
        self.scene = scene
        self.collide = collide
        self.solver = newton.solvers.SolverXPBD(
            scene.model,
            iterations=cfg.iterations,
            fluid_rest_distance=cfg.spacing,
            fluid_smoothing_length=cfg.smoothing_length,
            fluid_rest_density=None,  # auto-calibrated lattice sum
            fluid_cohesion=cfg.cohesion,
            fluid_viscosity=cfg.viscosity,
            fluid_vorticity_confinement=0.0,
            max_diffuse_particles=0,
        )
        self.name = self.label

    @property
    def info(self) -> dict:
        s = self.solver
        return {
            "iterations": s.iterations,
            "rest_distance": float(s._fluid_rest_distance_eff),
            "smoothing_length": float(s._fluid_h),
            "rest_density": float(s._fluid_rest_density_eff),
            "cohesion": float(s.fluid_cohesion),
            "viscosity": float(s.fluid_viscosity),
            "max_neighbors": int(s.fluid_max_neighbors),
            "neighbor_structure": "wp.HashGrid",
        }

    def substep(self, dt: float) -> None:
        sc = self.scene
        sc.state_0.clear_forces()
        if self.collide and sc.contacts is not None:
            sc.model.collide(sc.state_0, sc.contacts)
        self.solver.step(sc.state_0, sc.state_1, None, sc.contacts, dt)
        sc.state_0, sc.state_1 = sc.state_1, sc.state_0


# --------------------------------------------------------------------------
# OmniSurg: PositionBasedFluidSolver (korzen-nv/omnisurg feature/fluids)
# --------------------------------------------------------------------------

OMNISURG_MODES = {
    "baseline": {},
    "uniform-grid": {"grid_backend": "uniform"},
    "fused": {"fuse_neighbor_build_first_lambda": True},
    "specialized": {"use_specialized_kernels": True},
    "skip-render": {"skip_unused_render_surface": True},
    "sorted": {"use_sorted_scratch": True},
    "fused+specialized": {
        "fuse_neighbor_build_first_lambda": True,
        "use_specialized_kernels": True,
    },
    "all": {
        "fuse_neighbor_build_first_lambda": True,
        "use_specialized_kernels": True,
        "skip_unused_render_surface": True,
        "use_sorted_scratch": True,
    },
    "all+uniform": {
        "fuse_neighbor_build_first_lambda": True,
        "use_specialized_kernels": True,
        "skip_unused_render_surface": True,
        "use_sorted_scratch": True,
        "grid_backend": "uniform",
    },
    "all+flex": {
        "fuse_neighbor_build_first_lambda": True,
        "use_specialized_kernels": True,
        "skip_unused_render_surface": True,
        "use_sorted_scratch": True,
        "density_constraint_mode": "flex_approx",
    },
}


class OmnisurgRunner:
    key = "omnisurg"

    def __init__(
        self,
        scene: Scene,
        *,
        mode: str = "baseline",
        max_neighbors: int = 96,
        external_graph: bool = True,
        **_,
    ):
        from omnisurg.fluids.systems import PositionBasedFluidSolver
        from omnisurg.physics.sidecars import OmniSurgModel, OmniSurgSidecars, OmniSurgState

        cfg = scene.cfg
        self.scene = scene
        self.mode = mode
        self.label = f"OmniSurg PBF [{mode}]"
        self.name = self.label

        self.sidecars = OmniSurgSidecars(model=OmniSurgModel(), state=OmniSurgState())
        fluid_state = self.sidecars.position_based_fluid_state
        fluid_state.densities = wp.zeros(scene.model.particle_count, dtype=wp.float32, device=scene.model.device)
        fluid_state.smooth_positions = wp.zeros(scene.model.particle_count, dtype=wp.vec3, device=scene.model.device)
        for attr in ("anisotropy_q1", "anisotropy_q2", "anisotropy_q3"):
            if hasattr(fluid_state, attr):
                setattr(
                    fluid_state,
                    attr,
                    wp.zeros(scene.model.particle_count, dtype=wp.vec3, device=scene.model.device),
                )

        h = cfg.smoothing_length
        # OmniSurg accumulates a *massless* kernel-weight sum, so its
        # "rest_density" is the lattice sum of W(r) rather than kg/m^3. Newton
        # multiplies by particle mass. Using each solver's own calibration makes
        # the normalized density constraint (rho/rho0 - 1) identical.
        from omnisurg.fluids.systems import _estimate_rest_density

        rest_density = _estimate_rest_density(h, cfg.spacing, 0)

        opts = dict(
            grid_backend="hashgrid",
            fuse_neighbor_build_first_lambda=False,
            use_specialized_kernels=False,
            skip_unused_render_surface=False,
            use_sorted_scratch=False,
            density_constraint_mode="standard",
        )
        opts.update(OMNISURG_MODES[mode])
        self.opts = opts

        self.solver = PositionBasedFluidSolver(
            scene.model,
            self.sidecars,
            particle_start=0,
            particle_count=scene.model.particle_count,
            iterations=cfg.iterations,
            radius=h,
            rest_distance=cfg.spacing,
            rest_density=rest_density,
            density_epsilon=1.0e-6,
            max_neighbors=max_neighbors,
            relaxation=1.0,
            cohesion=cfg.cohesion,
            cohesion_model="cohesion",
            surface_tension=0.0,
            viscosity=cfg.viscosity,
            wall_friction=0.0,
            wall_viscosity=0.0,
            velocity_update_blend=1.0,
            particle_radius=cfg.radius,
            particle_collisions=False,
            particle_collision_relaxation=1.0,
            use_spiky_kernels=False,  # poly6, matching Newton's kernel choice
            neighbor_weight_averaging=False,
            grid_rebuild_frequency="substep",
            neighbor_rebuild_frequency="substep",
            bounds_min=scene.bounds_min,
            bounds_max=scene.bounds_max,
            **opts,
        )
        # The solver captures its own per-substep density CUDA graph and launches
        # it from step(); that nested launch is illegal inside an outer capture
        # (CUDA error 900). `_engine_managed_stages` is the solver's own escape
        # hatch for exactly this case (used when OmniSurg's engine owns the
        # graph), so setting it lets us capture the whole frame the way the
        # Newton examples do -- an apples-to-apples comparison.
        self.external_graph = bool(external_graph)
        self.solver._engine_managed_stages = self.external_graph
        # `skip_unused_render_surface` only takes effect when nothing is asking
        # for the surface. Newton's equivalent (`update_render_particles`) is
        # never called headless, so declaring the surface unused is what makes
        # the two solvers do the same amount of work.
        if opts["skip_unused_render_surface"]:
            self.solver.set_render_surface_demand(False)

    @property
    def info(self) -> dict:
        s = self.solver
        return {
            "iterations": s.iterations,
            "rest_distance": float(s.rest_distance),
            "smoothing_length": float(s.radius),
            "rest_density": float(s.rest_density),
            "cohesion": float(s.cohesion),
            "viscosity": float(s.viscosity),
            "max_neighbors": int(s.max_neighbors),
            "neighbor_structure": ("uniform grid" if s.grid_backend == "uniform" else "wp.HashGrid")
            + " + materialized neighbor list",
            "opts": self.opts,
            "graph_mode": "external" if self.external_graph else "solver-internal",
        }

    def substep(self, dt: float) -> None:
        sc = self.scene
        sc.state_0.clear_forces()
        self.solver.step(sc.state_0, sc.state_1, None, None, dt)
        sc.state_0, sc.state_1 = sc.state_1, sc.state_0


def make_runner(spec: str, scene: Scene, **kwargs):
    if spec == "newton":
        return NewtonRunner(scene, **kwargs)
    if spec == "newton-nocollide":
        return NewtonRunner(scene, collide=False, **kwargs)
    if spec.startswith("omnisurg:"):
        mode = spec.split(":", 1)[1]
        native = mode.endswith("!native")
        return OmnisurgRunner(
            scene,
            mode=mode.removesuffix("!native"),
            external_graph=not native,
            **kwargs,
        )
    raise ValueError(f"unknown runner spec {spec!r}")

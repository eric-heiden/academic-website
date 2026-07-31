# SPDX-License-Identifier: Apache-2.0
"""Controlled variants of the Franka cube shake test.

Each variant changes exactly one thing relative to the baseline so the cause of
the per-cycle grasp drift can be isolated.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import warp as wp

import newton
from newton.viewer import ViewerNull

from newtontests.franka_cube_shake import FrankaCubeShake, Phase, create_parser
from newtontests.measure import rel_pose, quat_conj, quat_mul


class Variant(FrankaCubeShake):
    """FrankaCubeShake with configurable contact/solver plumbing."""

    def __init__(self, viewer, args, cfg):
        self.cfg = cfg
        super().__init__(viewer, args)

    def _build_franka(self):
        builder = super()._build_franka()
        cfg = self.cfg
        if cfg.get("rigid_gap") is not None:
            builder.rigid_gap = cfg["rigid_gap"]
            for i in range(len(builder.shape_gap)):
                builder.shape_gap[i] = cfg["rigid_gap"]
        if cfg.get("finger_target_ke") is not None:
            builder.joint_target_ke[7:9] = [cfg["finger_target_ke"]] * 2
        if cfg.get("finger_solimp") is not None:
            solimp = builder.custom_attributes.get("mujoco:geom_solimp")
            for shape_index, body_index in enumerate(builder.shape_body):
                if body_index in (12, 13):
                    solimp.values[shape_index] = cfg["finger_solimp"]
        if cfg.get("no_priority"):
            priority = builder.custom_attributes.get("mujoco:geom_priority")
            for shape_index, body_index in enumerate(builder.shape_body):
                if body_index in (12, 13):
                    priority.values[shape_index] = 0
        if cfg.get("finger_mu") is not None:
            for shape_index, body_index in enumerate(builder.shape_body):
                if body_index in (12, 13):
                    builder.shape_material_mu[shape_index] = cfg["finger_mu"]
        if cfg.get("finger_ke") is not None:
            for shape_index, body_index in enumerate(builder.shape_body):
                if body_index in (12, 13):
                    builder.shape_material_ke[shape_index] = cfg["finger_ke"]
                    if cfg.get("finger_kd") is not None:
                        builder.shape_material_kd[shape_index] = cfg["finger_kd"]
        return builder

    def _add_cube(self, scene):
        cfg = self.cfg
        idx = super()._add_cube(scene)
        shape = len(scene.shape_body) - 1
        if cfg.get("cube_mu") is not None:
            scene.shape_material_mu[shape] = cfg["cube_mu"]
        if cfg.get("cube_ke") is not None:
            scene.shape_material_ke[shape] = cfg["cube_ke"]
        if cfg.get("cube_kd") is not None:
            scene.shape_material_kd[shape] = cfg["cube_kd"]
        if cfg.get("cube_gap") is not None:
            scene.shape_gap[shape] = cfg["cube_gap"]
        return idx

    def _capture_graphs(self):
        cfg = self.cfg
        # Rebuild the solver / pipeline if the variant asks for it, before capture.
        if not getattr(self, "_variant_applied", False):
            self._apply_variant()
            self._variant_applied = True
        super()._capture_graphs()

    def _apply_variant(self):
        cfg = self.cfg
        contact_max = 4096

        if cfg.get("reduce_contacts") is not None or cfg.get("broad_phase"):
            self.collision_pipeline = newton.CollisionPipeline(
                self.model,
                reduce_contacts=cfg.get("reduce_contacts", True),
                rigid_contact_max=contact_max,
                broad_phase=cfg.get("broad_phase", "nxn"),
            )
            self.contacts = self.collision_pipeline.contacts()

        solver_kwargs = dict(
            solver="newton",
            integrator="implicitfast",
            iterations=15,
            ls_iterations=100,
            nconmax=contact_max,
            njmax=contact_max * 2,
            cone="elliptic",
            impratio=50.0,
            use_mujoco_contacts=False,
        )
        solver_kwargs.update(cfg.get("solver_kwargs", {}))
        if solver_kwargs != dict(
            solver="newton",
            integrator="implicitfast",
            iterations=15,
            ls_iterations=100,
            nconmax=contact_max,
            njmax=contact_max * 2,
            cone="elliptic",
            impratio=50.0,
            use_mujoco_contacts=False,
        ):
            self.solver = newton.solvers.SolverMuJoCo(self.model, **solver_kwargs)

        if cfg.get("substeps"):
            self.sim_substeps = int(cfg["substeps"])
            self.sim_dt = self.frame_dt / self.sim_substeps

    def _simulate(self):
        cfg = self.cfg
        mj_contacts = cfg.get("solver_kwargs", {}).get("use_mujoco_contacts", False)
        collide_every_substep = cfg.get("collide_every_substep", False)

        if mj_contacts:
            for _ in range(self.sim_substeps):
                self.state_0.clear_forces()
                self.viewer.apply_forces(self.state_0)
                self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
                self.state_0, self.state_1 = self.state_1, self.state_0
            return

        if collide_every_substep:
            for _ in range(self.sim_substeps):
                self.collision_pipeline.collide(self.state_0, self.contacts)
                self.state_0.clear_forces()
                self.viewer.apply_forces(self.state_0)
                self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
                self.state_0, self.state_1 = self.state_1, self.state_0
            return

        n = int(cfg.get("collide_every_n", self.sim_substeps))
        for i in range(self.sim_substeps):
            if i % n == 0:
                self.collision_pipeline.collide(self.state_0, self.contacts)
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0


def run_variant(name, cfg, num_frames=900, amplitude=0.03, frequency=1.0, verbose=True):
    argv_backup = sys.argv
    sys.argv = ["exp"]
    args = newton.examples.default_args(create_parser())
    sys.argv = argv_backup
    args.shake_amplitude = amplitude
    args.shake_frequency = frequency

    viewer = ViewerNull(num_frames=num_frames)
    ex = Variant(viewer, args, cfg)

    ref_pos = ref_quat = None
    rows = []
    for frame in range(num_frames):
        ex.step()
        body_q = ex.state_0.body_q.numpy()
        cube, tcp = body_q[ex.cube_index], body_q[ex.ee_index]
        rp, rq = rel_pose(tcp, cube)
        phase = int(ex.phase_index.numpy()[0])
        if phase == Phase.SHAKE.value and ref_pos is None:
            ref_pos, ref_quat = rp.copy(), rq.copy()
        slip = float(np.linalg.norm(rp - ref_pos)) if ref_pos is not None else 0.0
        if ref_quat is not None:
            dq = quat_mul(rq, quat_conj(ref_quat))
            ang = float(np.degrees(2.0 * np.arccos(np.clip(abs(dq[3]), 0.0, 1.0))))
        else:
            ang = 0.0
        jq = ex.state_0.joint_q.numpy()
        rows.append(
            {
                "frame": frame, "t": frame / 60.0, "phase": phase,
                "cube_z": float(cube[2]), "rel_x": float(rp[0]),
                "rel_y": float(rp[1]), "rel_z": float(rp[2]),
                "slip": slip, "slip_deg": ang,
                "finger0": float(jq[7]), "finger1": float(jq[8]),
                "ncon": int(ex.contacts.rigid_contact_count.numpy()[0]),
            }
        )
        if verbose and frame % 120 == 0:
            r = rows[-1]
            print(f"[{name}] f={frame:4d} phase={Phase(phase).name:8s} cube_z={r['cube_z']:.4f} "
                  f"slip={slip*1000:7.3f}mm rot={ang:5.2f}deg ncon={r['ncon']}", flush=True)

    sh = [r for r in rows if r["phase"] == Phase.SHAKE.value]
    if len(sh) > 120:
        t = np.array([r["t"] for r in sh])
        rx = np.array([r["rel_x"] for r in sh])
        rz = np.array([r["rel_z"] for r in sh])
        drift_x = float(np.polyfit(t, rx, 1)[0]) * 1000.0
        drift_z = float(np.polyfit(t, rz, 1)[0]) * 1000.0
    else:
        drift_x = drift_z = float("nan")
    summary = {
        "name": name, "cfg": {k: str(v) for k, v in cfg.items()},
        "shake_frames": len(sh),
        "drift_x_mm_per_s": drift_x, "drift_z_mm_per_s": drift_z,
        "final_slip_mm": rows[-1]["slip"] * 1000.0,
        "max_slip_mm": max((r["slip"] for r in sh), default=0.0) * 1000.0,
        "final_slip_deg": rows[-1]["slip_deg"],
        "final_cube_z": rows[-1]["cube_z"],
        "mean_ncon": float(np.mean([r["ncon"] for r in sh])) if sh else 0.0,
    }
    print("[" + name + "] SUMMARY " + json.dumps(summary), flush=True)
    return summary, rows


VARIANTS = {
    "baseline": {},
    "collide_every_substep": {"collide_every_substep": True},
    "collide_every_4": {"collide_every_n": 4},
    "mujoco_contacts": {"solver_kwargs": {"use_mujoco_contacts": True}},
    "no_reduce": {"reduce_contacts": False},
    "substeps1_dt60": {"substeps": 1},
    "pyramidal": {"solver_kwargs": {"cone": "pyramidal"}},
    "impratio1": {"solver_kwargs": {"impratio": 1.0}},
    # --- friction capacity: does the grasp saturate its friction cone? ---
    "mu_0p3": {"cube_mu": 0.3},
    "finger_mu_0p2": {"finger_mu": 0.2},
    "finger_mu_0p5": {"finger_mu": 0.5},
    "finger_mu_3": {"finger_mu": 3.0},
    "finger_mu_10": {"finger_mu": 10.0},
    "no_priority": {"no_priority": True},
    "mu_5": {"cube_mu": 5.0},
    "mu_20": {"cube_mu": 20.0},
    # --- solver residual: does more solver work reduce the creep? ---
    "iters_50": {"solver_kwargs": {"iterations": 50}},
    "iters_200": {"solver_kwargs": {"iterations": 200}},
    "solver_cg": {"solver_kwargs": {"solver": "cg"}},
    # --- timestep: is the creep per-step or per-second? ---
    "substeps_4": {"substeps": 4},
    "substeps_32": {"substeps": 32},
    "substeps_64": {"substeps": 64},
    # --- constraint hardness ---
    "impratio_200": {"solver_kwargs": {"impratio": 200.0}},
    "tol_1e10": {"solver_kwargs": {"tolerance": 1e-10, "ls_tolerance": 1e-10, "iterations": 100}},
    "tol_1e6": {"solver_kwargs": {"tolerance": 1e-6}},
    "finger_ke_2p5e4": {"finger_ke": 2.5e4, "finger_kd": 3.0e2},
    "finger_ke_2p5e5": {"finger_ke": 2.5e5, "finger_kd": 1.0e3},
    "finger_ke_2p5e6": {"finger_ke": 2.5e6, "finger_kd": 3.0e3},
    "solimp_099": {"finger_solimp": (0.9, 0.99, 0.0001, 0.5, 2.0)},
    "solimp_09999": {"finger_solimp": (0.95, 0.9999, 0.0001, 0.5, 2.0)},
    "fixed": {
        "finger_ke": 2.5e5,
        "finger_kd": 1.0e3,
        "cube_gap": 0.005,
    },
    "impratio_1000": {"solver_kwargs": {"impratio": 1000.0}},
    "solimp_hard": {"finger_solimp": (0.95, 0.9999, 0.0001, 0.5, 2.0)},
    # --- gap / speculative contact band ---
    "gap0": {"rigid_gap": 0.0, "cube_gap": 0.0},
    # --- grip force ---
    "grip_ke_1600": {"finger_target_ke": 1600.0},
    # --- integrator ---
    "implicit": {"solver_kwargs": {"integrator": "implicit"}},
    "euler": {"solver_kwargs": {"integrator": "euler"}},
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="+")
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--amplitude", type=float, default=0.03)
    ap.add_argument("--frequency", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    results = []
    for v in a.variants:
        s, rows = run_variant(v, VARIANTS[v], a.frames, a.amplitude, a.frequency)
        results.append({"summary": s, "rows": rows})
    if a.out:
        json.dump(results, open(a.out, "w"))
    print(json.dumps([r["summary"] for r in results], indent=2))

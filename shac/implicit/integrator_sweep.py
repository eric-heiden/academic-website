"""Forward stability and one-control-step accuracy across integrators and timesteps.

For a fixed 20 ms control interval, each (integrator, timestep) pair is evaluated
with two protocols:

* stability: ``nworld`` worlds start from the nominal pose with a small random
  perturbation and receive the same temporally correlated random actions for
  ``duration`` seconds. A world diverges when its state becomes non-finite, any
  generalized velocity exceeds 1e3, or the root leaves the [-1, 20] m height band.
* accuracy: states sampled from a reference trajectory (implicitfast, 0.25 ms) are
  advanced by one control interval with the same control; the result is compared
  with the reference advanced from the same state.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from diffsim import DiffSim, SimConfig
from robots import ROBOTS

CONTROL_DT = 0.02
TIMESTEPS = (0.001, 0.002, 0.004, 0.005, 0.01, 0.02)
INTEGRATORS = ("euler_explicit", "euler", "implicitfast")
REF = ("implicitfast", 0.00025)


def quat_angle_deg(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    dot = (q1 * q2).sum(-1).abs().clamp(max=1.0)
    return torch.rad2deg(2.0 * torch.arccos(dot))


def make_actions(steps, nworld, nu, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    a = torch.zeros(nworld, nu, device=device)
    out = torch.empty(steps, nworld, nu, device=device)
    for t in range(steps):
        a = (0.9 * a + 0.45 * torch.randn(nworld, nu, device=device, generator=g)).clamp(-1.0, 1.0)
        out[t] = a
    return out


def initial_state(sim: DiffSim, nworld: int, seed: int):
    g = torch.Generator(device=sim.torch_device).manual_seed(seed + 1)
    q = sim.init_qpos.expand(nworld, -1).clone()
    q[:, 7:] += 0.05 * (2 * torch.rand(nworld, sim.nq - 7, device=sim.torch_device, generator=g) - 1)
    v = 0.1 * (2 * torch.rand(nworld, sim.nv, device=sim.torch_device, generator=g) - 1)
    return q, v


def build(robot, integrator, dt, nworld):
    r = ROBOTS[robot]
    k = round(CONTROL_DT / dt)
    cfg = SimConfig(xml=r.xml, integrator=integrator, timestep=dt, substeps=k, keyframe=r.keyframe,
                    nconmax=r.nconmax, njmax=r.njmax)
    return DiffSim(cfg, nworld, backward=False)


def diverged(q, v):
    bad = ~torch.isfinite(q).all(1) | ~torch.isfinite(v).all(1)
    bad |= v.abs().amax(1) > 1e3
    bad |= (q[:, 2] < -1.0) | (q[:, 2] > 20.0)
    return bad


def stability(sim: DiffSim, robot, actions, seed):
    r = ROBOTS[robot]
    nworld = actions.shape[1]
    home = sim.init_qpos[7:] if r.position_control else None
    q, v = initial_state(sim, nworld, seed)
    w = torch.zeros_like(v)
    sim.clear_overflow()
    dead = torch.zeros(nworld, dtype=torch.bool, device=q.device)
    first = torch.full((nworld,), float("nan"), device=q.device)
    maxspeed = torch.zeros(nworld, device=q.device)
    min_dist = math.inf
    states = []
    for t in range(actions.shape[0]):
        ctrl = sim.action_to_ctrl(actions[t], r.position_control, home)
        q, v, w = sim.step_nograd(q, v, ctrl, w)
        bad = diverged(q, v)
        first = torch.where(bad & ~dead, torch.full_like(first, (t + 1) * CONTROL_DT), first)
        dead |= bad
        # Freeze diverged worlds at a finite state so they cannot poison shared buffers.
        q = torch.where(dead[:, None], sim.init_qpos.expand(nworld, -1), q)
        v = torch.where(dead[:, None], 0.0, v)
        w = torch.where(dead[:, None], 0.0, w)
        maxspeed = torch.maximum(maxspeed, torch.where(dead, 0.0, v[:, 6:].abs().amax(1)))
        n = int(sim.t_nacon[0].item())
        if n:
            min_dist = min(min_dist, float(sim.t_contact_dist[:n].min().item()))
        if (t + 1) % 25 == 0:
            states.append((q.clone(), v.clone(), t))
    return {
        "diverged_fraction": float(dead.float().mean().item()),
        "median_time_to_divergence_s": float(first[dead].median().item()) if dead.any() else None,
        "p99_max_joint_speed": float(maxspeed.quantile(0.99).item()),
        "min_contact_distance_m": min_dist,
        "overflow": sim.overflowed(),
        "overflow_world_fraction": sim.overflow_stats(),
    }, states


def one_step(sim: DiffSim, robot, q, v, a):
    r = ROBOTS[robot]
    home = sim.init_qpos[7:] if r.position_control else None
    ctrl = sim.action_to_ctrl(a, r.position_control, home)
    q1, v1, _ = sim.step_nograd(q, v, ctrl, torch.zeros_like(v))
    return q1, v1


def errors(q, v, qr, vr):
    ok = ~diverged(q, v) & ~diverged(qr, vr)
    pos = (q[:, :3] - qr[:, :3]).norm(dim=1)
    ang = quat_angle_deg(q[:, 3:7], qr[:, 3:7])
    joint = torch.rad2deg((q[:, 7:] - qr[:, 7:]).pow(2).mean(1).sqrt())
    linv = (v[:, :3] - vr[:, :3]).norm(dim=1)
    jointv = (v[:, 6:] - vr[:, 6:]).pow(2).mean(1).sqrt()
    out = {"valid_fraction": float(ok.float().mean().item())}
    for name, x in (("root_pos_mm", 1e3 * pos), ("root_ang_deg", ang), ("joint_rms_deg", joint),
                    ("root_linvel", linv), ("joint_vel_rms", jointv)):
        x = x[ok]
        out[name] = {"median": float(x.median().item()), "p95": float(x.quantile(0.95).item())}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--robot", required=True)
    p.add_argument("--nworld", type=int, default=1024)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    steps = round(args.duration / CONTROL_DT)

    # Reference trajectory and sampled states.
    t0 = time.time()
    ref = build(args.robot, *REF, args.nworld)
    actions = make_actions(steps, args.nworld, ref.nu, ref.torch_device, args.seed)
    ref_stab, states = stability(ref, args.robot, actions, args.seed)
    # Use states that have not diverged; 4 time points x nworld samples.
    picks = states[1::max(1, len(states) // 4)][:4]
    Q = torch.cat([s[0] for s in picks]); V = torch.cat([s[1] for s in picks])
    A = torch.cat([actions[min(s[2] + 1, steps - 1)] for s in picks])
    nsample = Q.shape[0]
    print(f"reference built {time.time()-t0:.1f}s diverged={ref_stab['diverged_fraction']}", flush=True)

    # Reference one-step results in chunks of nworld.
    def chunked(sim, fn):
        qs, vs = [], []
        for i in range(0, nsample, args.nworld):
            q1, v1 = fn(sim, args.robot, Q[i:i + args.nworld], V[i:i + args.nworld], A[i:i + args.nworld])
            qs.append(q1); vs.append(v1)
        return torch.cat(qs), torch.cat(vs)

    QR, VR = chunked(ref, one_step)
    result = {"robot": args.robot, "control_dt": CONTROL_DT, "nworld": args.nworld, "duration_s": args.duration,
              "reference": {"integrator": REF[0], "timestep": REF[1], "stability": ref_stab}, "configs": []}
    del ref

    for integ in INTEGRATORS:
        for dt in TIMESTEPS:
            t0 = time.time()
            sim = build(args.robot, integ, dt, args.nworld)
            stab, _ = stability(sim, args.robot, actions, args.seed)
            q1, v1 = chunked(sim, one_step)
            acc = errors(q1, v1, QR, VR)
            row = {"integrator": integ, "timestep": dt, "substeps": sim.cfg.substeps, "stability": stab, "accuracy": acc}
            result["configs"].append(row)
            print(f"{integ:15s} dt={dt*1e3:5.2f}ms div={stab['diverged_fraction']:.3f} "
                  f"pos={acc['root_pos_mm']['median']:.3g}/{acc['root_pos_mm']['p95']:.3g}mm "
                  f"joint={acc['joint_rms_deg']['median']:.3g}/{acc['joint_rms_deg']['p95']:.3g}deg "
                  f"ovf={stab['overflow']} ({time.time()-t0:.0f}s)", flush=True)
            del sim
            Path(args.out).write_text(json.dumps(result, indent=1))

    # Reference uncertainty: a second fine integrator from the same states.
    sim = build(args.robot, REF[0], REF[1] / 2, args.nworld)
    q1, v1 = chunked(sim, one_step)
    result["reference_uncertainty"] = {"integrator": REF[0], "timestep": REF[1] / 2, "accuracy": errors(q1, v1, QR, VR)}
    Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

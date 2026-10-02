"""Gait analysis of a legged policy under fixed velocity commands in the reference simulator.

For each command, runs --envs environments for --steps control steps with the deterministic policy and records,
per foot, the floor contact reported by the simulator and the foot-body position (forward kinematics of the
training model). After a --skip transient, and over the environments that stay alive, it reports per foot:

  touchdowns_per_s  number of touchdowns (end of an airborne phase of at least --min-air control steps) per second
  duty              fraction of steps in contact
  swing_s           mean duration of the airborne phases
  clearance_m       mean peak height of the foot body above its median stance height during airborne phases
  stride_m          mean horizontal distance between consecutive touchdowns of the foot
  slip_mps          mean horizontal foot speed while in contact
  drag_frac         share of the foot's horizontal travel that happens while it is in contact (0 for clean steps,
                    1 for a foot that is only dragged)

and for two-legged robots the alternation of touchdowns (share of consecutive touchdowns made by different feet)
and the symmetry (smaller over larger value) of the touchdown counts, strides, swing durations and clearances. The contact timeline and foot heights of the
first environment are stored for gait diagrams.

Usage: python gait_eval.py --ckpt runs/loop/x.pt --out results/loop/gait_x.json [--cmds "0.5 0 0" ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from eval2 import load
from shac2 import build_env, make_actor


def runs(mask):
    """(start, end) index pairs of the True runs of a 1-D boolean array."""
    m = np.concatenate(([False], mask, [False])).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def foot_stats(c, p, dt, min_air):
    """Statistics of one foot: c (T,) contact, p (T, 3) position."""
    T = len(c)
    air = [(s, e) for s, e in runs(~c) if e - s >= min_air]
    # Touchdowns: ends of airborne phases that are followed by contact within the window.
    tds = [e for s, e in air if e < T]
    stance_z = np.median(p[c, 2]) if c.any() else np.min(p[:, 2])
    clear = [float(p[s:e, 2].max() - stance_z) for s, e in air]
    td_pos = p[tds, 0:2] if tds else np.zeros((0, 2))
    stride = np.linalg.norm(np.diff(td_pos, axis=0), axis=-1) if len(tds) > 1 else np.zeros(0)
    dxy = np.linalg.norm(np.diff(p[:, 0:2], axis=0), axis=-1)
    both = c[1:] & c[:-1]
    slip = dxy[both].sum()
    total = dxy.sum()
    return dict(touchdowns=len(tds), touchdown_steps=tds, touchdowns_per_s=len(tds) / (T * dt), duty=float(c.mean()),
                swing_s=float(np.mean([e - s for s, e in air]) * dt) if air else 0.0,
                clearance_m=float(np.mean(clear)) if clear else 0.0, stride_m=float(stride.mean()) if len(stride) else 0.0,
                slip_mps=float(dxy[both].mean() / dt) if both.any() else 0.0,
                drag_frac=float(slip / total) if total > 1e-6 else 0.0)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--skip", type=int, default=100)
    ap.add_argument("--min-air", type=int, default=2)
    ap.add_argument("--cmds", nargs="+", default=["0.5 0 0", "1.0 0 0", "0.5 0 0.5", "0 0.3 0", "0 0 0"])
    a = ap.parse_args()
    ck, cfg = load(a.ckpt)
    over = dict(integrator="implicitfast", timestep=0.001, substeps=round(cfg.control_dt / 0.001), iterations=100,
                ls_iterations=50, geom_solimp=None, geom_margin=None, solref_timeconst=None, cone="elliptic")
    env = build_env(cfg, num_envs=a.envs, seed=99, backward=False, **over)
    meta = ck["meta"]
    actor = make_actor(cfg, meta["obs_dim"], meta["act_dim"]).cuda()
    actor.load_state_dict(ck["actor"])
    mean, var = ck["obs_rms"]["mean"].cuda(), ck["obs_rms"]["var"].cuda()
    squash = meta.get("action_squash", True)
    m = env.sim.mjm
    d = mujoco.MjData(m)
    task = env.task
    feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in task.foot_bodies] + \
           [int(m.geom_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)]) for g in task.foot_geoms]
    names = list(task.foot_bodies) + list(task.foot_geoms)
    dt = env.dt
    rows = []
    for cs in a.cmds:
        cmd = torch.tensor([float(x) for x in cs.split()], device=env.dev)
        env.gen.manual_seed(99)
        env.reset_all()
        env.cmd_resample = 10 ** 9
        env.cmd[:] = cmd
        alive = torch.ones(env.N, dtype=torch.bool, device=env.dev)
        C, P, V = [], [], []
        obs = env.obs()
        for t in range(a.steps):
            u = actor((obs - mean) / torch.sqrt(var + 1e-5), deterministic=True)
            obs, rew, done, info = env.step(torch.tanh(u) if squash else u, differentiable=False)
            env.cmd[:] = cmd
            alive &= ~info["terminated"] & ~done
            if t >= a.skip:
                qs = env.q.cpu().numpy()
                pos = np.zeros((env.N, len(feet), 3))
                for i in range(env.N):
                    d.qpos[:] = qs[i]
                    mujoco.mj_kinematics(m, d)
                    pos[i] = d.xpos[feet]
                P.append(pos)
                C.append(env.contact[:, :len(feet)].cpu().numpy())
                f = env.F(env, env.q, env.v)
                V.append(torch.cat((f.lin_vel_yaw[:, 0:2], f.ang_vel_b[:, 2:3]), -1).cpu().numpy())
        C, P, V = np.stack(C, 1), np.stack(P, 1), np.stack(V, 1)  # (N, T, F[,3])
        ok = alive.cpu().numpy()
        per_env = []
        for i in np.nonzero(ok)[0]:
            fs = [foot_stats(C[i, :, k], P[i, :, k], dt, a.min_air) for k in range(len(feet))]
            e = dict(feet=fs)
            if len(feet) == 2:
                ev = sorted([(s, k) for k in range(2) for s in fs[k]["touchdown_steps"]])
                e["alternation"] = (float(np.mean([ev[j][1] != ev[j + 1][1] for j in range(len(ev) - 1)]))
                                    if len(ev) > 1 else 0.0)
                n0, n1 = fs[0]["touchdowns"], fs[1]["touchdowns"]
                e["count_symmetry"] = min(n0, n1) / max(n0, n1) if max(n0, n1) else 0.0
                for kk, name in (("stride_m", "stride_symmetry"), ("swing_s", "swing_symmetry"),
                                 ("clearance_m", "clearance_symmetry")):
                    s0, s1 = fs[0][kk], fs[1][kk]
                    e[name] = min(s0, s1) / max(s0, s1) if max(s0, s1) > 1e-6 else 0.0
                e["double_support"] = float((C[i, :, 0] & C[i, :, 1]).mean())
                e["min_foot_sep"] = float(np.linalg.norm(P[i, :, 0, 0:2] - P[i, :, 1, 0:2], axis=-1).min())
                e["flight"] = float((~C[i, :, 0] & ~C[i, :, 1]).mean())
            per_env.append(e)
        keys = ("touchdowns_per_s", "duty", "swing_s", "clearance_m", "stride_m", "slip_mps", "drag_frac")
        agg = {"cmd": cmd.tolist(), "survival": float(ok.mean()), "n_alive": int(ok.sum())}
        if per_env:
            agg["feet"] = {names[k]: {kk: float(np.mean([e["feet"][k][kk] for e in per_env])) for kk in keys}
                           for k in range(len(feet))}
            for kk in ("alternation", "count_symmetry", "stride_symmetry", "swing_symmetry", "clearance_symmetry",
                       "double_support", "flight", "min_foot_sep"):
                if kk in per_env[0]:
                    agg[kk] = float(np.mean([e[kk] for e in per_env]))
            vm = V[ok].mean(1)
            agg["v_mean"] = vm.mean(0).tolist()
            agg["err_xy"] = float(np.linalg.norm(vm[:, 0:2] - cmd[0:2].cpu().numpy(), axis=-1).mean())
            agg["err_wz"] = float(np.abs(vm[:, 2] - float(cmd[2])).mean())
            i0 = int(np.nonzero(ok)[0][0])
            n_show = min(C.shape[1], round(4.0 / dt))
            agg["timeline"] = {"dt": dt, "contact": C[i0, :n_show].T.astype(int).tolist(),
                               "height": (P[i0, :n_show, :, 2] - P[i0, :n_show, :, 2].min(0)).T.round(4).tolist()}
        rows.append(agg)
        brief = {k: v for k, v in agg.items() if k not in ("timeline", "feet")}
        brief["feet"] = {n: {k: round(v, 3) for k, v in s.items()} for n, s in agg.get("feet", {}).items()}
        print(json.dumps(brief), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows))


if __name__ == "__main__":
    main()

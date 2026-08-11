"""How sensitive is the grasp to the parameters people normally tune?

If stable grasping needs hours of tuning, the result should move a lot when the
position-correction factor, the regularisation term, or the friction
coefficient change. This measures how much it actually moves.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import warp as wp

sys.path.insert(0, "/home/horde/repos/fpgs-study")
sys.path.insert(0, "/home/horde/repos/newton-tests/src")

import newton  # noqa: E402
from harness import make_env, summarize  # noqa: E402
from exp_gravcomp import run_gc  # noqa: E402


def one(tag, solver_kw=None, mu_cube=None, solver="fpgs", frames=700):
    env = make_env(solver, solver_kw=solver_kw, mu_cube=mu_cube)
    r = run_gc(env, frames, gc=(solver == "fpgs"))
    s = summarize(r)
    z = r["cube_pos"][:, 2]
    m = r["phase"] == 4
    s["z_at_shake"] = float(z[m][0]) if m.any() else None
    s["z_end"] = float(z[-1])
    s["held"] = bool(m.any() and (z[m][0] > 0.10) and (z[-1] > 0.10)
                     and np.linalg.norm(r["rel_pos"][m], axis=1).max() < 0.08)
    np.savez_compressed(f"data/tune_{tag}.npz",
                        **{k: v for k, v in r.items() if k != "failure"},
                        failure=np.asarray(r["failure"]))
    del env
    return s


if __name__ == "__main__":
    wp.init()
    wp.set_device("cuda:0")
    out = {}
    for beta in (0.0, 0.05, 0.2, 0.5, 0.9):
        out[f"beta{beta}"] = one(f"beta{beta}", {"pgs_beta": beta})
        out[f"beta{beta}"]["pgs_beta"] = beta
        print(f"beta{beta}", json.dumps(out[f"beta{beta}"], default=str)[:200], flush=True)
    for cfm in (0.0, 1e-6, 1e-4, 1e-2):
        out[f"cfm{cfm}"] = one(f"cfm{cfm}", {"pgs_cfm": cfm})
        out[f"cfm{cfm}"]["pgs_cfm"] = cfm
        print(f"cfm{cfm}", json.dumps(out[f"cfm{cfm}"], default=str)[:200], flush=True)
    for mu in (0.3, 0.5, 0.8, 1.2, 2.0):
        for solv in ("fpgs", "mujoco"):
            k = f"mu{mu}_{solv}"
            out[k] = one(k, mu_cube=mu, solver=solv)
            out[k]["mu_cube"] = mu
            out[k]["solver"] = solv
            print(k, json.dumps(out[k], default=str)[:200], flush=True)
    print("===JSON===")
    print(json.dumps(out, indent=2, default=str))

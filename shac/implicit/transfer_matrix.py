"""Cross-evaluates seed-0 policies trained at each timestep in simulators with every timestep."""

import json
import subprocess
from pathlib import Path

from run_matrix import PY, ROOT

TRAIN = {"implicit_2ms": "runs/e4/{r}_implicit_2ms_s0.pt", "implicit_5ms": "runs/e4/{r}_implicit_5ms_s0.pt",
         "implicit_10ms": "runs/e4/{r}_implicit_10ms_s0.pt", "implicit_20ms": "runs/e4/{r}_implicit_20ms_s0.pt",
         "implicit_10ms_long": "runs/e5/{r}_implicit_10ms_s0.pt"}
EVAL_DT = (0.02, 0.01, 0.005, 0.002, 0.001)
out = ROOT / "results" / "transfer_matrix.json"
rows = json.loads(out.read_text()) if out.exists() else []
have = {(r["robot"], r["train"], r["eval_dt"]) for r in rows}
for robot in ("ant", "go1"):
    for name, pat in TRAIN.items():
        ck = ROOT / pat.format(r=robot)
        if not ck.exists():
            continue
        for dt in EVAL_DT:
            if (robot, name, dt) in have:
                continue
            tmp = ROOT / "results" / "tmp_transfer.json"
            subprocess.run([PY, str(ROOT / "evaluate.py"), "--checkpoint", str(ck), "--integrator", "implicitfast",
                            "--dt", str(dt), "--num-envs", "128", "--steps", "500", "--seed", "4242", "--out", str(tmp)],
                           cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            d = json.loads(tmp.read_text())
            rows.append({"robot": robot, "train": name, "eval_dt": dt, "speed": d["mean_forward_speed"],
                         "survival": d["survival_fraction"]})
            print(robot, name, dt, round(d["mean_forward_speed"], 2), round(d["survival_fraction"], 2), flush=True)
            out.write_text(json.dumps(rows, indent=1))

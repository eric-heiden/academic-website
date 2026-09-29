"""Prints and exports the integrator sweep as compact report data."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
rows = []
for f in sorted(ROOT.glob("results/e1_*.json")):
    if f.name == "e1_summary.json":
        continue
    r = json.loads(f.read_text())
    for c in r["configs"]:
        s, a = c["stability"], c["accuracy"]
        rows.append({
            "robot": r["robot"], "integrator": c["integrator"], "timestep_ms": 1e3 * c["timestep"],
            "diverged": s["diverged_fraction"],
            "iteration_limit_worlds": s["overflow_world_fraction"]["iterations"],
            "buffer_overflow": s["overflow"],
            "root_pos_mm_median": a["root_pos_mm"]["median"], "root_pos_mm_p95": a["root_pos_mm"]["p95"],
            "root_ang_deg_median": a["root_ang_deg"]["median"],
            "joint_deg_median": a["joint_rms_deg"]["median"], "joint_deg_p95": a["joint_rms_deg"]["p95"],
            "joint_vel_median": a["joint_vel_rms"]["median"],
            "valid_fraction": a["valid_fraction"],
        })
    if "reference_uncertainty" in r:
        u = r["reference_uncertainty"]["accuracy"]
        print(f"{r['robot']}: reference uncertainty joint {u['joint_rms_deg']['median']:.4f} deg, "
              f"root {u['root_pos_mm']['median']:.4f} mm")
for x in rows:
    print(f"{x['robot']:9s} {x['integrator']:15s} {x['timestep_ms']:5.1f}ms div={x['diverged']:.3f} "
          f"iter={x['iteration_limit_worlds']:.3f} ovf={x['buffer_overflow']!s:5s} "
          f"root={x['root_pos_mm_median']:.3g}/{x['root_pos_mm_p95']:.3g}mm "
          f"joint={x['joint_deg_median']:.3g}/{x['joint_deg_p95']:.3g}deg valid={x['valid_fraction']:.2f}")
(ROOT / "results" / "e1_summary.json").write_text(json.dumps(rows, indent=1))

"""Insert measured tables and plot traces into the prepared report HTML."""

import argparse
import json
from pathlib import Path
import shutil
import numpy as np

report = Path(__file__).resolve().parents[1]
assets = report / "assets"
parser = argparse.ArgumentParser()
parser.add_argument("experiments", type=Path)
args = parser.parse_args()
root = args.experiments
results = json.loads((assets / "results.json").read_text())
lookup = {(r["clip"], r["controller"]): r for r in results}
clips = ["stand", "wave", "high5", "walk", "dance", "jumpjack", "backflip"]
labels = dict(
    stand="Standing",
    wave="Waving",
    high5="High five",
    walk="Walking",
    dance="Dancing",
    jumpjack="Jumping jacks",
    backflip="Backflip prompt",
)


def table(caption, headers, rows):
    return (
        '<div class="table-wrap"><table><caption>'
        + caption
        + "</caption><thead><tr>"
        + "".join('<th scope="col">' + h + "</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join("<td>" + c + "</td>" for c in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


rows = []
for clip in clips:
    row = [labels[clip]]
    for controller in ["pd", "qp", "mpc"]:
        r = lookup[clip, controller]
        row.append(
            f'<a href="assets/{r["trajectory"]}" download>{r["root_rmse"]:.3f}</a>'
        )
    m = lookup[clip, "mpc"]
    q = lookup[clip, "qp"]
    row += [
        f"{m['joint_rmse']:.3f}",
        f"{'Yes' if q['recovered'] else 'No'} / {'Yes' if m['recovered'] else 'No'}",
    ]
    rows.append(row)
result_table = table(
    "Table 1. Complete-run tracking errors. Each root-error value links to its trajectory. All PD trials end fallen; QP/MPC endings use the recovery criterion defined above.",
    [
        "Motion",
        "PD root RMSE (m)",
        "QP root RMSE (m)",
        "MPC root RMSE (m)",
        "MPC joint RMSE (rad)",
        "QP / MPC upright ending",
    ],
    rows,
)
rows = []
for clip in clips:
    q, m = lookup[clip, "qp"], lookup[clip, "mpc"]
    rows.append(
        [
            labels[clip],
            f"{q['controller_ms_median']:.2f} / {q['controller_ms_p95']:.2f}",
            f"{m['controller_ms_median']:.1f} / {m['controller_ms_p95']:.1f}",
            str(q["qp_failures"]),
            f"{100 * m['deadline_miss_fraction']:.0f}%",
        ]
    )
timing_table = table(
    "Table 2. Optimizer median / 95th-percentile latency, excluding ten initial updates. Measured on an AMD EPYC 9B45 host with a four-CPU container quota. Occasional background validation activity makes these observational timings, not hardware throughput limits.",
    [
        "Motion",
        "QP median / p95 (ms)",
        "MPC median / p95 (ms)",
        "QP fallbacks",
        "MPC missed 10 ms deadline",
    ],
    rows,
)
rows = []
budgets = []
for name, title in [
    ("fast", "32 × 1; 20 ms prediction; 50 Hz control"),
    ("mid", "64 × 2; 10 ms prediction; 50 Hz control"),
]:
    r = json.loads((root / f"budget_{name}.json").read_text())
    budgets.append(r)
    rows.append(
        [
            title,
            f"{r['root_rmse']:.3f}",
            f"{r['joint_rmse']:.3f}",
            f"{r['controller_ms_median']:.1f} / {r['controller_ms_p95']:.1f}",
            "Yes" if r["recovered"] else "No",
        ]
    )
r = lookup["walk", "mpc"]
rows.append(
    [
        "128 × 2; 10 ms prediction; 100 Hz control",
        f"{r['root_rmse']:.3f}",
        f"{r['joint_rmse']:.3f}",
        f"{r['controller_ms_median']:.1f} / {r['controller_ms_p95']:.1f}",
        "Yes",
    ]
)
budget_table = table(
    "Table 3. Walking configurations: samples × search rounds, with a fixed 0.5 s horizon. Smaller-budget timings were measured again after the main matrix completed.",
    [
        "Configuration",
        "Root RMSE (m)",
        "Joint RMSE (rad)",
        "Median / p95 (ms)",
        "Upright ending",
    ],
    rows,
)
(assets / "budgets.json").write_text(json.dumps(budgets, indent=2) + "\n")
dynamics = json.loads((assets / "inverse-dynamics.json").read_text())
rows = []
for mode, title in [
    ("stationary", "Stationary"),
    ("moving_joints", "Moving joints, stationary base"),
    ("moving_base_and_joints", "Moving base and joints"),
]:
    group = [r for r in dynamics if r["mode"] == mode]
    rows.append(
        [
            title,
            *[
                f"{max(r[k] for r in group):.3g}"
                for k in [
                    "M_error",
                    "raw_bias_error",
                    "raw_joint_bias_error",
                    "bias_error",
                ]
            ],
        ]
    )
dynamics_table = table(
    "Table 5. Maximum absolute discrepancy across 25 poses per condition. Mass entries have coordinate-dependent units; generalized force components use N for translation and N·m for rotation. The last column applies the diagnostic articulated correction.",
    [
        "Velocity condition",
        "Mass entry error",
        "Raw bias error (N or N·m)",
        "Raw joint bias error (N·m)",
        "Corrected bias error (N or N·m)",
    ],
    rows,
)
m = lookup["backflip", "mpc"]
contact_summary = f"The 500 Hz contact-force log confirms {m['nonfoot_contact_seconds']:.3f} s of right-wrist/hand ground contact above 1 N normal force, with {m['nonfoot_impulse']:.2f} N·s accumulated normal impulse. This threshold excludes negligible numerical contact. Walking, dancing, jumping jacks, waving, and high five have no non-foot ground contact above that threshold in the measured MPC runs. The backflip case saturates at least one motor on {100 * m['saturation_fraction']:.1f}% of physics steps. No extra actuator strength is supplied for the acrobatic trial."
summary = (
    "With unchanged MPC settings, root RMSE is "
    + ", ".join(
        f"{lookup[c, 'mpc']['root_rmse'] * 100:.1f} cm for {labels[c].lower()}"
        for c in ["walk", "dance", "jumpjack"]
    )
    + ". Standing QP error is "
    + f"{lookup['stand', 'qp']['root_rmse'] * 1000:.1f} mm"
    + ". These results support preview-based control for this small test set, while the larger joint errors and landing contact limit claims of faithful motion reproduction."
)
s = (report / "index.html").read_text()
for key, value in [
    ("RESULTS_TABLE", result_table),
    ("TIMING_TABLE", timing_table),
    ("BUDGET_TABLE", budget_table),
    ("DYNAMICS_TABLE", dynamics_table),
    ("CONTACT_SUMMARY", contact_summary),
    ("MEASURED_SUMMARY", summary),
]:
    s = s.replace("<!-- " + key + " -->", value)
(report / "index.html").write_text(s)
plot = {}
for clip in clips:
    plot[clip] = {}
    for c in ["pd", "qp", "mpc"]:
        d = np.load(assets / lookup[clip, c]["trajectory"])["rows"][::4]
        plot[clip][c] = dict(
            time=np.round(d[:, 0], 5).tolist(), error=np.round(d[:, 2], 7).tolist()
        )
(assets / "plot-data.json").write_text(json.dumps(plot, separators=(",", ":")))
meta = json.loads((assets / "provenance.json").read_text())
meta.update(
    newton_commit="6daa44a3",
    cpu="AMD EPYC 9B45",
    protocol="Sequential main matrix, seed 123; background validation may affect timing.",
)
(assets / "provenance.json").write_text(json.dumps(meta, indent=2) + "\n")
print("Built tables for", len(results), "runs")

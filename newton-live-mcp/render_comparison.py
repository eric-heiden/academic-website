"""Create accessible report tables from the independently audited comparison JSON."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

METHODS = {
    "live": "Newton MCP",
    "restart": "Restart",
    "ipython": "IPython MCP",
    "ipython_fixed": "Corrected IPython",
}
COHORTS = {
    "existing_primary": "Existing tasks",
    "real_primary": "Real identification",
    "sensitivity": "Corrected-IPython sensitivity",
}
SCENARIOS = {
    "panda": "Panda",
    "allegro": "Allegro",
    "hug": "HUG",
    "panda_calibration": "Synthetic calibration",
    "panda_real": "Real identification",
}


def cell(value, *, number=False, best=False):
    content = html.escape(str(value))
    if best:
        content = f"<strong>{content}</strong>"
    classes = [
        name for name, enabled in (("num", number), ("best-eligible", best)) if enabled
    ]
    return f'<td class="{" ".join(classes)}">{content}</td>'


def table(caption, headings, rows):
    header = "".join(
        f'<th scope="col">{html.escape(heading)}</th>' for heading in headings
    )
    return (
        f'<div class="table-wrap" tabindex="0" role="region" aria-label="{html.escape(caption, quote=True)}">'
        f'<table class="text-table result-table"><caption class="table-caption">{html.escape(caption)}</caption>'
        f"<thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def cost(value, *, tokens=False):
    return (
        "Unavailable"
        if value is None
        else f"{value:,.0f}"
        if tokens
        else f"{value:,.2f}"
    )


def overview(result):
    rows = []
    for cohort in ("existing_primary", "real_primary", "sensitivity"):
        for method, method_label in METHODS.items():
            matches = [
                group
                for group in result["groups"]
                if group["cohort"] == cohort
                and group["condition"] == method
                and group["scope"] == "all"
            ]
            if not matches:
                continue
            group = matches[0]
            totals = group["all_completed_costs"]
            count = group["completed_trials"]
            coverage = group["cost_coverage_completed_trials"]
            token_note = (
                ""
                if coverage["input_output_tokens"] == count
                else f" (complete usage {coverage['input_output_tokens']}/{count})"
            )
            physical = sum(
                bool(row.get("physics_success"))
                for row in result["trials"]
                if row["cohort"] == cohort and row["condition"] == method
            )
            rows.append(
                "<tr>"
                + cell(COHORTS[cohort])
                + cell(method_label)
                + cell(f"{physical}/{group['registered_trials']}", number=True)
                + cell(
                    f"{group['successful_eligible_trials']}/{group['registered_trials']}",
                    number=True,
                )
                + cell(cost(totals["startup_inclusive_seconds"]), number=True)
                + cell(
                    cost(totals["input_output_tokens"], tokens=True) + token_note,
                    number=True,
                )
                + cell(cost(totals["candidate_rollouts"], tokens=True), number=True)
                + "</tr>"
            )
    return table(
        "Table 3. All registered contexts: physical checks, eligible success, and total observed costs including failures.",
        [
            "Task set",
            "Method",
            "Physics pass",
            "Eligible pass",
            "Total time [s] ↓",
            "Input + output tokens ↓",
            "Candidates",
        ],
        rows,
    )


def ratios(result):
    rows = []
    for comparison in result["pairwise"]:
        if comparison["scope"] != "all" or comparison["cohort"] == "sensitivity":
            continue
        values = []
        for metric in (
            "startup_inclusive_seconds",
            "input_output_tokens",
            "uncached_input_output_tokens",
        ):
            stats = comparison["metrics"][metric]
            interval = stats["bootstrap_95_interval"]
            values.append(
                "Unavailable"
                if interval is None
                else f"{1 / stats['geometric_ratio']:.3f} [{1 / interval[1]:.3f}, {1 / interval[0]:.3f}]; n={stats['n']}"
            )
        denominator = METHODS[comparison["denominator_condition"]]
        numerator = METHODS[comparison["numerator_condition"]]
        rows.append(
            "<tr>"
            + cell(COHORTS[comparison["cohort"]])
            + cell(f"{denominator} / {numerator}")
            + "".join(cell(value, number=True) for value in values)
            + "</tr>"
        )
    return table(
        "Table 4. Geometric paired cost ratios and descriptive 95% bootstrap intervals. Ratios below 1 favor the first named method.",
        [
            "Task set",
            "Numerator / denominator",
            "Time ratio",
            "Input + output ratio",
            "Uncached input + output ratio",
        ],
        rows,
    )


def sensitivity(result):
    rows = []
    for comparison in result["pairwise"]:
        if comparison["cohort"] != "sensitivity" or comparison["scope"] not in {
            "all",
            "panda_real",
        }:
            continue
        values = []
        for metric in (
            "startup_inclusive_seconds",
            "input_output_tokens",
            "uncached_input_output_tokens",
        ):
            stats = comparison["metrics"][metric]
            interval = stats["bootstrap_95_interval"]
            values.append(
                "Unavailable"
                if interval is None
                else f"{stats['geometric_ratio']:.3f} [{interval[0]:.3f}, {interval[1]:.3f}]; n={stats['n']}"
            )
        scope = (
            "Seven matched cases"
            if comparison["scope"] == "all"
            else "Real identification"
        )
        label = f"{METHODS[comparison['numerator_condition']]} / {METHODS[comparison['denominator_condition']]}"
        rows.append(
            "<tr>"
            + cell(scope)
            + cell(label)
            + "".join(cell(value, number=True) for value in values)
            + "</tr>"
        )
    return table(
        "Table 5. Exploratory corrected-IPython ratios against matched primary cases. Ratios below 1 favor corrected IPython.",
        [
            "Subset",
            "Numerator / denominator",
            "Time ratio",
            "Input + output ratio",
            "Uncached input + output ratio",
        ],
        rows,
    )


def individual(result):
    sections = []
    for index, cohort in enumerate(COHORTS, 6):
        rows = []
        selected = sorted(
            [row for row in result["trials"] if row["cohort"] == cohort],
            key=lambda row: (
                row["scenario"],
                row["variant"],
                list(METHODS).index(row["condition"]),
            ),
        )
        for row in selected:
            status = (
                "Pass"
                if row["eligible_success"]
                else "Ineligible"
                if not row["eligible"]
                else "Fail"
            )
            physical = (
                "Unavailable"
                if not row.get("quality")
                else "Pass"
                if row.get("physics_success")
                else "Fail"
            )
            best = row.get("best_eligible_metrics", [])
            values = (
                "<tr>"
                + cell(f"{SCENARIOS[row['scenario']]} {row['variant']}")
                + cell(METHODS[row["condition"]])
                + cell(physical)
                + cell(status)
            )
            for metric in (
                "startup_inclusive_seconds",
                "input_output_tokens",
                "uncached_input_output_tokens",
            ):
                tokens = metric.endswith("tokens")
                available = (
                    row.get(metric) if not tokens or row.get("usage_complete") else None
                )
                values += cell(
                    cost(available, tokens=tokens), number=True, best=metric in best
                )
            values += cell(row.get("candidate_rollouts", "Unavailable"), number=True)
            values += cell(
                row.get("simulation_process_starts", "Unavailable"), number=True
            )
            values += cell(row.get("failed_candidates", "Unavailable"), number=True)
            values += f'<td><a href="data/v2/trials/{html.escape(row["id"], quote=True)}/summary.json">Evidence</a></td></tr>'
            rows.append(values)
        sections.append(
            f'<details class="development-detail"><summary>{COHORTS[cohort]}: every registered result</summary>'
            + table(
                f"Table {index}. {COHORTS[cohort]}. Time includes startup; verifier processes are excluded from the simulation-start count.",
                [
                    "Case",
                    "Method",
                    "Physics",
                    "Study",
                    "Time [s] ↓",
                    "Input + output ↓",
                    "Uncached + output ↓",
                    "Candidates",
                    "Sim starts",
                    "Failed candidates",
                    "Record",
                ],
                rows,
            )
            + "</details>"
        )
    return "\n".join(sections)


def failure_details(result):
    """Expose narrow numerical misses without conflating them with infrastructure."""
    names = {
        "max_joint_torque_rmse_nm": "maximum joint torque RMSE [N·m]",
        "max_joint_torque_normalized_rmse": "maximum normalized joint torque RMSE",
        "max_joint_position_rmse_rad": "maximum joint position RMSE [rad]",
        "max_joint_velocity_rmse_rad_s": "maximum joint velocity RMSE [rad/s]",
        "position_p95_rad": "95th percentile position error [rad]",
        "max_joint_speed_rad_s": "maximum joint speed [rad/s]",
    }
    records = []
    for row in result["trials"]:
        if row["eligible_success"]:
            continue
        measured = row.get("quality") or {}
        reasons = []
        for group, values in [
            ("pooled", measured),
            *[
                (f"recording {item.get('episode', '?')}", item)
                for item in measured.get("per_episode", [])
            ],
        ]:
            for key, limit in measured.get("thresholds", {}).items():
                value = values.get(key)
                if (
                    isinstance(value, (int, float))
                    and math.isfinite(value)
                    and value > limit
                ):
                    reasons.append(
                        f"{group}: {names.get(key, key)} {value:.9g} exceeds {limit:.9g}"
                    )
        if measured.get("finite") is not True:
            reasons.append("finite-trajectory requirement not established")
        if measured.get("sample_count") != measured.get("expected_frames"):
            reasons.append(
                f"complete trajectory not established ({measured.get('sample_count')} / {measured.get('expected_frames')} samples)"
            )
        reasons.extend(row.get("issues", []))
        reasons.extend(
            reason
            for reason in row.get("failures", [])
            if reason != "Independent physical quality or matching training failed"
        )
        if not reasons:
            reasons.append(
                "matching training, physical validity, or other study-success requirement failed; see the full record"
            )
        label = f"{SCENARIOS[row['scenario']]} {row['variant']} · {METHODS[row['condition']]}"
        records.append(
            f'<li><strong>{html.escape(label)}.</strong> {html.escape("; ".join(reasons))}. <a href="data/v2/trials/{html.escape(row["id"], quote=True)}/summary.json">Record</a>.</li>'
        )
    if not records:
        return "<p>Every registered context passed the physical and study-eligibility requirements.</p>"
    return (
        '<details class="development-detail"><summary>Unsuccessful contexts: numerical and eligibility details</summary><p>These are the unchanged registered gates. A small threshold exceedance is reported at its numerical size; it is not evidence of a large practical quality difference.</p><ul>'
        + "".join(records)
        + "</ul></details>"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = json.loads(args.comparison.read_text())
    if result["completed_trials"] != result["registered_trials"]:
        raise ValueError("Refusing to generate final tables from incomplete trials")
    rendered = (
        overview(result)
        + "\n"
        + ratios(result)
        + "\n"
        + sensitivity(result)
        + "\n"
        + '<p class="table-note">Physical checks include matching logged training and fresh verifiers where applicable; eligible success additionally requires valid budgets, process completion, and integrity. Paired ratios include only jointly successful eligible cases. The intervals resample registered pairs and are unadjusted descriptive comparisons; small samples and conditioning on success limit inference. Total costs are comparable only within the same task set. Corrected-IPython runs followed the primary block and cover fewer cases; their paired results are exploratory and may reflect run-order drift or agent variation. Bold, shaded values in individual tables identify the lowest eligible same-case costs; ties at displayed precision share emphasis. Sensitivity rows have no within-cohort competitors.</p>'
        + "\n"
        + individual(result)
    )
    args.output.write_text(rendered + "\n" + failure_details(result) + "\n")


if __name__ == "__main__":
    main()

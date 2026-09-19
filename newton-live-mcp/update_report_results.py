"""Render exact original-task tables from audited report data, without simulation."""

import argparse
import html
import json
from pathlib import Path


ROOT = Path(__file__).parent
START = "<!-- ORIGINAL_RESULTS:START -->"
END = "<!-- ORIGINAL_RESULTS:END -->"
LABELS = {"panda": "Panda", "allegro": "Allegro", "hug": "HUG"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    data = json.loads((ROOT / "data/confirmation/results.json").read_text())
    if args.require_complete and not data["all_registered_trials_complete"]:
        raise ValueError("Every registered result must be complete before final rendering")
    rows = [row for row in data["trials"] if row["study"] == "original"]
    by_id = {row["run_id"]: row for row in rows}
    pairs = [pair for pair in data["pairs"] if not pair["pair_id"].startswith("panda_calibration")]
    completed = [row for row in rows if row["status"] == "completed"]
    eligible = [row for row in completed if row["eligible_success"]]
    full = len(completed) == 12
    body = []
    for pair in pairs:
        members = [by_id[pair[condition]] for condition in ("live", "restart")]
        body.append("<tbody>")
        for row in members:
            task = f"{LABELS[row['scenario']]} {row['variant']}"
            cells = [f'<th scope="row">{task}</th>', f"<td>{row['condition'].title()}</td>"]
            if row["status"] != "completed":
                cells.append('<td colspan="5">Pending registered trial</td>')
            else:
                for key, formatting in (
                    ("startup_inclusive_seconds", ".2f"),
                    ("input_tokens", ","),
                    ("output_tokens", ","),
                ):
                    value = format(row[key], formatting)
                    best = pair["both_eligible_successes"] and float(value.replace(",", "")) == min(
                        float(format(member[key], formatting).replace(",", "")) for member in members
                    )
                    cells.append(f'<td class="num{" metric-best" if best else ""}">{"<strong>" + value + "</strong>" if best else value}</td>')
                cells.append(f'<td class="num">{row["candidate_rollouts"]}</td>')
                key, multiplier, unit = ("wrist_rmse_m", 1000, "mm") if row["scenario"] == "hug" else ("tracking_rmse_rad", 1000, "mrad")
                error = row["quality"].get(key)
                text = f"{error * multiplier:.3f} {unit}" if error is not None else "Unavailable"
                if not row["eligible_success"]:
                    text += " · failed eligibility"
                cells.append(f'<td class="num">{html.escape(text)}</td>')
            body.append("<tr>" + "".join(cells) + "</tr>")
        body.append("</tbody>")
    intro = (
        f"All {len(eligible)} original-task agents passed physical quality and the complete eligibility checks."
        if full and len(eligible) == 12
        else f"{len(completed)} of the 12 original-task results are complete; {len(eligible)} pass every eligibility check. Pending entries remain visible."
    )
    successful_pairs = [pair for pair in pairs if pair["both_eligible_successes"]]
    joint_wins = sum(
        pair["ratios_restart_over_live"]["startup_inclusive_seconds"] > 1
        and pair["ratios_restart_over_live"]["input_output_tokens"] > 1
        for pair in successful_pairs
    )
    time_wins = sum(pair["ratios_restart_over_live"]["startup_inclusive_seconds"] > 1 for pair in successful_pairs)
    totals = {item["condition"]: item for item in data["condition_totals"] if item["study"] == "original"}
    summary = ""
    if full and len(eligible) == 12:
        live, restart = totals["live"]["sums"], totals["restart"]["sums"]
        summary = (
            f'<p>Across the six original-task pairs, startup-inclusive time totals {live["startup_inclusive_seconds"]:.2f} s for live and '
            f'{restart["startup_inclusive_seconds"]:.2f} s for restart, a {100 * (1 - live["startup_inclusive_seconds"] / restart["startup_inclusive_seconds"]):.2f}% reduction. '
            f'Input plus output totals {live["input_output_tokens"]:,} versus {restart["input_output_tokens"]:,} tokens, '
            f'{100 * (1 - live["input_output_tokens"] / restart["input_output_tokens"]):.2f}% lower with live. '
            'These sums include every original-task agent and are separate from calibration totals.</p>'
        )
    interpretation = (
        f'<p>Live is faster in {time_wins} of {len(successful_pairs)} completed eligible pairs, and reduces both elapsed time and input plus output in {joint_wins}. '
        'Panda variant 0 is a counterexample to uniform token savings: live uses 137,115 input tokens versus 125,401 for restart, '
        'despite a shorter elapsed time. Most original-task agents select a passing controller with one candidate. '
        'Allegro live variant 0 evaluates two candidates; HUG restart variant 1 evaluates three, including a failed initial candidate and two passing repeats. All attempts remain counted. HUG live variant 1’s unusually long 11.61 s startup is included in its 66.36 s total. '
        'The small scenes and short tasks leave model/tool latency as a substantial part of elapsed time.</p>'
    )
    block = f'''{START}
        <h3 id="original-results">Original tracking and replay comparisons</h3>
        <p>{intro} Table 4 retains all twelve registered original-task entries; calibration appears separately below.</p>
        <div class="table-wrap" tabindex="0" role="region" aria-label="All twelve original tracking and replay results"><table class="result-table"><caption class="table-caption">Table 4. Original tracking and replay tasks. Time includes startup; ↓ indicates lower values. Joint RMSE is in milliradians for Panda and Allegro; HUG wrist RMSE is in millimeters. Every displayed eligible result also meets all other task criteria.</caption><thead><tr><th scope="col">Task / variant</th><th scope="col">Method</th><th class="num" scope="col">Time ↓<br>s</th><th class="num" scope="col">Input ↓<br>tokens</th><th class="num" scope="col">Output ↓<br>tokens</th><th class="num" scope="col">Candidates</th><th class="num" scope="col">Final RMSE</th></tr></thead>{''.join(body)}</table></div>
        <p class="table-note">Bold shading marks lower time/token values within an eligible pair, with ties at displayed precision marked equally. Physical errors are shown without ranking controllers that already meet the full criteria. Exact quality, cached/uncached input and all other measurements remain in <a href="data/confirmation/results.json">the complete results</a>.</p>
        {summary}
        {interpretation}
{END}'''
    page = (ROOT / "index.html").read_text()
    if START in page:
        before, after = page.split(START, 1)
        _, tail = after.split(END, 1)
        page = before + block + tail
    else:
        page = page.replace('        <h3 id="calibration">', block + '\n        <h3 id="calibration">', 1)
    (ROOT / "index.html").write_text(page)
    print(json.dumps({"completed_original_results": len(completed), "eligible_original_results": len(eligible), "time_and_total_token_wins": joint_wins}))


if __name__ == "__main__":
    main()

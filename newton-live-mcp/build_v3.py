"""Build index.html for the visual-calibration study from exported data (run export_v3.py first)."""

from __future__ import annotations

import html
import json
from pathlib import Path

from charts_v3 import CONDITIONS, MODELS, STYLE, TASKS, dumbbell, legend, ratio_chart
from export_v3 import pairs

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "v3"
NARRATIVE = json.loads((HERE / "narrative_v3.json").read_text())


def esc(text) -> str:
    return html.escape(str(text))


def table(caption: str, head: list[str], rows: list[list[str]], numeric: set[int] = frozenset(), label: str | None = None, prose: bool = False) -> str:
    thead = "".join(f'<th scope="col">{esc(h)}</th>' for h in head)
    body = "".join(
        "<tr>" + "".join(f'<td class="{"num" if i in numeric else ""}">{cell}</td>' for i, cell in enumerate(row)) + "</tr>" for row in rows
    )
    return (
        f'<div class="table-wrap" tabindex="0" role="region" aria-label="{esc(label or caption)}"><table class="text-table{"" if prose else " result-table"}">'
        f'<caption class="table-caption">{esc(caption)}</caption><thead><tr>{thead}</tr></thead><tbody>{body}</tbody></table></div>'
    )


def fmt_tokens(value) -> str:
    return f"{value / 1e6:.2f}M" if value >= 1e6 else f"{value / 1e3:.0f}k"


def ratio_text(value) -> str:
    if not value:
        return "–"
    return f"{value['ratio']:.2f} [{value['ci95'][0]:.2f}, {value['ci95'][1]:.2f}]; n={value['n']}"


def figure(src: str, alt: str, number: int, caption: str, width: int, height: int, max_width: int | None = None) -> str:
    style = f' style="max-width:{max_width}px"' if max_width else ""
    return (
        f'<figure class="evidence-figure"><img src="{src}" width="{width}" height="{height}"{style} loading="lazy" alt="{esc(alt)}">'
        f'<figcaption class="figure-caption"><span class="caption-label">Figure {number}.</span> {caption}</figcaption></figure>'
    )


def section(section_id: str, number: int, title: str, body: str) -> str:
    return (
        f'<section class="report-section" id="{section_id}"><div class="section-heading"><div class="section-title-wrap">'
        f'<h2><span class="section-number">{number}.</span> {esc(title)}</h2></div></div>{body}</section>'
    )


def paragraphs(key: str) -> str:
    return "".join(f"<p>{text}</p>" for text in NARRATIVE[key])


def followup_table() -> str:
    path = DATA / "followup-analysis.json"
    if not path.exists():
        return ""
    analysis = json.loads(path.read_text())
    rows = []
    for pair in pairs(analysis):
        for condition, name in (("mcp_workers", "MCP + 3 workers"), ("restart", "Edit/restart")):
            r = pair.get(condition)
            if not r:
                continue
            worst = r["normalized_worst"] or {}
            rows.append(
                [
                    f'{TASKS[pair["task"]]} · {MODELS[pair["model"]]}',
                    name,
                    "Pass" if r["success"] else ("Timeout" if r["timed_out"] else "Fail"),
                    f"{max(worst.values()):.2f}" if worst else "–",
                    f'{r["seconds"]:,.0f}',
                    fmt_tokens(r["input_tokens"]) if r["input_tokens"] else "lost",
                    str(r["tool_calls"]),
                    str(r["simulator_processes"]),
                    f'<a href="data/v3/followup/{r["trial"]}/summary.json">JSON</a>',
                ]
            )
    paired = analysis["paired_mcp_over_restart"]["all"]["all_pairs"]
    caption = (
        "Table 6. Exploratory follow-up, one pair per task and model (separate registration, run after the confirmation). "
        f"Paired MCP-with-workers/restart ratios over all {analysis['paired_mcp_over_restart']['all']['pairs']} pairs: "
        f"time {ratio_text(paired['seconds'])}, input tokens {ratio_text(paired['input_tokens'])}, tool calls {ratio_text(paired['tool_calls'])}."
    )
    return table(caption, ["Case", "Condition", "Result", "Quality", "Time [s]", "Input tok", "Tool calls", "Sim processes", "Record"], rows, numeric={3, 4, 5, 6, 7})


def build() -> str:
    analysis = json.loads((DATA / "analysis.json").read_text())
    rows = analysis["trials"]
    grouped = pairs(analysis)
    paired = analysis["paired_mcp_over_restart"]
    successes = {c: sum(r["success"] for r in rows if r["condition"] == c) for c in CONDITIONS}
    counts = {c: sum(r["condition"] == c for r in rows) for c in CONDITIONS}

    # Table: all trials.
    trial_rows = []
    for pair in grouped:
        for condition in CONDITIONS:
            r = pair.get(condition)
            if not r:
                continue
            status = "Pass" if r["success"] else ("Timeout" if r["timed_out"] else "Fail")
            worst = r["normalized_worst"] or {}
            quality = max(worst.values()) if worst else None
            trial_rows.append(
                [
                    f'{TASKS[pair["task"]]} · {MODELS[pair["model"]]} · {pair["replicate"]}',
                    CONDITIONS[condition],
                    status,
                    f"{quality:.2f}" if quality is not None else "–",
                    f'{r["seconds"]:,.0f}',
                    fmt_tokens(r["input_tokens"]) if r["input_tokens"] else "lost",
                    f'{r["output_tokens"]:,}',
                    str(r["tool_calls"]),
                    str(r["images_in_context"]),
                    str(r["parameter_sets"]),
                    str(r["simulator_processes"]),
                    f'{r["first_passing_seconds"]:,.0f}' if r.get("first_passing_seconds") is not None else "–",
                    f'<a href="data/v3/confirmation/{r["trial"]}/summary.json">JSON</a>',
                ]
            )
    all_trials = table(
        "Table 3. Every registered confirmation trial. Quality is the largest held-out error divided by its limit (≤ 1 passes). "
        "Time includes live-application startup for MCP trials. Images viewed: images returned by any tool (Codex has no file-viewing tool, so its restart trials count none). First pass: seconds until the agent first simulated a parameter set that passes held-out verification (post-hoc).",
        ["Case", "Condition", "Result", "Quality", "Time [s]", "Input tok", "Output tok", "Tool calls", "Images viewed", "Param sets", "Sim processes", "First pass [s]", "Record"],
        trial_rows,
        numeric={3, 4, 5, 6, 7, 8, 9, 10, 11},
    )

    # Table: paired ratios.
    ratio_rows = []
    labels = {"all": "All pairs", "opus": "Opus 5.5", "astra": "GPT-6 Astra", "arm_offsets": "Arm", "cloth_drape": "Cloth", "push": "Push"}
    for scope, name in labels.items():
        data = paired[scope]
        every, joint = data["all_pairs"], data["jointly_successful"]
        ratio_rows.append(
            [
                name,
                f'{data["mcp_successes"]}/{data["pairs"]} · {data["restart_successes"]}/{data["pairs"]}',
                ratio_text(every["seconds"]),
                ratio_text(joint["seconds"]),
                ratio_text(every["input_tokens"]),
                ratio_text(joint["input_tokens"]),
                ratio_text(every["tool_calls"]),
            ]
        )
    ratios = table(
        "Table 4. Geometric-mean MCP/restart ratios with 95% bootstrap intervals (20,000 resamples of pairs). Below 1 favors MCP. "
        "“All” uses every registered pair (token ratios exclude pairs without a usage record); “joint” only pairs in which both trials passed.",
        ["Subset", "Passes MCP · restart", "Time (all)", "Time (joint)", "Input tokens (all)", "Input tokens (joint)", "Tool calls (all)"],
        ratio_rows,
    )
    time_groups = [(labels[s], paired[s]["all_pairs"]["seconds"]) for s in labels]
    token_groups = [(labels[s], paired[s]["all_pairs"]["input_tokens"]) for s in labels]

    charts = (
        '<div id="result-charts">'
        + STYLE
        + legend()
        + dumbbell(grouped, "seconds", "s", "Elapsed time per registered pair [s, log scale]", "Paired elapsed time for each registered pair")
        + dumbbell(grouped, "input_tokens", "tokens", "Input tokens per registered pair [log scale]", "Paired input tokens for each registered pair")
    )
    ratio_charts = (
        ratio_chart(time_groups, "Elapsed-time ratio MCP / restart, all registered pairs (geometric mean, 95% interval)", "Paired time ratios")
        + ratio_chart(token_groups, "Input-token ratio MCP / restart, all pairs with usage records", "Paired input-token ratios")
        + "</div>"
    )

    registration = json.loads((DATA / "REGISTRATION.json").read_text())
    n = NARRATIVE
    head = (HERE / "index_head_v3.html").read_text()
    body = [
        head.replace("{{ABSTRACT}}", n["abstract"]).replace("{{MODIFIED}}", n["modified"]),
        section("question", 1, "Question and design", paragraphs("design")
                + table("Table 1. Conditions. Both use the same task module, reference photos, parameter bounds, held-out verifier, and 30-minute budget.",
                        ["Condition", "Access", "Simulator lifetime"],
                        [["Newton MCP", "Live application through the Newton MCP (execute, observe, filmstrip, describe, rebuild); the agent may also write and run scripts.", "One application process for the trial; scripts it writes start their own processes."],
                         ["Edit/restart", "Edit params.json or write scripts; <code>sim.py</code> renders the training episodes and writes simulated | reference | mismatch sheets.", "A fresh process for every script run."]], prose=True)),
        section("mcp", 2, "What changed in the MCP", paragraphs("mcp")
                + figure("assets/v3/cloth_drape-filmstrip.png", "Filmstrip from the MCP: three simulated frames of an untuned cloth over a box, the matching noisy reference photos, and magenta mismatch panels.", 1,
                         "Actual <code>newton_filmstrip</code> output for untuned default cloth parameters: simulated row, reference photos, and mismatch (magenta: max channel difference above 24/255). One tool call; the image arrives inline.", 1448, 1088)
                + figure("assets/v3/push-views.jpg", "Four auto-framed views of the pushing scene in one image.", 2,
                         "The four panels of one <code>newton_observe(views=[\"iso\", \"top\", \"front\", \"right\"])</code> response, auto-framed without any camera parameters. At the registered commit the tool stacked views vertically, and they are arranged 2 × 2 here for display; it now tiles them in a grid itself.", 644, 484, 644)),
        section("tasks", 3, "Calibration tasks from photographs", paragraphs("tasks")
                + table("Table 2. Tasks, hidden parameters, and frozen held-out gates. Gates were set from private perturbation studies before any agent trial.",
                        ["Task", "Parameters (clean-slate start)", "Evidence given", "Held-out gate"],
                        n["task_rows"], prose=True)
                + figure("assets/v3/push-sheet_push_b.jpg", "Reference sheet for a training push: top and side views at six times.", 3,
                         "One of the reference sheets agents receive (push, episode B): photos from the calibrated top and side cameras at six times, with fixed-seed sensor noise.", 2900, 724)
                + figure("assets/v3/arm_offsets-sheet_pose_2.jpg", "Reference sheet for one Panda pose from three cameras.", 4,
                         "Panda reference photos for one commanded pose (front, side, top). Agents must infer seven joint zero offsets.", 480, 1088, 360)),
        section("development", 4, "Development pilots and what they changed", paragraphs("development")),
        section("results", 5, "Registered results", paragraphs("results") + charts + ratio_charts + ratios
                + f'<details class="development-detail" open><summary>All {len(rows)} confirmation trials</summary>{all_trials}</details>'
                + paragraphs("results_detail")),
        section("when", 6, "When does the live MCP pay off?", paragraphs("when")
                + table("Table 5. Warm-cache start-up of stock Newton examples on the study machine (two frames, null viewer). A restart pays this on every script run; the live MCP pays it once.",
                        ["Scene", "Start-up [s]"], n["startup_rows"], numeric={1})),
        section("followup", 7, "Exploratory follow-up: parallel live workers", paragraphs("followup") + followup_table()),
        section("limits", 8, "Limits", paragraphs("limits")),
        section("source", 9, "Source and reproduction", paragraphs("source").replace("{{COMMIT}}", registration["commit"])),
    ]
    footer = (
        '<footer class="report-footer"><span>Newton live simulation · experimental research</span>'
        '<a href="index.html">Harness improvement loop (v4, latest)</a><a href="study2.html">Three-way study (v2)</a><a href="historical.html">Original study</a><a href="../">All reports</a></footer>'
        "</main></div></body></html>"
    )
    return "".join(body) + footer


if __name__ == "__main__":
    (HERE / "study3.html").write_text(build())
    print("wrote study3.html")

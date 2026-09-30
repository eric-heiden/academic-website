"""Build index.html for the harness-improvement loop (run export_v4.py first)."""

from __future__ import annotations

import json
from pathlib import Path

from build_v3 import esc, figure, fmt_tokens, ratio_text, section, table
from charts_v3 import STYLE
from charts_v4 import CONDITIONS, MODELS, TASKS, dumbbell, legend, ratio_chart
from export_v4 import pairs

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "v4"
NARRATIVE = json.loads((HERE / "narrative_v4.json").read_text())


def paragraphs(key: str) -> str:
    return "".join(f"<p>{text}</p>" for text in NARRATIVE[key])


def trial_table(rows: list[dict]) -> str:
    body = []
    for pair in pairs(rows):
        for condition in CONDITIONS:
            r = pair.get(condition)
            if not r:
                continue
            worst = r["normalized_worst"] or {}
            status = "Pass" if r["success"] else ("Timeout" if r["timed_out"] else "Fail")
            if r.get("reverified_from") is not None and r["reverified_from"] != r["success"]:
                status += " (re-verified)"
            body.append(
                [
                    f'{pair["iteration"]} · {TASKS[pair["task"]]} · {MODELS[pair["model"]]} · {pair["replicate"]}',
                    CONDITIONS[condition],
                    status,
                    f"{max(worst.values()):.2f}" if worst else "–",
                    f'{r["seconds"]:,.0f}',
                    f'{r["model_seconds"]:,.0f}' if r.get("model_seconds") is not None else "–",
                    f'{r["tool_seconds"]:,.0f}' if r.get("tool_seconds") is not None else "–",
                    fmt_tokens(r["input_tokens"]) if r["input_tokens"] else "–",
                    f'{r["output_tokens"]:,}' if r["output_tokens"] else "–",
                    f'${r["cost_usd"]:.2f}' if r.get("cost_usd") else "–",
                    str(r["tool_calls"]),
                    str(r["mcp_calls"]) if condition == "mcp" else "–",
                ]
            )
    return table(
        "Table 3. Every loop trial. Quality is the largest verified error divided by its limit (≤ 1 passes). Model and tool "
        "time split the elapsed time by event timestamps (tool time is the union of tool-execution intervals). Cost is the "
        "API list price reported by Claude Code; Codex does not report cost.",
        ["Trial", "Condition", "Result", "Quality", "Time [s]", "Model [s]", "Tools [s]", "Input tok", "Output tok", "Cost", "Tool calls", "MCP calls"],
        body,
        numeric={3, 4, 5, 6, 7, 8, 9, 10, 11},
        label="All loop trials",
    )


def scope_name(scope: str) -> str:
    iteration, _, model = scope.partition(":")
    name = "All iterations" if iteration == "all" else f"{iteration} ({NARRATIVE['harness_names'].get(iteration, iteration)})"
    return f"{name} · {MODELS[model]}" if model else name


def iteration_table(summary: dict) -> str:
    rows = []
    for scope, group in summary.items():
        r = group["ratios"]
        rows.append(
            [
                scope_name(scope),
                f'{group["mcp_pass"]}/{group["pairs"]} · {group["restart_pass"]}/{group["pairs"]}',
                ratio_text(r["seconds"]),
                ratio_text(r["input_tokens"]),
                ratio_text(r["output_tokens"]),
                ratio_text(r["tool_calls"]),
                ratio_text(r["cost_usd"]),
            ]
        )
    return table(
        "Table 2. Paired MCP/no-MCP ratios per harness iteration (geometric mean over pairs, 95% bootstrap interval). "
        "Below 1 favors the MCP. Cost ratios cover Claude Code pairs only.",
        ["Iteration", "Passes MCP · no MCP", "Time", "Input tokens", "Output tokens", "Tool calls", "Cost"],
        rows,
        label="Paired ratios per iteration",
    )


def build() -> str:
    loop = json.loads((DATA / "loop.json").read_text())
    rows, summary = loop["trials"], loop["summary"]
    grouped = [p for p in pairs(rows) if "mcp" in p and "restart" in p]
    scopes = [scope for scope in summary if ":" in scope]
    short = {"opus": "Opus", "astra": "Astra"}

    def label(scope):
        iteration, _, model = scope.partition(":")
        return f"{'all' if iteration == 'all' else iteration} · {short[model]}"

    charts = (
        '<div id="loop-charts">'
        + STYLE
        + ratio_chart(
            [(label(s), summary[s]["ratios"]["input_tokens"]) for s in scopes],
            "Input-token ratio MCP / no MCP per iteration and agent (geometric mean, 95% interval)",
            "Paired input-token ratios per iteration and agent",
            hi=2.6,
        )
        + ratio_chart(
            [(label(s), summary[s]["ratios"]["seconds"]) for s in scopes],
            "Elapsed-time ratio MCP / no MCP per iteration and agent (geometric mean, 95% interval)",
            "Paired time ratios per iteration and agent",
            hi=2.6,
        )
        + legend()
        + dumbbell(grouped, "input_tokens", "tokens", "Input tokens per pair [log scale]", "Paired input tokens per trial pair")
        + dumbbell(grouped, "seconds", "s", "Elapsed time per pair [s, log scale]", "Paired elapsed time per trial pair")
        + "</div>"
    )
    n = NARRATIVE
    head = (HERE / "index_head_v4.html").read_text()
    body = [
        head.replace("{{ABSTRACT}}", n["abstract"]).replace("{{MODIFIED}}", n["modified"]),
        section("loop", 1, "The improvement loop", paragraphs("loop")
                + table("Table 1. Loop tasks. Every submission is verified in a fresh process against checks the agent cannot see.",
                        ["Task", "Kind", "Starting problem", "Verified goal"], n["task_rows"], prose=True, label="Loop tasks")),
        section("results", 2, "MCP versus no MCP, per iteration", paragraphs("results") + iteration_table(summary) + charts
                + f'<details class="development-detail"><summary>All {len(rows)} trials</summary>{trial_table(rows)}</details>'),
        section("harness", 3, "What each iteration changed", paragraphs("harness")
                + table("Table 4. Harness versions. Each version is a commit on the Newton branch; trials record the version they ran.",
                        ["Version", "Commit", "Changes"], n["harness_rows"], prose=True, label="Harness versions")
                + figure("assets/v4/grinding-overlay.jpg", "Two observation panels of the grinding scene: an isometric view and a top view in which a groove is visible in the workpiece.", 1,
                         "An h5 observation of a solved grinding scene (iso and top views). The ground workpiece surface is drawn by the example's own <code>render()</code>; before h5, observations showed only the wheel.", 964, 320)),
        section("real", 4, "Real-robot calibration data", paragraphs("real")
                + table("Table 5. Held-out window error of reference parameter sets for the DFKI double pendulum (mean joint-angle RMSE over 0.5 s open-loop windows).",
                        ["Parameters", "20 s excitation [rad]", "75 s full swings [rad]"], n["dp_reference_rows"], numeric={1, 2}, label="Double pendulum reference errors")
                + paragraphs("real_cube")
                + table("Table 6. Held-out error of reference contact models for the ContactNets cube tosses (170 tosses; mean over tosses of the time-averaged error, full open-loop rollout from release).",
                        ["Contact model", "Position [m]", "Orientation [rad]"], n["cube_reference_rows"], numeric={1, 2}, label="Cube toss reference errors")),
        section("findings", 5, "Findings so far", paragraphs("findings")),
        section("limits", 6, "Limits", paragraphs("limits")),
        section("source", 7, "Source and data", paragraphs("source")),
    ]
    footer = (
        '<footer class="report-footer"><span>Newton live simulation · experimental research</span>'
        '<a href="study3.html">Visual calibration study (v3)</a><a href="study2.html">Three-way study (v2)</a>'
        '<a href="historical.html">Original study</a><a href="../">All reports</a></footer>'
        "</main></div></body></html>"
    )
    return "".join(body) + footer


if __name__ == "__main__":
    (HERE / "index.html").write_text(build())
    print("wrote index.html")

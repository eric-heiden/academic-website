"""Build the short summary page (index.html) of the MCP improvement loop; the full development log is log.html.

Run export_v4.py first. Numbers come from data/v4/loop.json; the prose lives in summary.json.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from build_v3 import esc, section, table
from charts_v3 import STYLE
from charts_v4 import TASKS, ratio_chart
from export_v4 import paired_ratio, pairs

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "v4"
TEXT = json.loads((HERE / "summary.json").read_text())


def ratio(value: dict | None) -> str:
    if not value:
        return "–"
    return f'{value["ratio"]:.2f}×'


def iteration_key(name: str) -> int:
    return int(name[1:])


def iteration_rows(rows: list[dict], summary: dict) -> list[list[str]]:
    tasks = defaultdict(set)
    for row in rows:
        tasks[row["iteration"]].add(row["task"])
    seen = set()
    out = []
    for iteration in sorted((s for s in summary if ":" not in s and s != "all"), key=iteration_key):
        group = summary[iteration]
        r = group["ratios"]
        new = sorted(tasks[iteration] - seen)
        seen |= tasks[iteration]
        names = ", ".join(("<b>" if t in new else "") + TASKS.get(t, t) + ("</b>" if t in new else "") for t in sorted(tasks[iteration]))
        out.append(
            [
                f"{iteration}<br><span class=\"muted\">{TEXT['iterations'][iteration]['harness']}</span>",
                TEXT["iterations"][iteration]["change"],
                names,
                f'{group["pairs"]}',
                f'{group["mcp_pass"]} / {group["restart_pass"]}',
                ratio(r["seconds"]),
                ratio(r["input_tokens"]),
                ratio(r.get("cost_usd")),
            ]
        )
    return out


def effect_rows(rows: list[dict]) -> list[list[str]]:
    """All iterations pooled by task, sorted by the time ratio."""
    groups = defaultdict(list)
    for pair in pairs(rows):
        if "mcp" in pair and "restart" in pair:
            groups[pair["task"]].append(pair)
    out = []
    for task, group in groups.items():
        time = paired_ratio([(p["mcp"]["seconds"], p["restart"]["seconds"]) for p in group])
        tokens = paired_ratio([(p["mcp"]["input_tokens"], p["restart"]["input_tokens"]) for p in group])
        out.append(
            (
                time["ratio"] if time else 9,
                [
                    esc(TASKS.get(task, task)),
                    str(len(group)),
                    f'{sum(p["mcp"]["success"] for p in group)} / {sum(p["restart"]["success"] for p in group)}',
                    f'{ratio(time)} <span class="muted">({time["ci95"][0]:.2f}\u2013{time["ci95"][1]:.2f})</span>' if time else "\u2013",
                    f'{time["mcp_better"]} of {time["n"]}' if time else "\u2013",
                    ratio(tokens),
                ],
            )
        )
    return [row for _, row in sorted(out, key=lambda item: item[0])]


def task_rows(rows: list[dict]) -> list[list[str]]:
    used = defaultdict(set)
    for row in rows:
        used[row["task"]].add(row["iteration"])
    out = []
    for key, info in TEXT["tasks"].items():
        its = sorted(used.get(key, ()), key=iteration_key)
        span = f"{its[0]}–{its[-1]}" if len(its) > 1 else (its[0] if its else "–")
        out.append([f"<b>{esc(TASKS.get(key, key))}</b>", info["what"], info["budget"], span])
    return out


def build() -> str:
    loop = json.loads((DATA / "loop.json").read_text())
    rows, summary = loop["trials"], loop["summary"]
    iterations = sorted((s for s in summary if ":" not in s and s != "all"), key=iteration_key)
    chart = (
        '<div id="loop-charts">'
        + STYLE
        + ratio_chart(
            [(f"{i} ({TEXT['iterations'][i]['harness']})", summary[i]["ratios"]["seconds"]) for i in iterations]
            + [("all iterations", summary["all"]["ratios"]["seconds"])],
            "Time with the MCP relative to without it, per iteration (geometric mean over pairs, 95% interval)",
            "Paired time ratios per iteration",
            hi=2.6,
        )
        + "</div>"
    )
    head = (HERE / "index_head_summary.html").read_text()
    body = [
        head.replace("{{ABSTRACT}}", TEXT["abstract"]).replace("{{MODIFIED}}", TEXT["modified"]),
        section(
            "setup",
            1,
            "How the study works",
            "".join(f"<p>{p}</p>" for p in TEXT["setup"])
            + table(
                "Table 1. Tasks. Every submission is verified in a fresh process against checks the agent cannot see; "
                "bold task names in Table 2 mark the iteration that introduced the task.",
                ["Task", "What the agent has to do", "Budget", "Iterations"],
                task_rows(rows),
                prose=True,
                label="Tasks",
            ),
        ),
        section(
            "iterations",
            2,
            "Iterations at a glance",
            "".join(f"<p>{p}</p>" for p in TEXT["iterations_intro"])
            + table(
                "Table 2. One row per iteration. Each iteration ran on one harness version (a commit on the Newton branch). "
                "Passes count verified successes with and without the MCP. Ratios are geometric means of MCP/no-MCP over "
                "pairs; below 1 favors the MCP. Cost covers Claude Code (Opus) pairs only.",
                ["Iteration", "What changed", "Tasks", "Pairs", "Passes MCP / no MCP", "Time", "Input tokens", "Opus cost"],
                iteration_rows(rows, summary),
                prose=True,
                label="Iterations",
            )
            + chart
            + table(
                "Table 3. All iterations pooled by task, sorted by the time ratio (geometric mean MCP/no MCP with 95% "
                "interval). The MCP helps where each experiment is cheap but pays setup and inspection overhead.",
                ["Task", "Pairs", "Passes MCP / no MCP", "Time", "MCP faster in", "Input tokens"],
                effect_rows(rows),
                numeric={1, 4},
                label="Effect by task",
            ),
        ),
        section("findings", 3, "What we learned", "<ol class=\"findings\">" + "".join(f"<li>{f}</li>" for f in TEXT["findings"]) + "</ol>"),
        section("newton", 4, "Newton fixes from the study", "".join(f"<p>{p}</p>" for p in TEXT["newton"])),
        section("next", 5, "Current MCP and next steps", "".join(f"<p>{p}</p>" for p in TEXT["next"])),
        section("limits", 6, "Limits", "<ul>" + "".join(f"<li>{p}</li>" for p in TEXT["limits"]) + "</ul>"),
        section("source", 7, "Data, code, and the full log", "".join(f"<p>{p}</p>" for p in TEXT["source"])),
    ]
    footer = (
        '<footer class="report-footer"><span>Newton live simulation · experimental research</span>'
        '<a href="log.html">Full development log</a><a href="study3.html">Visual calibration study (v3)</a>'
        '<a href="study2.html">Three-way study (v2)</a><a href="historical.html">Original study</a><a href="../">All reports</a></footer>'
        "</main></div></body></html>"
    )
    return "".join(body) + footer


if __name__ == "__main__":
    (HERE / "index.html").write_text(build())
    print("wrote index.html")

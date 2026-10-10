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
                ratio(r.get("first_pass_seconds")),
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


V6_ARMS = ("B", "C", "C-mcp", "H", "H-noMCP", "E")


def _median(values: list) -> float | None:
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


def _minutes(value: float | None) -> str:
    return "–" if value is None else f"{value / 60:.1f}"


def _dollars(value: float | None) -> str:
    return "–" if value is None else f"${value:.2f}"


def v6_arm(row: dict) -> str:
    """Arm label with the harness iteration for harness arms (H v6.1, ...)."""
    return f'{row["arm"]} {row["iteration"]}' if row.get("harness") else row["arm"]


def v6_columns(trials: list[dict]) -> list[str]:
    order = {arm: index for index, arm in enumerate(V6_ARMS)}
    return sorted({v6_arm(r) for r in trials}, key=lambda a: (order.get(a.split()[0], 9), a))


def v6_result_rows(trials: list[dict], model: str | None = None) -> tuple[list[str], list[list[str]]]:
    """One row per task, one column per arm (B pooled over Opus and Astra; other arms for ``model`` only):
    passes, median time (and time to the first passing snapshot), median cost."""
    trials = [r for r in trials if r["arm"] == "B" or model is None or r["model"] == model]
    columns = v6_columns(trials)
    order = list(TEXT["v6"]["task_order"])
    tasks = sorted({r["task"] for r in trials}, key=lambda t: order.index(t) if t in order else 99)
    rows = []
    for task in tasks + ["all"]:
        cells = [f"<b>{esc(TASKS.get(task, task))}</b>" if task != "all" else "<b>All tasks</b>"]
        for column in columns:
            group = [r for r in trials if v6_arm(r) == column and (task == "all" or r["task"] == task)]
            if not group:
                cells.append("\u2013")
                continue
            passed = sum(r["success"] for r in group)
            first = _median([r["first_pass_seconds"] for r in group if r["success"]])
            verdict = (("pass" if passed else "fail") if len(group) == 1 else f"{passed}/{len(group)}")
            cells.append(
                f"{verdict} \u00b7 {_minutes(_median([r['seconds'] for r in group]))} min"
                + (f" (first {_minutes(first)})" if first is not None else "")
                + f" \u00b7 {_dollars(_median([r['cost_usd'] for r in group]))}"
            )
        rows.append(cells)
    return ["Task"] + columns, rows


def v6_usage_rows(trials: list[dict]) -> tuple[list[str], list[list[str]]]:
    """Per arm: what the agents used (medians per trial; card counts are totals over the arm's trials)."""
    out = []
    labels = {"haiku": "Haiku", "luna": "Luna"}
    groups = []
    for column in v6_columns(trials):
        members = [r for r in trials if v6_arm(r) == column]
        if column == "B":
            groups.append((column, members))
            continue
        for model in sorted({r["model"] for r in members}):
            groups.append((f"{column} \u00b7 {labels.get(model, model)}", [r for r in members if r["model"] == model]))
    for column, group in groups:
        cards = [r.get("card_usage") or {} for r in group]
        harness = any(r.get("harness") for r in group)
        out.append(
            [
                esc(column),
                str(len(group)),
                f'{_median([r["tool_calls"] for r in group]) or 0:.0f}',
                f'{_median([r["mcp_calls"] for r in group]) or 0:.0f}',
                f'{_median([r["shell_python"] for r in group]) or 0:.0f}',
                f'{_median([r["source_reads"] for r in group]) or 0:.0f}',
                f'{sum(1 for c in cards if c.get("cards_retrieved"))} of {len(group)}' if harness else "\u2013",
                f'{sum(len(c.get("cards_retrieved") or []) for c in cards)}' if harness else "\u2013",
                f'{sum(c.get("auto_attached", 0) for c in cards)}' if harness else "\u2013",
                f'{(_median([r["input_tokens"] for r in group]) or 0) / 1e6:.2f}M',
            ]
        )
    head = ["Arm", "Trials", "Tool calls", "MCP calls", "Python runs", "Source reads", "Trials reading cards", "Cards read", "Cards attached", "Input tokens"]
    return head, out


def v6_section(number: int) -> str:
    text = TEXT["v6"]
    path = DATA.parent / "v6" / "trials.json"
    trials = json.loads(path.read_text()) if path.exists() else []
    body = "".join(f"<p>{p}</p>" for p in text["intro"])
    body += table(
        "Table 4. Arms of the cheaper-model study. Every trial runs alone in the same sandbox with the same budget, "
        "hidden verifier and compile-cache seed as in iterations 0\u201317.",
        ["Arm", "Models", "MCP", "Debug Cards and procedure", "Question"],
        text["arms"],
        prose=True,
        label="v6 arms",
    )
    if trials:
        for number, (model, name) in enumerate((("luna", "GPT-6 Luna"), ("haiku", "Claude Haiku 4.5")), start=5):
            head, rows = v6_result_rows(trials, model)
            body += table(
                f"Table {number}. {name} against the expensive baseline, per task: verdict (or passes/trials) \u00b7 agent "
                "time in minutes (time to the first passing snapshot) \u00b7 cost at list prices. B pools Opus and Astra "
                "(medians); every other column is one trial per task. sdf_grind is held out: no card was mined from it.",
                head,
                rows,
                prose=True,
                label=f"v6 results {name}",
            )
        head, rows = v6_usage_rows(trials)
        body += table(
            "Table 7. What the agents used, per arm: medians per trial for tool calls, MCP calls, fresh Python runs, Newton "
            "source reads and input tokens; card counts are totals over the arm's trials (distinct cards per trial).",
            head,
            rows,
            numeric=set(range(1, 10)),
            label="v6 usage",
        )
    body += "".join(f"<p>{p}</p>" for p in text["log"])
    return section("v6", number, text["title"], body)


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
                "pairs; below 1 favors the MCP. Time to first pass (from iteration 16) is the first workspace snapshot that passes "
                "the verifier, over pairs where both runs passed. Cost covers Claude Code (Opus) pairs only.",
                ["Iteration", "What changed", "Tasks", "Pairs", "Passes MCP / no MCP", "Time", "Time to first pass", "Input tokens", "Opus cost"],
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
        v6_section(4),
        section("newton", 5, "Newton fixes from the study", "".join(f"<p>{p}</p>" for p in TEXT["newton"])),
        section("next", 6, "Current MCP and next steps", "".join(f"<p>{p}</p>" for p in TEXT["next"])),
        section("limits", 7, "Limits", "<ul>" + "".join(f"<li>{p}</li>" for p in TEXT["limits"]) + "</ul>"),
        section("source", 8, "Data, code, and the full log", "".join(f"<p>{p}</p>" for p in TEXT["source"])),
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

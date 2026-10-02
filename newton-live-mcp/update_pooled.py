"""Refresh the pooled numbers that the narrative quotes (pairs, trials, ratios) from data/v4/loop.json."""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / "data/v4/loop.json").read_text())
summary, trials = data["summary"], data["trials"]
n = json.loads((HERE / "narrative_v4.json").read_text())


def r(scope, metric):
    return summary[scope]["ratios"][metric]


pairs = summary["all"]["pairs"]
faster = r("all", "seconds")["mcp_better"]
t = r("all", "seconds")
tok = r("all", "input_tokens")
astra = r("all:astra", "input_tokens")
cost = r("all:opus", "cost_usd")
total, passed = len(trials), sum(bool(x["success"]) for x in trials)
replay_fail = [x for x in trials if x["task"] in ("abc_replay", "abc_bin") and not x["success"]]
fail_mcp = sum(x["condition"] == "mcp" for x in replay_fail)
fail_restart = len(replay_fail) - fail_mcp
ci = lambda v: f"{v['ci95'][0]:.2f}–{v['ci95'][1]:.2f}"  # noqa: E731

rules = [
    (r"harness versions and \d+ pairs", f"harness versions and {pairs} pairs"),
    (r"Across all pairs the MCP was faster in \d+ \(time ratio [0-9.]+, 95% interval [0-9.–]+\)",
     f"Across all pairs the MCP was faster in {faster} (time ratio {t['ratio']:.2f}, 95% interval {ci(t)})"),
    (r"GPT-6 Astra used fewer tokens in \d+ of \d+ \([0-9.]+×\)",
     f"GPT-6 Astra used fewer tokens in {astra['mcp_better']} of {astra['n']} ({astra['ratio']:.2f}×)"),
    (r"faster in \d+ of \d+ \([0-9.]+×\)", f"faster in {faster} of {pairs} ({t['ratio']:.2f}×)"),
    (r"Across \d+ completed pairs", f"Across {pairs} completed pairs"),
    (r"\(geometric-mean ratio [0-9.]+, 95% interval [0-9.–]+; faster in \d+ pairs\) and input tokens \([0-9.]+, [0-9.–]+; fewer in \d+ pairs\)",
     f"(geometric-mean ratio {t['ratio']:.2f}, 95% interval {ci(t)}; faster in {faster} pairs) and input tokens "
     f"({tok['ratio']:.2f}, {ci(tok)}; fewer in {tok['mcp_better']} pairs)"),
    (r"Opus's API cost was [0-9.]+× overall", f"Opus's API cost was {cost['ratio']:.2f}× overall"),
    (r"GPT-6 Astra used fewer tokens in \d+ of its \d+ pairs \([0-9.]+×, [0-9.–]+\)",
     f"GPT-6 Astra used fewer tokens in {astra['mcp_better']} of its {astra['n']} pairs ({astra['ratio']:.2f}×, {ci(astra)})"),
    (r"Of \d+ verified trials, \d+ passed", f"Of {total} verified trials, {passed} passed"),
    (r"\d+ of \d+ verified trials passed", f"{passed} of {total} verified trials passed"),
    (r"the other \w+ are physical replays that failed the unseen episodes \(three with the MCP, two without\)|the other \w+ are physical replays that failed the unseen episodes \(\w+ with the MCP, \w+ without\)",
     f"the other {len(replay_fail)} are physical replays that failed the unseen episodes ({fail_mcp} with the MCP, {fail_restart} without)"),
    (r"The other \w+ were physical replays \(\w+ with the MCP, \w+ without\)",
     f"The other {len(replay_fail)} were physical replays ({fail_mcp} with the MCP, {fail_restart} without)"),
]
changed = 0


def apply(text):
    global changed
    for pattern, new in rules:
        text, k = re.subn(pattern, new, text)
        changed += k
    return text


for key, value in n.items():
    if isinstance(value, str):
        n[key] = apply(value)
    elif isinstance(value, list):
        n[key] = [apply(v) if isinstance(v, str) else v for v in value]
(HERE / "narrative_v4.json").write_text(json.dumps(n, indent=1, ensure_ascii=False) + "\n")
print(f"{changed} replacements; pairs {pairs}, trials {passed}/{total}, replay failures {fail_mcp} MCP / {fail_restart} restart")

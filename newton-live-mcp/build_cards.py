"""The Debug-Card part of the v6 study: card catalog, usage and usefulness, harness versions, per-trial table.

Writes cards.html (the full card library and the evidence) and provides the compact tables that build_summary.py
puts into the summary page. Inputs in data/v6/: cards.json (catalog, card texts), harnesses.json (procedure and
prompt texts), trials.json (export_v6.py), card_usefulness.json (trace classification of every card read or
attachment, merged from research/v6/card_usefulness/*.json by merge_usefulness()).
"""

from __future__ import annotations

import collections
import html
import json
import re
from pathlib import Path

from build_v3 import esc, section, table

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "v6"
USEFULNESS_SOURCES = Path("/home/horde/artifacts/newton-live-mcp-v5/research/v6/card_usefulness")
MODELS = {"haiku": "Haiku", "luna": "Luna", "sol": "Sol", "opus": "Opus", "astra": "Astra"}
CLASSES = ("decisive", "helpful", "unused", "misleading")
TASK_NAMES = {
    "g1_track": "G1 tracking",
    "g1_hard": "G1 hard",
    "grasp_drift": "Grasp drift",
    "cube_toss": "Cube toss",
    "abc_arm": "ABC arm replay",
    "sdf_grind": "SDF grinding",
}


def load(name: str):
    path = DATA / name
    return json.loads(path.read_text()) if path.exists() else []


def merge_usefulness() -> list[dict]:
    """Merge the per-group classification files into data/v6/card_usefulness.json."""
    trials = []
    for path in sorted(USEFULNESS_SOURCES.glob("*.json")):
        group = json.loads(path.read_text())
        for trial in group.get("trials", []):
            trial["group"] = group.get("group", path.stem)
            trials.append(trial)
    (DATA / "card_usefulness.json").write_text(json.dumps(trials, indent=1, ensure_ascii=False) + "\n")
    return trials


def _model(trial: dict) -> str:
    name = trial["trial"].split("/")[-1]
    return name.split("-")[1]


# -- Markdown (the subset the cards use) ----------------------------------------------------------------------------


def _inline(text: str) -> str:
    out = []
    for i, part in enumerate(re.split(r"(`[^`]*`)", text)):
        if i % 2:
            out.append(f"<code>{html.escape(part[1:-1])}</code>")
        else:
            part = html.escape(part)
            part = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", part)
            out.append(part)
    return "".join(out)


def markdown(text: str) -> str:
    """Card markdown without front matter and title: ## headings, nested - lists, | tables |, paragraphs."""
    lines = text.splitlines()
    out, stack, rows = [], [], []

    def close_lists(depth=0):
        while len(stack) > depth:
            out.append("</li></ul>")
            stack.pop()

    def flush_table():
        if rows:
            head, body = rows[0], [r for r in rows[2:]]
            out.append('<div class="table-wrap"><table class="text-table"><thead><tr>')
            out.append("".join(f"<th>{_inline(c)}</th>" for c in head) + "</tr></thead><tbody>")
            out.extend("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in body)
            out.append("</tbody></table></div>")
            rows.clear()

    for line in lines:
        if line.startswith("|"):
            close_lists()
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
            continue
        flush_table()
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("## "):
            close_lists()
            out.append(f"<h4>{_inline(stripped[3:])}</h4>")
        elif stripped.startswith("# "):
            continue
        elif stripped.startswith("- "):
            depth = (len(line) - len(line.lstrip())) // 2 + 1
            if depth > len(stack):
                while depth > len(stack):
                    out.append("<ul><li>")
                    stack.append(depth)
            else:
                close_lists(depth)
                out.append("</li><li>")
            out.append(_inline(stripped[2:]))
        else:
            close_lists()
            out.append(f"<p>{_inline(stripped)}</p>")
    close_lists()
    flush_table()
    return "".join(out)


def card_body(text: str) -> str:
    return markdown(text.split("---", 2)[2] if text.startswith("---") else text)


# -- Aggregates ----------------------------------------------------------------------------------------------------


def card_stats(trials: list[dict], usefulness: list[dict]) -> dict[str, dict]:
    """Per card: harness trials that read it (per model), automatic attachments, and the trace classes."""
    stats = collections.defaultdict(lambda: {"read": collections.Counter(), "attached": 0, "classes": collections.Counter(),
                                             "lost": 0.0, "examples": []})
    if not usefulness:
        for row in trials:
            if not row.get("harness"):
                continue
            usage = row.get("card_usage") or {}
            for card in usage.get("cards_retrieved") or []:
                stats[card]["read"][row["model"]] += 1
            for card in usage.get("cards_attached") or []:
                stats[card]["attached"] += 1
    for trial in usefulness:
        # Reads and attachments come from the transcript classification: the usage logger missed glob reads
        # such as `cat cards/FIT-03*.md`.
        for card in trial.get("cards", []):
            entry = stats[card["id"]]
            if card.get("how") == "auto":
                entry["attached"] += 1
            else:
                entry["read"][_model(trial)] += 1
            entry["classes"][card["class"]] += 1
            entry["lost"] += float(card.get("seconds_lost") or 0) if card["class"] == "misleading" else 0.0
            entry["examples"].append({**card, "trial": trial["trial"], "model": _model(trial), "success": trial.get("success")})
    return stats


def verdict(entry: dict) -> str:
    """One word for a card's observed value."""
    c = entry["classes"]
    seen = sum(c.values())
    if not seen:
        return "not used"
    if c["decisive"] >= 2 or (c["decisive"] and c["decisive"] + c["helpful"] >= seen / 2):
        return "high"
    if c["misleading"] > c["decisive"] + c["helpful"]:
        return "misleading"
    if c["decisive"] + c["helpful"] >= max(1, seen / 3):
        return "some"
    return "low"


def class_cell(entry: dict) -> str:
    c = entry["classes"]
    if not sum(c.values()):
        return "–"
    return " · ".join(f"{c[k]} {k}" for k in CLASSES if c[k])


def read_cell(entry: dict) -> str:
    read = entry["read"]
    if not read:
        return "–"
    return ", ".join(f"{MODELS.get(m, m)} {n}" for m, n in sorted(read.items(), key=lambda kv: -kv[1]))


def catalog_rows(cards: list[dict], stats: dict, link: bool = True) -> list[list[str]]:
    rows = []
    for card in cards:
        entry = stats.get(card["id"], {"read": {}, "attached": 0, "classes": collections.Counter()})
        anchor = f'<a href="cards.html#{card["id"]}">{card["id"]}</a>' if link else card["id"]
        rows.append(
            [
                f'<b style="white-space:nowrap">{anchor}</b>',
                f"<b>{esc(card['title'])}.</b> {esc(card['gist'])}",
                read_cell(entry),
                str(entry["attached"]) if entry["attached"] else "–",
                class_cell(entry),
                f"<b>{verdict(entry)}</b>",
            ]
        )
    return rows


CATALOG_HEAD = ["Card", "What it says", "Read in trials (by model)", "Attached", "Trace classes", "Value"]
CATALOG_CAPTION = (
    "The 36 Debug Cards. Read: harness trials whose agent opened the card (file or card()), per model. Attached: "
    "times the MCP host attached it automatically. Trace classes: every read or attachment classified from the "
    "transcript as decisive (the main diagnosis or fix follows it), helpful (informed a check or ruled out a cause), "
    "unused, or misleading (a detour). Value: high = decisive at least twice or decisive and mostly useful; some = "
    "useful in at least a third of uses; low = mostly unused; misleading = more detours than help; not used."
)


def harness_rows(harnesses: list[dict], trials: list[dict]) -> list[list[str]]:
    summary = {
        "cards-v1": ("v6.1", "35 cards as a skill; index path in the prompt; MCP card helpers and attachments; a "
                     "seven-step procedure (criteria checklist, local check script, cards before source, one change "
                     "at a time, loop monitor, save passing versions, stop with margin)."),
        "cards-v2": ("v6.2", "36 cards (adds FIT-05, revises CON-02); all card titles listed in the prompt with an "
                     "instruction to open the matching cards before the first experiment; an eight-step procedure "
                     "adding a tested metric module, a held-out split, in-process experiments, wide sweeps first, a "
                     "noise floor, and no finishing while a criterion fails."),
        "cards-v3": ("v6.3–v6.5", "the lite harness: the same 36 cards and title list, and four rules instead of the "
                     "procedure (check the criteria as verification will, with a held-out split; prefer the task's "
                     "own evaluation and wide sweeps; open a card when a symptom is unclear; do not finish while a "
                     "criterion fails)."),
    }
    rows = []
    for h in harnesses:
        iterations, what = summary[h["name"]]
        groups = collections.defaultdict(list)
        for row in trials:
            if row.get("harness") == h["name"]:
                groups[row["model"]].append(row)
        results = "; ".join(
            f"{MODELS.get(m, m)} {sum(r['success'] for r in g)}/{len(g)}" for m, g in sorted(groups.items())
        )
        rows.append([f"<b>{h['name']}</b><br>{iterations}", esc(what), f"{h['prompt_chars_added']['mcp']:,}", results])
    return rows


HARNESS_HEAD = ["Harness", "What the agent gets", "Prompt characters added", "Passes (per model)"]


TASK_NEEDS = {
    "g1_track": ("Stiffer, load-based servo gains with velocity feedforward, and balance feedback through the joints "
                 "for faster playback.", "CTL-02, CTL-01"),
    "g1_hard": ("Tracking clips whose reference is not dynamically feasible: make the plan feasible or balance "
                "through a contact-aware controller.", "CTL-03, CTL-04"),
    "grasp_drift": ("The finger pads have a higher geom priority and inherit a soft robot-wide contact default, so "
                    "the pads' material decides the contact; stiffen the deciding geoms.", "CON-01"),
    "cube_toss": ("Contact damping high enough that impacts barely rebound, friction near 0.2, validated on held-out "
                  "tosses.", "CON-02 (v2), FIT-04, FIT-05"),
    "abc_arm": ("Command delay, per-joint gains and partial gravity compensation fitted jointly across logs, checked "
                "on held-out logs.", "FIT-03, FIT-05"),
    "sdf_grind": ("Edit the SDF data that collision samples, in place on both levels, along the swept tool path.",
                  "BLD-05"),
}


def task_rows(trials: list[dict], usefulness: list[dict]) -> list[list[str]]:
    """Per task: the decisive knowledge, the card that states it, and outcomes with and without reading it."""
    rows = []
    for task, (needed, cards) in TASK_NEEDS.items():
        key = cards.split(",")[0].split(" ")[0]
        harness = [t for t in usefulness if t.get("task") == task]
        read = [t for t in harness if any(c["id"] == key and c["how"] != "auto" for c in t.get("cards", []))]
        unread = [t for t in harness if t not in read]
        decisive = sum(1 for t in harness for c in t.get("cards", []) if c["id"] == key and c["class"] == "decisive")
        plain = [r for r in trials if r["task"] == task and not r.get("harness")]
        rows.append(
            [
                f"<b>{TASK_NAMES[task]}</b>",
                esc(needed),
                cards,
                f"{sum(t['success'] for t in read)}/{len(read)}" if read else "–",
                f"{sum(t['success'] for t in unread)}/{len(unread)}" if unread else "–",
                str(decisive),
                f"{sum(r['success'] for r in plain)}/{len(plain)}",
            ]
        )
    return rows


TASK_HEAD = ["Task", "What it needed", "Card(s)", "Passes when the first card was read", "Passes when not read",
             "Times decisive", "Passes in arms without cards (all models)"]


def trial_rows(usefulness: list[dict]) -> list[list[str]]:
    rows = []
    for t in sorted(usefulness, key=lambda t: (t["trial"].split("/")[0], t["task"], t["trial"])):
        cards = ", ".join(
            f'<span class="card-{c["class"]}">{c["id"]}{"*" if c["how"] == "auto" else ""} ({c["class"]})</span>'
            for c in t.get("cards", [])
        ) or "–"
        note = t.get("failure_reason") if not t.get("success") else ""
        missing = t.get("missing_card")
        rows.append(
            [
                esc(t["trial"]),
                "pass" if t.get("success") else "fail",
                cards,
                esc(t.get("needed") or ""),
                esc(" ".join(x for x in (note or "", f"Missing: {missing}" if missing else "") if x)),
            ]
        )
    return rows


# -- Page --------------------------------------------------------------------------------------------------------------


def card_section(card: dict, entry: dict) -> str:
    examples = sorted(entry["examples"], key=lambda e: (CLASSES.index(e["class"]), e["trial"])) if entry else []
    evidence = "".join(
        f'<li><span class="card-{e["class"]}">{e["class"]}</span> · {esc(e["trial"])} ({MODELS.get(e["model"], e["model"])}, '
        f'{"pass" if e["success"] else "fail"}{", attached" if e["how"] == "auto" else ""}): {esc(e.get("evidence") or "")}</li>'
        for e in examples
    )
    meta = (
        f'<p class="muted">{esc(card["family"])} · introduced in {card["introduced"]}'
        + (f" · changed: {esc(card['changed'])}" if card.get("changed") else "")
        + f' · verification confidence: {esc(card["confidence"])}'
        + (f' · {card["signatures"]} attachment signature(s)' if card["signatures"] else " · no attachment signature")
        + "</p>"
    )
    return (
        f'<article class="debug-card" id="{card["id"]}"><h3>{card["id"]}: {esc(card["title"])}</h3>{meta}'
        f'<p><b>In short.</b> {esc(card["gist"])}</p>'
        f'<p><b>Mined from.</b> {esc(card["evidence"])}</p>'
        + (f'<p><b>Verification.</b> {esc(card["verify_changes"])}</p>' if card.get("verify_changes") else "")
        + f'<details><summary>Full card text (as given to agents in cards-v3)</summary>{card_body(card["text"])}</details>'
        + (f"<p><b>Use in the harness trials</b> ({class_cell(entry)}):</p><ul>{evidence}</ul>" if evidence else
           "<p><b>Use in the harness trials:</b> never opened or attached.</p>")
        + "</article>"
    )


STYLE = """<style>
.debug-card{border-top:1px solid var(--rule, #ddd);padding-top:.6rem;margin-top:1.2rem}
.debug-card h3{font-size:1.05rem;margin-bottom:.2rem}
.debug-card details{margin:.4rem 0 .6rem}
.debug-card details summary{cursor:pointer}
.debug-card h4{font-size:.95rem;margin:.6rem 0 .2rem}
.card-decisive{color:#1a7f37;font-weight:600}
.card-helpful{color:#0b6bcb}
.card-unused{color:#6b7280}
.card-misleading{color:#b42318;font-weight:600}
pre.harness-text{white-space:pre-wrap;font-size:.85rem;background:rgba(127,127,127,.12)!important;color:inherit!important;padding:.6rem;border-radius:4px;max-height:28rem;overflow:auto}
</style>"""


def build_page() -> str:
    cards, harnesses, trials = load("cards.json"), load("harnesses.json"), load("trials.json")
    usefulness = load("card_usefulness.json")
    stats = card_stats(trials, usefulness)
    head = (HERE / "index_head_summary.html").read_text()
    head = head.replace("<title>Improving the Newton MCP harness for simulation setup and tuning</title>",
                        "<title>Debug Cards for Newton agents: the cards, how they were made, and how useful each was</title>")
    start = head.index('<section class="report-hero"')
    head = head[:start]
    head = re.sub(r'<nav class="nav-links".*?</nav>',
                  '<nav class="nav-links" aria-label="Report sections"><a href="#making">How they were made</a>'
                  '<a href="#catalog">Catalog</a><a href="#tasks">Tasks</a><a href="#harness">Harness versions</a>'
                  '<a href="#cards">Cards</a><a href="#trials">Trials</a><a href="index.html#v6">Summary</a></nav>',
                  head, flags=re.S)
    hero = (
        '<section class="report-hero" id="overview"><p class="paper-series">Newton live-MCP study · cheaper-model phase '
        '(v6) · Debug Cards</p><h1>Debug Cards for Newton agents</h1>'
        "<p>This page documents the 36 Debug Cards used in the cheaper-model phase of the "
        '<a href="index.html#v6">Newton live-MCP study</a>: how they were distilled from the study\'s transcripts and '
        "checked, how agents received them, what each card says, and how useful each one was in the 78 trials that "
        "had them. Usefulness comes from reading the transcripts: every time an agent opened a card, or the MCP host "
        "attached one, the following actions were classified as decisive, helpful, unused or misleading, with the "
        "evidence quoted below each card.</p></section>"
    )
    making = section(
        "making", 1, "How the cards were made and delivered",
        "<p><strong>Mining.</strong> Six readers, one per task family (G1 control, grasping, identification, physical "
        "replays, scene building from photos, visual tasks), read all 245 transcripts of iterations 0–17 (Opus 5.5 "
        "and GPT-6 Astra, with and without the MCP). They extracted episodes where agents struggled, giving the trial, "
        "line and time, and drafted 110 candidate cards with about 90 workflow lessons. A distiller merged them into "
        "35 cards of at most about 250 words each, in the Text2Sim shape: symptoms, likely causes, checks (short "
        "Newton calls), repairs (directions, not values), and pitfalls. Workflow lessons went into the procedure "
        "instead.</p>"
        "<p><strong>Verification.</strong> A separate verifier checked every claim against the Newton tree the trials "
        "ran on:\n"
        "<ul>"
        "<li>It resolved all 62 API names the cards use.</li>"
        "<li>It reproduced 14 warning and error scenarios in fresh CPU processes and tested every attachment signature "
        "against them and against the transcripts.</li>"
        "<li>It corrected wrong claims in 23 cards. For example, a mass-dependence claim in CON-02 was disproved by a CPU "
        "test, the overflow messages in CON-06 were current only for older builds, and BAT-04 had the wrong call "
        "form.</li>"
        "</ul></p>"
        "<p><strong>Neutrality.</strong> Cards hold general Newton knowledge only: no task values and no verifier "
        "details. A search of the neutral copies for task names, trial ids and study terms found nothing; trial "
        "references appear only in this page's provenance notes, which agents never saw. One card, FIT-05, and a "
        "revision of CON-02 were added after v6.1 from its traces. Five of the six evaluation tasks were among the "
        "mined families; SDF grinding was not, and serves as the held-out task.</p>"
        "<p><strong>Delivery.</strong> The cards reached agents in three ways:\n"
        "<ul>"
        "<li>As an agent skill in the trial workspace (<code>.claude/skills/</code> for Claude Code, "
        "<code>.agents/skills/</code> for Codex), with an index.</li>"
        "<li>From cards-v2 on, as a list of all titles in the prompt.</li>"
        "<li>In MCP trials, through <code>cards(query)</code> and <code>card(id)</code> in live cells. The host also "
        "names a card in a reply, once per session, when the reply contains a message matching one of its "
        "signatures (20 cards have signatures, mostly exact Newton, MuJoCo Warp and Warp warning strings).</li>"
        "</ul>"
        "Arms without the harness never see the cards: the harness directory is masked in their sandbox.</p>",
    )
    catalog = section(
        "catalog", 2, "Catalog: what each card says and how useful it was",
        table(f"Table 1. {CATALOG_CAPTION}", CATALOG_HEAD, catalog_rows(cards, stats, link=False), prose=True,
              label="Card catalog")
        + "<p>Counts cover the 78 harness trials (Haiku 18, Luna 24, Sol 12, Opus 12, Astra 12). A card can be read "
        "in a trial and attached in another. Opus rarely opened a card, so its rows mostly show attachments.</p>",
    )
    tasks = section(
        "tasks", 3, "What each task needed",
        table("Table 2. The knowledge each task turned on, the card that states it, and passes of harness trials that "
              "did or did not read that card (the first card listed). Times decisive: reads classified as decisive. The "
              "last column pools every arm without cards (B, B-mcp, C, C-mcp) for comparison.",
              TASK_HEAD, task_rows(trials, usefulness), prose=True, label="Task needs"),
    )
    versions = "".join(
        f'<h3>{h["name"]}</h3><p>{h["cards"]} cards; adds {h["prompt_chars_added"]["mcp"]:,} characters to the MCP '
        f"prompt. The text below is the block appended to the prompt (workspace paths shortened), followed by the "
        f'skill description.</p><pre class="harness-text">{html.escape(h["prompt_block_mcp"].strip())}</pre>'
        f'<details><summary>SKILL.md</summary><pre class="harness-text">{html.escape(h["skill"])}</pre></details>'
        for h in harnesses
    )
    harness = section(
        "harness", 4, "Harness versions",
        table("Table 3. The three harness versions and their results (all models, all iterations).", HARNESS_HEAD,
              harness_rows(harnesses, trials), prose=True, label="Harness versions")
        + "<p>In MCP trials the host guide also gets one line: <code>cards('query')</code> lists the debugging cards "
        "that best match the query's words; <code>card('id')</code> returns one; replies name cards whose symptoms "
        "match their output, once per card.</p>"
        + versions,
    )
    full = section(
        "cards", 5, "The cards",
        "".join(card_section(card, stats.get(card["id"])) for card in cards),
    )
    per_trial = section(
        "trials", 6, "Every harness trial",
        table("Table 4. Every harness trial: verdict, the cards it opened or received (* = attached by the MCP host) "
              "with their class, the knowledge the task needed in the reader's words, and for failures the reason and "
              "the missing knowledge.", ["Trial", "Verdict", "Cards (class)", "What it needed", "Failure"],
              trial_rows(usefulness), prose=True, label="Harness trials"),
    )
    footer = (
        '<footer class="report-footer"><span>Newton live simulation · experimental research</span>'
        '<a href="index.html#v6">Summary</a><a href="log.html">Full development log</a><a href="../">All reports</a>'
        "</footer></main></div></body></html>"
    )
    return head.replace("</head>", STYLE + "</head>") + hero + "".join(
        [making, catalog, tasks, harness, full, per_trial]
    ) + footer


if __name__ == "__main__":
    if USEFULNESS_SOURCES.exists():
        merge_usefulness()
    (HERE / "cards.html").write_text(build_page())
    print("wrote cards.html")

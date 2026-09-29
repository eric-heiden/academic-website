"""Inline SVG charts for the harness-improvement loop (styles shared with charts_v3)."""

from __future__ import annotations

import html
import math

from charts_v3 import _fmt, _log_ticks, ratio_chart  # noqa: F401

CONDITIONS = {"mcp": "Newton MCP", "restart": "No MCP (edit/restart)"}
TASKS = {
    "cube_toss": "Cube toss",
    "dp_real": "Real pendulum",
    "g1_track": "G1 tracking",
    "grasp_drift": "Grasp drift",
    "sdf_grind": "SDF grinding",
}
MODELS = {"opus": "Opus 5.5", "astra": "GPT-6 Astra"}


def legend() -> str:
    return (
        '<div class="viz-legend viz-root" aria-hidden="true">'
        '<span><i style="background:var(--series-mcp)"></i>Newton MCP</span>'
        '<span><i style="background:var(--series-restart)"></i>No MCP (edit/restart)</span>'
        '<span><i class="hollow" style="color:var(--text-secondary)"></i>Failed verification or timeout</span></div>'
    )


def dumbbell(pairs: list[dict], metric: str, unit: str, title: str, label: str) -> str:
    """One row per pair (iteration · task · model): MCP and no-MCP values on a shared log axis."""
    values = [p[c][metric] for p in pairs for c in CONDITIONS if p.get(c) and p[c].get(metric)]
    lo, hi = min(values) / 1.25, max(values) * 1.25
    width, left, right, row, top = 760, 230, 24, 22, 44
    height = top + row * len(pairs) + 36
    scale = lambda v: left + (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo)) * (width - left - right)  # noqa: E731
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(label)}">']
    parts.append(f'<text class="viz-title" x="0" y="16">{html.escape(title)}</text>')
    for tick in _log_ticks(lo, hi):
        x = scale(tick)
        parts.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{top - 8}" y2="{height - 30}"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - 12}" text-anchor="middle">{_fmt(tick, unit)}</text>')
    previous = None
    for index, pair in enumerate(pairs):
        y = top + index * row
        if previous is not None and previous != pair["iteration"]:
            parts.append(f'<line class="grid" x1="0" x2="{width}" y1="{y - row / 2:.1f}" y2="{y - row / 2:.1f}"/>')
        previous = pair["iteration"]
        replicate = "" if pair["replicate"] == "p0" else f' · {pair["replicate"]}'
        name = f'{pair["iteration"]} · {TASKS[pair["task"]]} · {MODELS[pair["model"]]}{replicate}'
        parts.append(f'<text x="0" y="{y + 4}">{html.escape(name)}</text>')
        xs = {c: scale(pair[c][metric]) for c in CONDITIONS if pair.get(c) and pair[c].get(metric)}
        if len(xs) == 2:
            parts.append(f'<line class="link" x1="{xs["mcp"]:.1f}" x2="{xs["restart"]:.1f}" y1="{y}" y2="{y}"/>')
        for condition, x in xs.items():
            trial = pair[condition]
            failed = not trial["success"]
            status = "passed" if not failed else ("timed out" if trial["timed_out"] else "failed verification")
            tip = f"{_fmt(trial[metric], unit)} {unit} — {CONDITIONS[condition]}, {name}, {status}"
            klass = f"mark-{condition}" + (" fail" if failed else "")
            parts.append(f'<circle class="{klass}" cx="{x:.1f}" cy="{y}" r="5.5" tabindex="0"><title>{html.escape(tip)}</title></circle>')
    parts.append("</svg>")
    return '<div class="viz-root">' + "".join(parts) + "</div>"

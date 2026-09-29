"""Inline SVG charts for the visual-calibration study (light/dark via CSS custom properties)."""

from __future__ import annotations

import html
import math

CONDITIONS = {"mcp": "Newton MCP", "restart": "Edit/restart"}
TASKS = {"arm_offsets": "Arm", "cloth_drape": "Cloth", "push": "Push"}
MODELS = {"opus": "Opus 5.5", "astra": "GPT-6 Astra"}

STYLE = """<style>
.viz-root{--surface-1:#ffffff;--text-primary:#0b0b0b;--text-secondary:#52514e;--grid:#e4e4e0;--series-mcp:#2a78d6;--series-restart:#eb6834;}
:root[data-theme="dark"] .viz-root{--surface-1:#091713;--text-primary:#ffffff;--text-secondary:#c3c2b7;--grid:#27433d;--series-mcp:#3987e5;--series-restart:#d95926;}
.viz-root svg{width:100%;height:auto;font-family:inherit}
.viz-root text{fill:var(--text-secondary);font-size:12px}
.viz-root .viz-title{fill:var(--text-primary);font-size:13px;font-weight:600}
.viz-root .grid{stroke:var(--grid);stroke-width:1}
.viz-root .ref{stroke:var(--text-secondary);stroke-width:1}
.viz-root .link{stroke:var(--grid);stroke-width:2}
.viz-root .mark-mcp{fill:var(--series-mcp);stroke:var(--surface-1);stroke-width:2}
.viz-root .mark-restart{fill:var(--series-restart);stroke:var(--surface-1);stroke-width:2}
.viz-root .fail{fill:var(--surface-1);stroke-width:2.5}
.viz-root .fail.mark-mcp{stroke:var(--series-mcp)}
.viz-root .fail.mark-restart{stroke:var(--series-restart)}
.viz-root .interval{stroke:var(--text-secondary);stroke-width:2;stroke-linecap:round}
.viz-root [tabindex]:focus{outline:2px solid var(--text-primary)}
.viz-legend{display:flex;gap:18px;flex-wrap:wrap;font-size:13px;margin:4px 0 6px}
.viz-legend span{display:inline-flex;align-items:center;gap:6px}
.viz-legend i{display:inline-block;width:10px;height:10px;border-radius:50%}
#results .result-table td:first-child,#followup .result-table td:first-child{white-space:nowrap}
.viz-legend i.hollow{background:transparent!important;border:2px solid currentColor;width:8px;height:8px}
</style>"""


def legend() -> str:
    return (
        '<div class="viz-legend viz-root" aria-hidden="true">'
        '<span><i style="background:var(--series-mcp)"></i>Newton MCP</span>'
        '<span><i style="background:var(--series-restart)"></i>Edit/restart</span>'
        '<span><i class="hollow" style="color:var(--text-secondary)"></i>Failed held-out check or timeout</span></div>'
    )


def _log_ticks(lo: float, hi: float) -> list[float]:
    ticks = []
    for exponent in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1):
        for mantissa in (1, 2, 3, 5):
            value = mantissa * 10**exponent
            if lo <= value <= hi:
                ticks.append(value)
    return ticks


def _fmt(value: float, unit: str) -> str:
    if unit == "tokens":
        return f"{value / 1e6:.1f}M" if value >= 1e6 else f"{value / 1e3:.0f}k"
    return f"{value:,.0f}"


def dumbbell(pairs: list[dict], metric: str, unit: str, title: str, label: str) -> str:
    """One row per registered pair: MCP and restart values joined on a shared log axis."""
    values = [p[c][metric] for p in pairs for c in CONDITIONS if p.get(c) and p[c][metric]]
    lo, hi = min(values) / 1.25, max(values) * 1.25
    width, left, right, row, top = 760, 150, 24, 22, 44
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
        if previous is not None and previous != (pair["task"], pair["model"]):
            parts.append(f'<line class="grid" x1="0" x2="{width}" y1="{y - row / 2:.1f}" y2="{y - row / 2:.1f}"/>')
        previous = (pair["task"], pair["model"])
        parts.append(f'<text x="0" y="{y + 4}">{TASKS[pair["task"]]} · {MODELS[pair["model"]]} · {pair["replicate"]}</text>')
        xs = {c: scale(pair[c][metric]) for c in CONDITIONS if pair.get(c) and pair[c][metric]}
        missing = [CONDITIONS[c] for c in CONDITIONS if pair.get(c) and not pair[c][metric]]
        if missing:
            # Rows with a missing record are Astra rows whose values sit at the right, so the left edge is free.
            parts.append(f'<text x="{left + 6}" y="{y + 4}">{html.escape(", ".join(missing))}: usage not recorded</text>')
        if len(xs) == 2:
            parts.append(f'<line class="link" x1="{xs["mcp"]:.1f}" x2="{xs["restart"]:.1f}" y1="{y}" y2="{y}"/>')
        for condition, x in xs.items():
            trial = pair[condition]
            failed = not trial["success"]
            status = "passed" if not failed else ("timed out" if trial["timed_out"] else "failed held-out check")
            tip = f"{_fmt(trial[metric], unit)} {unit} — {CONDITIONS[condition]}, {TASKS[pair['task']]} {MODELS[pair['model']]} replicate {pair['replicate']}, {status}"
            klass = f"mark-{condition}" + (" fail" if failed else "")
            parts.append(f'<circle class="{klass}" cx="{x:.1f}" cy="{y}" r="5.5" tabindex="0"><title>{html.escape(tip)}</title></circle>')
    parts.append("</svg>")
    return '<div class="viz-root">' + "".join(parts) + "</div>"


def ratio_chart(groups: list[tuple[str, dict]], title: str, label: str) -> str:
    """Geometric-mean MCP/restart ratios with bootstrap intervals on a log axis; 1 = parity."""
    lo, hi = 0.35, 2.0
    width, left, right, row, top = 760, 150, 60, 26, 44
    height = top + row * len(groups) + 36
    scale = lambda v: left + (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo)) * (width - left - right)  # noqa: E731
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(label)}">']
    parts.append(f'<text class="viz-title" x="0" y="16">{html.escape(title)}</text>')
    for tick in (0.4, 0.5, 0.7, 1, 1.4, 2):
        x = scale(tick)
        parts.append(f'<line class="{"ref" if tick == 1 else "grid"}" x1="{x:.1f}" x2="{x:.1f}" y1="{top - 10}" y2="{height - 30}"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - 12}" text-anchor="middle">{tick:g}×</text>')
    parts.append(f'<text x="{scale(1) - 8:.1f}" y="{top - 16}" text-anchor="end">← favors MCP</text>')
    parts.append(f'<text x="{scale(1) + 8:.1f}" y="{top - 16}">favors restart →</text>')
    for index, (name, value) in enumerate(groups):
        y = top + index * row
        parts.append(f'<text x="0" y="{y + 4}">{html.escape(name)}</text>')
        if not value:
            parts.append(f'<text x="{left}" y="{y + 4}">no jointly successful pairs</text>')
            continue
        a, b = (max(lo, min(hi, v)) for v in value["ci95"])
        parts.append(f'<line class="interval" x1="{scale(a):.1f}" x2="{scale(b):.1f}" y1="{y}" y2="{y}"/>')
        tip = f"{value['ratio']:.2f}× (95% interval {value['ci95'][0]:.2f}–{value['ci95'][1]:.2f}), n={value['n']} pairs, MCP cheaper in {value['mcp_better']}"
        parts.append(
            f'<circle class="mark-mcp" cx="{scale(max(lo, min(hi, value["ratio"]))):.1f}" cy="{y}" r="5.5" tabindex="0"><title>{html.escape(name + ": " + tip)}</title></circle>'
        )
        parts.append(f'<text x="{width - right}" y="{y + 4}" text-anchor="end">{value["ratio"]:.2f}×</text>')
    parts.append("</svg>")
    return '<div class="viz-root">' + "".join(parts) + "</div>"

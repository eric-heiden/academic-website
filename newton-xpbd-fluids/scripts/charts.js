/* Minimal SVG chart layer for the fluid benchmark report.
   Marks stay thin, grid is a hairline, every chart has a table twin. */
const NS = "http://www.w3.org/2000/svg";
const el = (n, a = {}) => {
  const e = document.createElementNS(NS, n);
  for (const [k, v] of Object.entries(a)) e.setAttribute(k, v);
  return e;
};
const cssv = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fmt = (v, d = 2) => (v === null || v === undefined ? "—" : v.toFixed(d));

let TIP;
function tip() {
  if (!TIP) {
    TIP = document.createElement("div");
    TIP.className = "tt";
    TIP.setAttribute("role", "status");
    document.body.appendChild(TIP);
  }
  return TIP;
}
function showTip(evt, html) {
  const t = tip();
  t.innerHTML = html;
  t.style.opacity = "1";
  const r = t.getBoundingClientRect();
  let x = evt.clientX + 14, y = evt.clientY - r.height - 12;
  if (x + r.width > window.innerWidth - 8) x = evt.clientX - r.width - 14;
  if (y < 8) y = evt.clientY + 18;
  t.style.left = x + "px";
  t.style.top = y + "px";
}
const hideTip = () => { if (TIP) TIP.style.opacity = "0"; };

function axes(svg, g, { x0, y0, w, h, xTicks, yTicks, xLabel, yLabel }) {
  const grid = cssv("--gridline"), base = cssv("--baseline"), muted = cssv("--muted");
  for (const t of yTicks) {
    g.appendChild(el("line", { x1: x0, y1: t.p, x2: x0 + w, y2: t.p, stroke: grid, "stroke-width": 1 }));
    const tx = el("text", { x: x0 - 10, y: t.p + 4, "text-anchor": "end", fill: muted, class: "tick" });
    tx.textContent = t.label;
    g.appendChild(tx);
  }
  for (const t of xTicks) {
    const tx = el("text", { x: t.p, y: y0 + 22, "text-anchor": "middle", fill: muted, class: "tick" });
    tx.textContent = t.label;
    g.appendChild(tx);
  }
  g.appendChild(el("line", { x1: x0, y1: y0, x2: x0 + w, y2: y0, stroke: base, "stroke-width": 1 }));
  if (xLabel) {
    const t = el("text", { x: x0 + w / 2, y: y0 + 46, "text-anchor": "middle", fill: muted, class: "axlab" });
    t.textContent = xLabel; g.appendChild(t);
  }
  if (yLabel) {
    const t = el("text", { x: 14, y: y0 - h / 2, "text-anchor": "middle", fill: muted, class: "axlab",
      transform: `rotate(-90 14 ${y0 - h / 2})` });
    t.textContent = yLabel; g.appendChild(t);
  }
}

function niceTicks(lo, hi, n = 5) {
  const span = hi - lo || 1;
  const step0 = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(v);
  return out;
}

/* ---------------- line chart (linear or log x, linear or log y) ---------------- */
function lineChart(host, cfg) {
  const W = host.clientWidth || 720, H = cfg.height || 330;
  const m = { t: 18, r: cfg.padRight || 96, b: 56, l: 66 };
  const w = W - m.l - m.r, h = H - m.t - m.b;
  const x0 = m.l, y0 = m.t + h;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", height: H, role: "img",
    "aria-label": cfg.ariaLabel || "line chart" });
  const g = el("g");
  svg.appendChild(g);

  const allX = cfg.series.flatMap((s) => s.points.map((p) => p.x));
  const allY = cfg.series.flatMap((s) => s.points.map((p) => p.y));
  const xLog = !!cfg.xLog, yLog = !!cfg.yLog;
  const xlo = Math.min(...allX), xhi = Math.max(...allX);
  const ylo = yLog ? Math.min(...allY) : 0, yhi = Math.max(...allY);
  const fx = (v) => x0 + (xLog
    ? (Math.log10(v) - Math.log10(xlo)) / (Math.log10(xhi) - Math.log10(xlo))
    : (v - xlo) / (xhi - xlo || 1)) * w;
  const pad = yLog ? 1 : 1.08;
  const fy = (v) => y0 - (yLog
    ? (Math.log10(v) - Math.log10(ylo * 0.85)) / (Math.log10(yhi * 1.2) - Math.log10(ylo * 0.85))
    : v / (yhi * pad || 1)) * h;

  const xTicks = (cfg.xTicks || (xLog ? allX.filter((v, i, a) => a.indexOf(v) === i) : niceTicks(xlo, xhi)))
    .map((v) => ({ p: fx(v), label: cfg.xFmt ? cfg.xFmt(v) : String(v) }));
  const yTickVals = yLog
    ? [1, 2, 5, 10, 20, 50, 100, 200].filter((v) => v >= ylo * 0.85 && v <= yhi * 1.2)
    : niceTicks(0, yhi * pad, 5);
  const yTicks = yTickVals.map((v) => ({ p: fy(v), label: cfg.yFmt ? cfg.yFmt(v) : String(v) }));
  axes(svg, g, { x0, y0, w, h, xTicks, yTicks, xLabel: cfg.xLabel, yLabel: cfg.yLabel });

  (cfg.rules || []).forEach((r) => {
    g.appendChild(el("line", { x1: x0, y1: fy(r.y), x2: x0 + w, y2: fy(r.y),
      stroke: cssv("--baseline"), "stroke-width": 1 }));
    const t = el("text", { x: x0 + 5, y: fy(r.y) - 7, "text-anchor": "start", fill: cssv("--muted"), class: "tick" });
    t.textContent = r.label; g.appendChild(t);
  });

  // end labels are placed after all series so they can be de-collided
  const endLabels = [];
  cfg.series.forEach((s) => {
    const pts = s.points.slice().sort((a, b) => a.x - b.x);
    const d = pts.map((p, i) => `${i ? "L" : "M"}${fx(p.x).toFixed(1)},${fy(p.y).toFixed(1)}`).join(" ");
    g.appendChild(el("path", { d, fill: "none", stroke: s.color, "stroke-width": 2,
      "stroke-linejoin": "round", "stroke-linecap": "round" }));
    pts.forEach((p) => {
      g.appendChild(el("circle", { cx: fx(p.x), cy: fy(p.y), r: 5.5, fill: cssv("--surface-1") }));
      g.appendChild(el("circle", { cx: fx(p.x), cy: fy(p.y), r: 4, fill: s.color }));
      const hit = el("circle", { cx: fx(p.x), cy: fy(p.y), r: 14, fill: "transparent", tabindex: "0",
        role: "img", "aria-label": `${s.name}, ${cfg.xFmt ? cfg.xFmt(p.x) : p.x}, ${fmt(p.y)} ${cfg.unit || ""}` });
      const html = () => `<b>${s.name}</b><br>${cfg.xLabel || "x"}: ${cfg.xFmt ? cfg.xFmt(p.x) : p.x}<br>` +
        `${fmt(p.y)} ${cfg.unit || ""}${p.note ? "<br><span class='dim'>" + p.note + "</span>" : ""}`;
      hit.addEventListener("mousemove", (e) => showTip(e, html()));
      hit.addEventListener("mouseleave", hideTip);
      hit.addEventListener("focus", (e) => showTip({ clientX: hit.getBoundingClientRect().x,
        clientY: hit.getBoundingClientRect().y }, html()));
      hit.addEventListener("blur", hideTip);
      g.appendChild(hit);
    });
    const last = pts[pts.length - 1];
    endLabels.push({ x: fx(last.x) + 10, y: fy(last.y) + 4, text: s.short || s.name });
  });

  // nudge apart any end labels that would overlap (converging series)
  endLabels.sort((a, b) => a.y - b.y);
  for (let i = 1; i < endLabels.length; i++) {
    const gap = endLabels[i].y - endLabels[i - 1].y;
    if (gap < 15) endLabels[i].y = endLabels[i - 1].y + 15;
  }
  endLabels.forEach((L) => {
    const lab = el("text", { x: L.x, y: L.y, fill: cssv("--text-secondary"), class: "dlab" });
    lab.textContent = L.text;
    g.appendChild(lab);
  });
  host.appendChild(svg);
}

/* ---------------- horizontal bar chart ---------------- */
function barChart(host, cfg) {
  const rows = cfg.rows;
  const W = host.clientWidth || 720;
  const rowH = cfg.rowH || 30, labelW = cfg.labelW || 190;
  const m = { t: 8, r: 74, b: 40, l: labelW };
  const h = rows.length * rowH, H = h + m.t + m.b;
  const w = W - m.l - m.r;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", height: H, role: "img",
    "aria-label": cfg.ariaLabel || "bar chart" });
  const g = el("g"); svg.appendChild(g);
  const hi = Math.max(...rows.map((r) => r.value));
  const fx = (v) => (v / (hi * 1.02)) * w;
  const y0 = m.t + h;
  const ticks = niceTicks(0, hi * 1.02, 5).map((v) => ({ p: m.l + fx(v), label: cfg.xFmt ? cfg.xFmt(v) : String(v) }));
  for (const t of ticks) {
    g.appendChild(el("line", { x1: t.p, y1: m.t, x2: t.p, y2: y0, stroke: cssv("--gridline"), "stroke-width": 1 }));
    const tx = el("text", { x: t.p, y: y0 + 20, "text-anchor": "middle", fill: cssv("--muted"), class: "tick" });
    tx.textContent = t.label; g.appendChild(tx);
  }
  g.appendChild(el("line", { x1: m.l, y1: m.t, x2: m.l, y2: y0, stroke: cssv("--baseline"), "stroke-width": 1 }));
  if (cfg.xLabel) {
    const t = el("text", { x: m.l + w / 2, y: y0 + 38, "text-anchor": "middle", fill: cssv("--muted"), class: "axlab" });
    t.textContent = cfg.xLabel; g.appendChild(t);
  }
  rows.forEach((r, i) => {
    const bh = rowH - 12, yy = m.t + i * rowH + 6;
    const bw = Math.max(fx(r.value), 2);
    const path = el("path", {
      d: `M${m.l},${yy} H${m.l + bw - 4} a4,4 0 0 1 4,4 V${yy + bh - 4} a4,4 0 0 1 -4,4 H${m.l} Z`,
      fill: r.color || cssv("--series-1"),
    });
    g.appendChild(path);
    const lab = el("text", { x: m.l - 12, y: yy + bh / 2 + 4, "text-anchor": "end",
      fill: r.emphasis ? cssv("--text-primary") : cssv("--text-secondary"), class: r.emphasis ? "rlab em" : "rlab" });
    lab.textContent = r.label; g.appendChild(lab);
    const val = el("text", { x: m.l + bw + 9, y: yy + bh / 2 + 4, fill: cssv("--text-secondary"), class: "dlab" });
    val.textContent = cfg.vFmt ? cfg.vFmt(r.value) : fmt(r.value); g.appendChild(val);
    const hit = el("rect", { x: m.l, y: yy - 5, width: Math.max(bw, 8) + 60, height: bh + 10,
      fill: "transparent", tabindex: "0", role: "img", "aria-label": `${r.label}: ${fmt(r.value)} ${cfg.unit || ""}` });
    const html = `<b>${r.label}</b><br>${fmt(r.value)} ${cfg.unit || ""}` + (r.note ? `<br><span class='dim'>${r.note}</span>` : "");
    hit.addEventListener("mousemove", (e) => showTip(e, html));
    hit.addEventListener("mouseleave", hideTip);
    g.appendChild(hit);
  });
  host.appendChild(svg);
}

/* ---------------- stacked horizontal bars ---------------- */
function stackChart(host, cfg) {
  const rows = cfg.rows, keys = cfg.keys;
  const W = host.clientWidth || 720;
  const rowH = cfg.rowH || 52, labelW = cfg.labelW || 210;
  const m = { t: 8, r: 76, b: 42, l: labelW };
  const h = rows.length * rowH, H = h + m.t + m.b;
  const w = W - m.l - m.r;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", height: H, role: "img",
    "aria-label": cfg.ariaLabel || "stacked bar chart" });
  const g = el("g"); svg.appendChild(g);
  const totals = rows.map((r) => keys.reduce((a, k) => a + (r.values[k.key] || 0), 0));
  const hi = Math.max(...totals);
  const fx = (v) => (v / (hi * 1.02)) * w;
  const y0 = m.t + h;
  for (const v of niceTicks(0, hi * 1.02, 5)) {
    const p = m.l + fx(v);
    g.appendChild(el("line", { x1: p, y1: m.t, x2: p, y2: y0, stroke: cssv("--gridline"), "stroke-width": 1 }));
    const tx = el("text", { x: p, y: y0 + 20, "text-anchor": "middle", fill: cssv("--muted"), class: "tick" });
    tx.textContent = String(v); g.appendChild(tx);
  }
  g.appendChild(el("line", { x1: m.l, y1: m.t, x2: m.l, y2: y0, stroke: cssv("--baseline"), "stroke-width": 1 }));
  if (cfg.xLabel) {
    const t = el("text", { x: m.l + w / 2, y: y0 + 38, "text-anchor": "middle", fill: cssv("--muted"), class: "axlab" });
    t.textContent = cfg.xLabel; g.appendChild(t);
  }
  rows.forEach((r, i) => {
    const bh = rowH - 24, yy = m.t + i * rowH + 10;
    let acc = 0;
    keys.forEach((k) => {
      const v = r.values[k.key] || 0;
      if (v <= 0) return;
      const xs = m.l + fx(acc), bw = fx(v);
      acc += v;
      // 2px surface gap between segments, never a border
      const seg = el("rect", { x: xs, y: yy, width: Math.max(bw - 2, 1), height: bh, fill: k.color, rx: 2 });
      g.appendChild(seg);
      const hit = el("rect", { x: xs, y: yy - 6, width: Math.max(bw, 6), height: bh + 12, fill: "transparent",
        tabindex: "0", role: "img", "aria-label": `${r.label}, ${k.name}: ${fmt(v)} ms` });
      const html = `<b>${k.name}</b><br>${r.label}<br>${fmt(v)} ms/frame · ${((v / totals[i]) * 100).toFixed(1)}%`;
      hit.addEventListener("mousemove", (e) => showTip(e, html));
      hit.addEventListener("mouseleave", hideTip);
      g.appendChild(hit);
    });
    const lab = el("text", { x: m.l - 12, y: yy + bh / 2 + 4, "text-anchor": "end",
      fill: cssv("--text-primary"), class: "rlab em" });
    lab.textContent = r.label; g.appendChild(lab);
    if (r.sub) {
      const s = el("text", { x: m.l - 12, y: yy + bh / 2 + 19, "text-anchor": "end", fill: cssv("--muted"), class: "tick" });
      s.textContent = r.sub; g.appendChild(s);
    }
    const val = el("text", { x: m.l + fx(totals[i]) + 9, y: yy + bh / 2 + 4, fill: cssv("--text-secondary"), class: "dlab" });
    val.textContent = fmt(totals[i], 1); g.appendChild(val);
  });
  host.appendChild(svg);
}

function legend(host, items) {
  const d = document.createElement("div");
  d.className = "legend";
  items.forEach((i) => {
    const s = document.createElement("span");
    s.innerHTML = `<i style="background:${i.color}"></i>${i.name}`;
    d.appendChild(s);
  });
  host.appendChild(d);
}

function table(host, cols, rows) {
  const wrap = document.createElement("div");
  wrap.className = "tablewrap";
  const t = document.createElement("table");
  const th = document.createElement("thead");
  th.innerHTML = "<tr>" + cols.map((c) => `<th${c.num ? ' class="num"' : ""}>${c.name}</th>`).join("") + "</tr>";
  t.appendChild(th);
  const tb = document.createElement("tbody");
  rows.forEach((r) => {
    const tr = document.createElement("tr");
    tr.innerHTML = cols.map((c) => `<td${c.num ? ' class="num"' : ""}>${r[c.key] ?? "—"}</td>`).join("");
    tb.appendChild(tr);
  });
  t.appendChild(tb);
  wrap.appendChild(t);
  host.appendChild(wrap);
}

function figure(id, { title, note, build, tableCols, tableRows, legendItems }) {
  const host = document.getElementById(id);
  if (!host) return;
  const fig = document.createElement("figure");
  fig.className = "fig";
  const cap = document.createElement("figcaption");
  cap.innerHTML = `<span class="figtitle">${title}</span>${note ? `<span class="fignote">${note}</span>` : ""}`;
  fig.appendChild(cap);
  if (legendItems) legend(fig, legendItems);
  const plot = document.createElement("div");
  plot.className = "plot";
  fig.appendChild(plot);
  if (tableCols) {
    const det = document.createElement("details");
    det.className = "tv";
    det.innerHTML = "<summary>Table view</summary>";
    fig.appendChild(det);
    table(det, tableCols, tableRows);
  }
  host.appendChild(fig);
  build(plot);
}

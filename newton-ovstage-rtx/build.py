# /// script
# requires-python = ">=3.11"
# dependencies = ["Markdown==3.10.3", "beautifulsoup4==4.13.5", "Pygments==2.19.2"]
# ///
"""Build the Aperture report: uv run newton-ovstage-rtx/build.py."""

import csv
import html
from pathlib import Path

import markdown
from bs4 import BeautifulSoup
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

ROOT = Path(__file__).resolve().parent
TITLE = "USD scene preservation for Newton RTX"
ABSTRACT = (
    "Current Newton and ovnewton workflows, duplicated physics import rules, and a "
    "proposed shared importer with source-scene RTX rendering; appearance comparisons "
    "and measured publication costs."
)
SECTION_LABELS = {
    "motivation": "Motivation and components",
    "current": "Current ovnewton workflow",
    "architecture": "Scene and viewer ownership",
    "ownership": "Shared import and future usage",
    "upgrade": "OVRTX upgrade",
    "evidence": "Visual comparison",
    "performance": "Performance and qualification",
}
SECTIONS = list(SECTION_LABELS)
SCENES = [
    ("anymal_1", "ANYmal / terrain", "ANYmal on an Isaac Lab heightfield",
     "Subdivision and surface appearance differ after visual reconstruction."),
    ("anymal_marble_1", "ANYmal / marble", "ANYmal with the task’s marble MDL material",
     "The preserved source retains the material module and texture dependencies."),
    ("franka_1", "Franka", "Franka with Isaac checker ground",
     "The robot appearance and full ground material remain in the source scene."),
    ("cartpole_1", "Cartpole", "Cartpole with Isaac checker ground",
     "A second articulated asset tests the same scene-preservation path."),
    ("anymal_64", "64 ANYmals", "64 referenced ANYmals",
     "Both images show the initial pose; only part of the 64-robot scene is visible."),
]


def diagram(source, caption, fallback):
    return (
        '<figure class="diagram-panel"><div class="mermaid-shell" '
        f'data-mermaid-source="{html.escape(source, quote=True)}" '
        f'role="img" aria-label="{html.escape(fallback, quote=True)}">'
        f'{html.escape(fallback)}</div><figcaption class="figure-caption">{caption}</figcaption></figure>'
    )


def comparisons():
    parts = ['<div class="scene-control"><label for="scene-select">Comparison scene</label>',
             '<select id="scene-select" aria-controls="comparison-panels">']
    for scene, label, _, _ in SCENES:
        parts.append(f'<option value="{scene}">{label}</option>')
    parts.append('</select></div><div id="comparison-panels">')
    for i, (scene, _, title, observation) in enumerate(SCENES):
        note = ""
        if scene in {"anymal_1", "franka_1", "anymal_64"}:
            note = " The source side uses the Newton prototype’s external-scene API."
        parts.append(f'''<figure class="comparison" id="compare-{scene}" {"hidden" if i else ""}>
<h3>{title}</h3>
<div class="compare-actions"><button class="button comparison-toggle" type="button" aria-pressed="false">Overlay comparison</button><span>OVRTX 0.5 · 640 × 480 pixels</span></div>
<div class="compare-images">
<div class="before"><span class="image-label">Reconstructed Newton visuals</span><img src="assets/{scene}-reconstructed.png" width="640" height="480" loading="lazy" alt="{title}: reconstructed from Newton’s portable visuals"></div>
<div class="after"><span class="image-label">Preserved source in ovstage</span><img src="assets/{scene}-source.png" width="640" height="480" loading="lazy" alt="{title}: original scene rendered through ovstage"></div>
</div>
<label class="slider-control" hidden>Reveal source image <input type="range" min="0" max="100" value="50" aria-label="Reveal source image for {title}"></label>
<figcaption class="figure-caption"><span class="caption-label">Figure {i+4}.</span> {title}. {observation}{note}</figcaption>
</figure>''')
    return "\n".join(parts) + "</div>"


def measurements():
    rows = list(csv.DictReader((ROOT / "assets/measurements.csv").open(newline="", encoding="utf-8")))
    output = ['<table class="measurements"><thead><tr><th>Scene / input / stage</th><th>Workload</th><th>Populate (s)</th><th>Frame median (ms)</th><th>Frame p95 (ms)</th><th>Publish median (ms)</th><th>CPU (MiB)</th><th>GPU (MiB)</th></tr></thead><tbody>']
    for row in rows:
        motion = row["motion"].lower() == "true"
        work = f'Moving / {row["transport"].upper()}' if motion else "Static"
        values = [f'{html.escape(row["scene"])}<small>{html.escape(row["mode"])} · {html.escape(row["domain"])} · {row["ovstage"]}</small>', work]
        for key, digits in (("population_s", 3), ("median_ms", 2), ("p95_ms", 2), ("publish_ms", 2), ("rss_mib", 0), ("gpu_mib", 0)):
            values.append(f'{float(row[key]):.{digits}f}' if row[key] else "—")
        output.append("<tr>" + "".join(f"<td>{v}</td>" for v in values) + "</tr>")
    output.append('</tbody></table><p class="caption">Static rows reuse a committed update. Moving rows write 1,088 transforms each frame; frame time includes publication. A static reconstructed scene is not a moving-scene baseline. Population excludes renderer creation and Newton import. CPU is resident process memory; GPU is a dedicated-allocation snapshot. MiB denotes 2²⁰ bytes. <a href="assets/measurements.csv" download>Download measurement data (CSV)</a>.</p>')
    return "".join(output)


def main():
    source = (ROOT / "report.md").read_text(encoding="utf-8")
    source = source.replace("<!-- CURRENT_FLOW -->", diagram(
        'flowchart LR\nUSD[Composed USD] --> PXR[Newton OpenUSD importer]\nPXR -->|Physics policy A| MODEL[Newton Model]\nUSD --> POP[ovpopulation]\nPOP --> STAGE[ovstage]\nSTAGE --> OV[ovnewton importer]\nOV -->|Physics policy B| MODEL',
        '<span class="caption-label">Figure 1.</span> Current alternative import routes. USD composition is handled by OpenUSD; Newton-specific physics interpretation exists in both importers. The duplication is maintained code, not necessarily two imports in one application.',
        "Two alternative routes build a Newton model: Newton's OpenUSD importer applies physics policy A, while ovpopulation loads ovstage and ovnewton applies a separate physics policy B."
    ))
    source = source.replace("<!-- RUNTIME_FLOW -->", diagram(
        'flowchart LR\nUSD[Composed USD] -->|ovpopulation| STAGE[ovstage]\nSTAGE --> RTX[OVRTX]\nRTX --> VIEW[ViewerRTX]\nSTATE[Newton state] --> POSE[Newton pose transport]\nPOSE --> STAGE',
        '<span class="caption-label">Figure 2.</span> Proposed runtime ownership. Appearance stays in ovstage; Newton writes poses. ovnewton would reuse the optional pose transport.',
        "Composed USD is loaded by ovpopulation into ovstage. Newton state reaches the same stage through pose transport. OVRTX renders that stage and ViewerRTX displays its outputs."
    ))
    source = source.replace("<!-- IMPORT_FLOW -->", diagram(
        'flowchart LR\nPXR[OpenUSD reader in Newton] --> IMPORT[Shared Newton physics importer]\nSTAGE[ovstage reader in ovnewton] --> IMPORT\nIMPORT --> MODEL[Newton Model]',
        '<span class="caption-label">Figure 3.</span> Proposed import ownership. Two source readers supply facts to one Newton implementation of physics interpretation. This shared importer is not implemented by the rendering prototype.',
        "The OpenUSD reader in Newton and the ovstage reader in ovnewton both feed one shared Newton physics importer, which builds the Newton model."
    ))
    source = source.replace("<!-- COMPARISONS -->", comparisons())
    source = source.replace("<!-- NEW_MEASUREMENTS -->", measurements())
    content = BeautifulSoup(markdown.markdown(source, extensions=["tables", "fenced_code", "md_in_html"]), "html.parser")
    assert content.h1.get_text() == TITLE
    content.h1.decompose()

    table_captions = [
        "Component responsibilities and the proposed integration boundary.",
        "How the inspected ovnewton implementation reads and interprets ovstage.",
        "Scene ownership in the two ViewerRTX modes.",
        "Proposed workflows preserve application entry points while sharing physics policy.",
        "Implementation order, repository ownership and acceptance criteria.",
        "Required changes for the OVRTX 0.5 migration.",
        "Measured costs and their implications for workload qualification.",
        "All 13 OVRTX 0.5 research runs; p95 is the 95th percentile of frame time.",
    ]
    assert len(content.find_all("table")) == len(table_captions)
    for i, table in enumerate(content.find_all("table")):
        figure = content.new_tag("figure", attrs={"class": "table-figure"})
        table.wrap(figure)
        wrap = content.new_tag("div", attrs={"class": "table-wrap", "tabindex": "0", "role": "region", "aria-label": f"Table {i+1}"})
        table.wrap(wrap)
        if len(table.select("thead th")) > 2:
            table["class"] = table.get("class", []) + ["table-wide"]
        caption = BeautifulSoup(f'<figcaption class="figure-caption"><span class="caption-label">Table {i+1}.</span> {table_captions[i]}</figcaption>', "html.parser")
        figure.append(caption)

    formatter = HtmlFormatter(nowrap=True, style="github-dark")
    for i, pre in enumerate(content.find_all("pre"), 1):
        code = pre.code
        original = code.get_text()
        code.clear()
        code["class"] = ["code-block", "language-python"]
        code["id"] = f"listing-{i}"
        # Parse inside <pre> so whitespace-only tokens keep their indentation
        # and blank lines instead of being normalized as ordinary HTML text.
        highlighted = BeautifulSoup(
            "<pre>" + highlight(original, PythonLexer(), formatter) + "</pre>",
            "html.parser",
        )
        code.extend(list(highlighted.pre.children))
        assert code.get_text() == original, f"Listing {i} changed during highlighting"
        shell = content.new_tag("div", attrs={"class": "code-shell"})
        pre.wrap(shell)
        header = BeautifulSoup(f'<div class="code-header"><span>Listing {i} · Python</span><button class="copy-button" type="button" data-copy-target="listing-{i}" aria-label="Copy Python listing {i}">Copy code</button></div>', "html.parser")
        shell.insert(0, header)
    (ROOT / "syntax.css").write_text(formatter.get_style_defs(".code-block"), encoding="utf-8")

    template = BeautifulSoup((ROOT.parent / "template-gallery/report.html").read_text(encoding="utf-8"), "html.parser")
    template.title.string = TITLE
    template.find("meta", attrs={"name": "description"})["content"] = ABSTRACT
    for link in template.head.find_all("link"):
        link.decompose()
    for href, relation in (("../favicon.svg", "icon"), ("aperture.css", "stylesheet"), ("report.css", "stylesheet"), ("syntax.css", "stylesheet")):
        template.head.append(template.new_tag("link", href=href, rel=relation))
    template.head.script.string = '''try {
  document.documentElement.dataset.theme = localStorage.getItem("report-theme") === "dark" ? "dark" : "light";
} catch (_) { document.documentElement.dataset.theme = "light"; }'''
    canonical = template.new_tag("link", rel="canonical", href="https://reports.eric-heiden.com/newton-ovstage-rtx/")
    template.head.append(canonical)
    template.body.clear()
    nav = BeautifulSoup('''<a class="skip-link" href="#main-content">Skip to report</a>
<header class="site-nav"><div class="nav-inner">
<a class="brand" href="../" aria-label="All research reports"><span class="brand-mark" aria-hidden="true">N</span><span>Newton / USD scene preservation</span></a>
<nav class="nav-links" aria-label="Report sections"><a href="#motivation" data-section-link>Motivation</a><a href="#current" data-section-link>Current use</a><a href="#ownership" data-section-link>Design</a><a href="#upgrade" data-section-link>Upgrade</a><a href="#evidence" data-section-link>Evidence</a></nav>
<div class="nav-actions"><button class="theme-toggle" id="theme-toggle" type="button" aria-pressed="false" aria-label="Switch to dark theme"><span class="theme-icon" aria-hidden="true">☀</span><span class="theme-label">Bright</span></button></div>
</div></header>''', "html.parser")
    template.body.append(nav)
    shell = template.new_tag("div", attrs={"class": "report-shell"})
    article = template.new_tag("main", attrs={"class": "report", "id": "main-content"})
    shell.append(article)
    template.body.append(shell)
    hero = template.new_tag("section", attrs={"class": "report-hero", "id": "overview"})
    hero.append(BeautifulSoup(f'''<div class="report-dates" aria-label="Report dates"><span>Created <time datetime="2026-09-10">2026-09-10</time></span><span>Modified <time datetime="2026-09-11">2026-09-11</time></span></div>
<p class="paper-series">Technical report</p><h1>{TITLE}</h1>
<dl class="paper-meta"><div><dt>Software</dt><dd>Newton 1.7 development · OVRTX 0.5 · ovstage 0.2</dd></div><div><dt>Evidence</dt><dd>13 research runs · RTX 4090 / Windows</dd></div></dl>''', "html.parser"))
    abstract = template.new_tag("div", attrs={"class": "abstract-block"})
    abstract.append(BeautifulSoup("<h2>Abstract</h2>", "html.parser"))
    abstract.append(content.p.extract())
    hero.append(abstract)
    toc = BeautifulSoup('<details class="mobile-toc"><summary>Contents</summary><nav aria-label="All report sections"></nav></details>', "html.parser")
    for identifier, label in SECTION_LABELS.items():
        toc.nav.append(BeautifulSoup(f'<a href="#{identifier}" data-section-link>{label}</a>', "html.parser"))
    hero.append(toc)
    article.append(hero)
    current = hero
    count = 0
    for node in list(content.children):
        if getattr(node, "name", None) == "h2":
            current = template.new_tag("section", attrs={"class": "report-section", "id": SECTIONS[count]})
            article.append(current)
            title = node.get_text().split(". ", 1)[1]
            count += 1
            heading = BeautifulSoup(f'<div class="section-heading"><div class="section-title-wrap"><h2><span class="section-number">{count}.</span> {title}</h2></div></div>', "html.parser")
            current.append(heading)
            node.decompose()
        else:
            current.append(node.extract())
    assert count == len(SECTIONS)
    article.append(BeautifulSoup('<footer class="report-footer"><span>Measured scene comparisons and proposed integration design.</span><a href="../">All research reports</a></footer>', "html.parser"))
    template.body.append(template.new_tag("script", src="report.js", defer=True))
    template.body.append(template.new_tag("script", src="mermaid.js", type="module"))
    (ROOT / "index.html").write_text(str(template), encoding="utf-8")
    print(f"Built {ROOT / 'index.html'}")


if __name__ == "__main__":
    main()

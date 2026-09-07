# Research report template gallery

Open `index.html` through a local web server. The gallery offers three visual directions backed by one semantic report page:

- **Prism** — neutral sans-serif typesetting with blue/violet figure accents.
- **Signal** — warm, serif-led research publishing.
- **Aperture** — monospaced display type with a teal figure palette.

The report uses a compact single-column article layout with responsive navigation, persistent bright/dark themes, an interactive Plotly figure, KaTeX equations, local video evidence, a Mermaid architecture diagram, formatted source code with copy behavior, a tolerance slider, and a responsive result table. Prism is the default direction, and bright mode uses a white page background.

## Files

- `index.html` — comparison gallery.
- `report.html` — shared report markup, selected with `?template=prism|signal|aperture`.
- `shared.css` — common layout plus the three design languages and both themes.
- `report.js` — theme, template, navigation, copy, plot, legend, and slider interactions.
- `mermaid.js` — theme-aware diagram rendering.
- `math.js` — deferred LaTeX rendering for inline and display equations.
- `assets/` — compact videos and posters reused from the determinism report work.

All report data is representative prototype content. Replace the chart series, table rows, prose, code, and Mermaid source with generated report data when adopting a direction.

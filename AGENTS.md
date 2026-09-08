# Research report repository instructions

These instructions apply to the entire repository.

## Canonical template

- Start every new report from the [Aperture research report template](template-gallery/report.html?template=aperture).
- The complete reusable implementation, including styles, scripts, media examples, and a layout index, is in [`template-gallery/`](template-gallery/).
- Copy the files needed by the new report into its own report folder, then replace the representative content and remove unused examples. Preserve the template's compact reading width, date treatment, navigation, responsive behavior, bright and dark modes, and accessible figure integration.
- Treat the template as a structural and visual reference rather than a content outline. Choose sections that fit the investigation and its evidence.

## Purpose and voice

- Treat every report as a research document and a stand-alone source of truth. Assume the reader has no knowledge of the discussions, requests, experiments, or incidents that preceded it.
- Use an objective, neutral, research-focused voice. Prefer natural sentences and familiar words. Do not use slogans, catchy headings, promotional language, marketing claims, or exaggerated conclusions.
- Explain the problem, relevant background, scope, method, evidence, result, limitations, and conclusion in enough detail for a new reader to follow the reasoning.
- Introduce technical terms in plain language when they first appear. Use jargon only when it adds necessary precision, and define it immediately.
- Define every mathematical variable and symbol when it first appears in prose. State units, ranges, indices, and conventions when they matter. Keep notation consistent throughout the report.
- Do not mention the request, prompt, requester, developer lead, stakeholder instructions, or report-generation process. Never write framing such as “given the request” or “the dev lead asked.”
- Do not narrate production mechanics such as which viewer, browser automation, encoder, or command generated an artifact. Include implementation details only when they affect scientific interpretation, reproducibility, performance, or correctness.
- Write conclusions at the strength supported by the evidence. Separate observation, interpretation, uncertainty, and speculation.

## Report identity and dates

- Put each report in `/<report-name>/` at the root of this published branch unless an established report uses another documented location.
- A report folder name must contain one to five brief, content-specific words separated by hyphens, for example `newton-xpbd-fluids` or `featherpgs-issues`. Use lowercase ASCII letters and numbers. Do not use generic names such as `new-report`, dates alone, or internal task identifiers.
- Show report dates at the top of the article, before or directly beside the title metadata.
- Always show `Created YYYY-MM-DD`. If the report has changed on a later day, also show `Modified YYYY-MM-DD`.
- Use ISO calendar dates exactly as `YYYY-MM-DD` in visible text and in `<time datetime="YYYY-MM-DD">` attributes. Do not use month names, locale-dependent dates, or timestamps without a date.
- The created date is immutable after publication unless its provenance was recorded incorrectly. Update the modified date whenever user-visible content, data, analysis, figures, conclusions, or presentation changes on a later day. Tooling-only changes that cannot affect the rendered report do not require a date change.

## Overview-page registration

- Every new report must be linked from the overview page served at `https://reports.eric-heiden.com/`. A direct report URL is not sufficient.
- Update the overview entry in the same change as the report. The entry must use the report's current title and a brief, faithful abstract.
- When a report title or abstract changes, update the overview entry in the same change so the two never disagree.
- Locate and edit the source that actually renders the production overview; do not create a competing index page. If the checked-out branch does not contain that source, report the missing integration point instead of claiming the report is complete.
- Use relative links that work in local previews and on the production domain. Verify the overview link before finishing.

## Aperture publication style

- Use the Aperture report design as the default visual language.
- Bright mode is the default and must use a pure white (`#ffffff`) page background. Dark mode must remain readable and must preserve the same information hierarchy.
- Use a compact article layout: approximately 760 px for figures and no more than approximately 680 px for prose. Keep the report title at or below 32 px on desktop and approximately 26 px on mobile. Keep section headings near 18–19 px.
- Use one coherent spacing scale and a consistent vertical rhythm. Align prose, headings, equations, code, tables, figures, and captions to the article column.
- Integrate plots, videos, tables, and diagrams into the reading flow like figures in a technical blog post. Use ordinary numbered captions immediately below them.
- Prefer flat, unframed content. Do not use KPI strips, summary-stat cards, dashboard tiles, marketing heroes, prominent calls to action, badges, decorative gradients, oversized type, ornamental shadows, or repeated rounded panels.
- Use accent colors sparingly for data series, links, focus states, and meaningful status distinctions. Color must communicate information rather than decorate the page.
- Make every layout responsive down to 320 px. Avoid page-level horizontal scrolling. Use contained horizontal scrolling only for tables or code that cannot reflow safely.
- Keep the bright/dark control in the navigation and persist the reader's selection locally. The first visit must render in bright mode.

## Mathematics

- Render mathematical notation from LaTeX with KaTeX, MathJax, or the site's existing equivalent. Do not use screenshots of equations.
- Use inline math for symbols used within a sentence and display math only for equations that benefit from their own line.
- Define every symbol directly before or after the equation. Explain what the equation measures and how to interpret larger, smaller, positive, negative, or limiting values.
- Pin the renderer version and load it with `defer`, lazy loading, or build-time rendering so mathematics does not block the article's first meaningful paint.
- Verify equations in both themes and at mobile width. A rendering failure must leave understandable fallback text.

## Plots and quantitative figures

- Prefer Plotly for interactive plots. Use a pinned version and the smallest official partial bundle that supports the required trace types. Defer or lazy-load it when possible.
- A plot must have labeled axes, units, readable ticks, a concise caption, accessible alternative text, and a clear explanation in the surrounding prose.
- Explain plotted quantities before asking the reader to interpret them. Define thresholds, reference lines, error bands, normalization, aggregation, and transformations such as logarithmic scales.
- Keep plots transparent and embedded in the article rather than inside dashboard cards. Match plot colors, grid lines, labels, and hover content to bright and dark modes.
- Preserve important values and uncertainty. Do not invent secondary metrics, scores, or KPI summaries merely to fill space.
- Use tables for exact values when the reader needs them. Do not duplicate the same data across multiple decorative visualizations.

## Source code

- Include source code only when it materially improves understanding or reproducibility. Prefer short, focused snippets over complete files.
- Use semantic `<pre><code>` markup with readable syntax formatting. Prefer build-time or lightweight formatting; do not add a large runtime highlighter for a small snippet.
- Explain the purpose of the snippet and any identifiers that are central to the report. Do not explain routine syntax line by line.
- Ensure code remains copyable, selectable, and readable in both themes. Allow code-local horizontal scrolling without causing page-level overflow.

## Diagrams, video, and other evidence

- Use Mermaid for architecture and process diagrams when labeled nodes and connections can explain the structure. Keep diagrams small, neutral, and close to the paragraph that introduces them.
- Simplify diagrams to the minimum set of nodes needed for the argument. Use a horizontal flow when space permits and a vertical flow on narrow screens.
- Treat videos as evidence, not decoration. Provide controls, a representative poster, `playsinline`, and `preload="metadata"`. Add a caption explaining what the reader should observe.
- Do not describe media encoding or rendering tools unless those details affect the evidence.
- State whether data and media are measured, simulated, illustrative, or representative. Never present representative template data as an experimental measurement.

## Performance, accessibility, and verification

- Prefer semantic HTML and native controls. All interactions must work by keyboard, maintain visible focus, and expose useful accessible names.
- Pin third-party dependencies. Avoid duplicate libraries, unused frameworks, blocking scripts, and unnecessarily large bundles.
- Verify each report at 1024 px, 736 px, and 360 px in both bright and dark modes. Check for page-level overflow, clipped labels, unreadable captions, broken equations, failed diagrams, console errors, and media failures.
- Exercise interactive plots, theme switching, copy controls, sliders, navigation, and video loading before finishing.
- Confirm that bright mode has a computed white page background and that a first visit with no saved preference opens in bright mode.
- Do not publish or push changes unless the user explicitly requests it.
- Never include “Codex” or another assistant/tool name in public-facing branch names, pull-request titles, report titles, or overview entries.

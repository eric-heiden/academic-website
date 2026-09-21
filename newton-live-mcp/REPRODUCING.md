# Reproducing the Newton comparison report

The report separates the original published two-way study (`historical.html`)
from the registered three-way comparison under `data/v2/`. The historical HTML
and its referenced evidence are preserved; its trials are not pooled with the
current comparison.

The simulation source is Newton revision
`fe4fda0199adae250d0ed1d5e37cb91f98c89f77`. Its
`tools/mcp_evaluation/README.md` describes the application, actual IPython MCP
integration, task runner, preparation, and fresh verification. The frozen
environment and registration are linked from the report. All timed contexts use
GPT-6 Astra with `xhigh` reasoning. The corrected-IPython sensitivity is separate
from the unmodified-upstream primary comparison.

## Recorded results and accounting

`data/v2/comparison.json` contains every registered row, independently recomputed
token/candidate/process accounting, study eligibility, numerical quality, group
totals, and matched-pair statistics. The CSV files provide tabular copies.
Failed contexts retain their costs and remain in success denominators. Ratios
only use jointly successful eligible cases. The small-sample bootstrap intervals
and exact tests are descriptive, unadjusted comparisons.

`build_comparison.py` audits original workspaces and creates these records:

```sh
uv run --no-sync python build_comparison.py \
  --registration /path/to/CONFIRMATION_REGISTRATION.json \
  --output data/v2 --require-complete --strict-audit --export-evidence
```

The registration contains original local workspace paths. Re-auditing original
raw hashes requires those original records. Public evidence copies intentionally
remove credentials, private connection descriptors, and embedded image payloads;
their export manifest records both original and exported digests. Redacted copies
must not be substituted for originals while claiming raw-hash equivalence. The
logged-command audit is distinct from mechanical accounting and is not an
operating-system access-control proof.

The completion ledger records a summary digest and cost fields as each trial
finishes. `data/v2/completion-ledger-audit.json` checks registered order,
nonoverlapping recorded intervals, summary digests, and exact agreement with
the independently audited cost fields. A complete ledger interval also includes
post-agent verification, so it is longer than the reported performance timer.
These are local provenance records, not an external timestamp service.

`render_comparison.py` creates the accessible HTML tables from audited JSON.
For readability, the primary tables take the reciprocal of the builder's
ratios, reversing interval endpoints as well. Values below
one favor the first named method. Sensitivity ratios retain corrected/original
orientation. All raw pair values and orientations remain in `comparison.json`.

## Physical data and verification

`data/v2/real-robot-data.zip` contains measured references, immutable Newton
features, sanitized geometry, licenses, and manifests. It excludes fitted
candidate parameters and publisher estimator code. Per-agent submitted numeric
configurations are in the corresponding public trial records. The archive's
README gives commands to independently verify a selected configuration against
the full training and held-out references.

Reference arrays and features match recorded task/verifier digests. Geometry
membership and bytes match the registered integrity manifest. Conversion
metadata was not individually registered as an agent input; its reference digest
is checked at export and this distinction appears in the archive manifest.
The held-out files are released only after all timed contexts have completed.

The public data derive from physical Panda recordings in
https://doi.org/10.5281/zenodo.12516500, attributed to MERL (2024), Giacomuzzo,
Carli, Romeres, and Dalla Libera under CC-BY-SA-4.0. Geometry retains its separate
Apache-2.0 license. Fitted effective dynamics need not uniquely recover actual
link properties. The task evaluates fixed 100 ms windows under measured
generalized torque, not raw motor commands, grasping, or hardware deployment.

## Figures

`assets/v2/comparison-figure-data.json` stores exact plotted ratios, quality
values, fixed-window samples, and omissions. Figures use all registered real-primary
submissions for aggregate quality, and the prespecified replicate-zero/window-five choice
for trace illustrations. No failed or missing window is replaced with another.

`data/v2/real-trace-windows.zip` contains the fixed-window NPZ excerpts needed
to regenerate the plots, plus full initial-model quality JSON. Its manifest
records original full-trace and excerpt digests measured at export, together with
configuration and reference checks against the audited results. The initial
model is checked against the registered uniform configuration. These NPZ excerpts have only
50 steps each; do not treat them as complete verification trajectories.

After unpacking the trace archive, with `comparison.json` and the report's
Python plotting dependencies available:

```sh
uv run --no-sync python plot_comparison.py \
  --comparison comparison.json --trial-root confirmation \
  --initial-training initial/training/metrics.npz \
  --initial-heldout initial/heldout/metrics.npz --output regenerated-figures
```

The script exports light/dark SVGs, PNGs, and mobile layouts. Mobile and desktop
layouts must return identical numerical evidence. Dark SVGs alter display colors,
not curve geometry or measurements. Nonfinite simulation values remain gaps and
JSON nulls. Both pooled and per-recording numerical quality gates are included.

The report utility regressions use `unittest`:

```sh
uv run --no-sync -m unittest test_build_comparison test_report_figures
```

Run report generation, plotting, or extra simulation checks after timed trials
finish; the confirmation study was serialized to avoid competing simulation jobs.

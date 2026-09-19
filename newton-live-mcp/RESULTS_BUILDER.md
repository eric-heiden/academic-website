# Report data builder

`build_results.py` reads the registered confirmation trial workspaces. It never
imports Newton or executes submitted code, and does not modify the report HTML.
Development summaries and their explicit accounting corrections remain separate.

For an interim data snapshot and sanitized evidence exports:

```bash
uv run --no-project python build_results.py \
  --registration /home/horde/artifacts/newton-live-mcp/confirmation-registration.json \
  --trials /home/horde/artifacts/newton-live-mcp/trials/confirmation \
  --export-evidence
```

Add `--require-complete` for final report data. That option fails before writing
if any registered summary is absent or incomplete. Neither mode publishes.
After an amendment, consume the canonical amended registration and add
`--registration-history /home/horde/artifacts/newton-live-mcp/confirmation-registration-v1.json`
to preserve the prior record and its original-file hash. The history option is
repeatable. The builder never changes any source registration file.

Outputs in `data/confirmation/`:

- `results.json`: every planned trial, success gates, timings, token subsets,
  original/calibration grouping, all completed failures, and per-pair ratios.
- `registration.json`: execution-order record and its temporal qualification.
- `registration-history/`: explicitly supplied earlier records, with original and
  sanitized hashes in `results.json`.
- `infrastructure/<run>/`: explicitly registered pre-agent startup logs, separate
  from agent trials and all token/time aggregates.
- `candidate-history.json`: candidate measurements from completed-run logs,
  including failures. Within-file order is exact; cross-file order is inferred
  from file modification time and disclosed as potentially approximate.
- `trials/<run>/`: allowlisted summaries, task/configuration, agent events,
  process/rollout logs, and final verification metrics, when export is requested.
- `export-manifest.json`: original and exported hashes, sizes, record counts,
  malformed lines, and explicit omissions for oversized files/records.

Success requires verified physical quality, zero agent exit status, no timeout,
the agent-time and completed-candidate budgets, unchanged sources, matching
registered source/task hashes, and consistent timing. Calibration additionally
requires passing training and held-out checks, unchanged references, and matching
registered reference commitments. Pair
ratios are computed only for two successful runs with matching source/reference
identities. All completed failures remain present and contribute to explicitly
labeled all-completed totals. Ratios are restart/live; they are not price ratios.
Per-trial `source_commit` and `task_source_sha256` override the original global
version; `trial_id` preserves logical identity when `run_id` and `workspace`
identify a retry. Different source versions within a pair suppress its ratios.
`infrastructure_attempts` must explicitly record `agent_launched: false`; each
item may list relative `logs` paths from the registration directory. A rejected
startup is not a nineteenth agent trial or an inferred zero-token agent result.
The default is `logs_base: "registration_directory"`, matching the canonical
registration's artifact-relative log paths. `logs_base: "workspace"` explicitly
selects paths inside the attempt workspace; other locations remain disallowed.

Completed candidate logs are cross-checked against the summary count. Event
accounting distinguishes command/MCP calls from file-change actions, direct
observation calls, and returned MCP image blocks. It does not infer that images
drove the agent's decisions. Candidate histories expose the first passing
candidate and any later refinement; full agent elapsed time includes those
choices, whereas the separate scripted benchmark isolates process overhead.

The exports redact common credential keys/patterns, private-calibration artifact
references, and embedded image/base64 payloads. Connection descriptors, reference
NPZs, private generating parameters, and seed files are never opened or exported.
Sanitized transcripts require a final content audit before publication; original
hashes allow the retained private originals to be identified. Public calibration
references, the frozen calibration manifest, and the eventual post-study truth
reveal are separate, deliberately authorized exports after the agents finish.

Tests use only stdlib fixtures:

```bash
uv run --no-project python -m unittest test_build_results -v
```

Final presentation should lead with confirmation results, retain development and
historical feasibility as supporting evidence, plot calibration candidate errors
from these saved measurements, and replace the main historical HUG video with
the corrected final-config MCP capture. No unpaired result establishes a win.

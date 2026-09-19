# Final evaluation fairness and accounting audit

All **18 eligible confirmation agent trials** were audited. Their raw records
reconcile with their summaries, and all submitted configurations pass their
independent final verifier. No forbidden input access, shared-source edits,
extra simulations, process-lifecycle violation, or outcome-dependent omission
was observed in the recorded workflows. The accounting checks report zero
mismatches. This is a bounded artifact audit, not an operating-system access
trace or a rerun of the physics.

Audit completed: 2026-09-19T10:06:50.184318+00:00. The first-pair review is retained separately in
[EVALUATION_AUDIT.md](EVALUATION_AUDIT.md); this document supersedes its limited
coverage. The reproducible stdlib audit is
[recompute_evaluation_audit.py](recompute_evaluation_audit.py), with exact values,
per-trial log paths, PIDs, all scored threshold metrics, and check results in
[EVALUATION_AUDIT_RECOMPUTED.json](EVALUATION_AUDIT_RECOMPUTED.json).

## Coverage and method

Reviewed the runner's task preparation, prompts, allowed actions, model settings,
source commitments, process cleanup, recursive log accounting, token accounting,
wall-time boundaries, training-result matching, and separate verification path.
Reviewed the frozen [original protocol](EVALUATION_PROTOCOL.md),
[calibration protocol](CALIBRATION_PROTOCOL.md), and
[amended execution-order record](confirmation/registration.json).

Read every eligible trial's completed command, MCP arguments and recorded file
changes in agent.jsonl, including nested shell/Python candidate batches. Checked
all 18 task manifests, summaries, saved configurations, recursive candidate and
process JSONL records, and verification metrics/process records. All nine paired
task.json objects are identical after removing only condition: initial values,
bounds, thresholds, duration, cameras, reference commitments and source hashes
match. Each pair uses the same source revision. Model and reasoning effort match
`gpt-6-astra` / `xhigh`; every task has the same 600-second agent budget and
12-completed-candidate ceiling. Optional discovery and observations do not force
extra live calls; both conditions may batch candidates.

No simulation, test, private calibration truth/seed/held-out input, or evaluator
steering was used for this audit. Only exported verification *metrics* were read.
Training reference copies were hashed; private held-out references were not
opened or hashed independently. Their committed hashes agree across task and
verification records, and the inspected runner integrity check reports unchanged
references for all six calibration trials.

## Reconciled trial measurements

Order follows the amended execution record. C/F means completed candidates /
failed-quality candidates; P is distinct trial simulation processes. Commands
includes failed completed shell commands. MCP counts are actual completed calls
to the `newton` server. Time includes the live application startup plus agent
execution and agent-process cleanup. It excludes subsequent live application
shutdown and the independent verifier; it is not total orchestration time.
Verification is excluded in both conditions. Values below are rounded only for
display; the linked summaries and recomputed JSON retain exact floats.

| # | Trial / source | C/F | P | Commands | MCP | Agent s | Startup s | Inclusive s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | [panda_calibration-live-0](confirmation/trials/panda_calibration-live-0/summary.json) / v1 | 9/8 | 1 | 6 | 6 | 103.918 | 3.509 | 107.427 |
| 2 | [panda_calibration-restart-0](confirmation/trials/panda_calibration-restart-0/summary.json) / v1 | 9/8 | 9 | 15 | 0 | 191.146 | 0.000 | 191.146 |
| 3 | [panda_calibration-restart-1](confirmation/trials/panda_calibration-restart-1/summary.json) / v1 | 9/6 | 9 | 10 | 0 | 154.048 | 0.000 | 154.048 |
| 4 | [panda_calibration-live-1](confirmation/trials/panda_calibration-live-1/summary.json) / v1 | 9/5 | 1 | 4 | 4 | 86.940 | 3.507 | 90.447 |
| 5 | [panda_calibration-live-2-attempt2](confirmation/trials/panda_calibration-live-2-attempt2/summary.json) / v2 | 9/8 | 1 | 7 | 6 | 90.546 | 3.557 | 94.104 |
| 6 | [panda_calibration-restart-2](confirmation/trials/panda_calibration-restart-2/summary.json) / v2 | 9/8 | 9 | 14 | 0 | 169.567 | 0.000 | 169.567 |
| 7 | [panda-live-0](confirmation/trials/panda-live-0/summary.json) / v2 | 1/0 | 1 | 6 | 2 | 41.116 | 3.507 | 44.624 |
| 8 | [panda-restart-0](confirmation/trials/panda-restart-0/summary.json) / v2 | 1/0 | 1 | 8 | 0 | 46.330 | 0.000 | 46.330 |
| 9 | [panda-restart-1](confirmation/trials/panda-restart-1/summary.json) / v2 | 1/0 | 1 | 8 | 0 | 50.590 | 0.000 | 50.590 |
| 10 | [panda-live-1](confirmation/trials/panda-live-1/summary.json) / v2 | 1/0 | 1 | 4 | 2 | 42.776 | 3.556 | 46.332 |
| 11 | [allegro-live-0](confirmation/trials/allegro-live-0/summary.json) / v2 | 2/1 | 1 | 6 | 3 | 44.681 | 2.556 | 47.237 |
| 12 | [allegro-restart-0](confirmation/trials/allegro-restart-0/summary.json) / v2 | 1/0 | 1 | 12 | 0 | 61.511 | 0.000 | 61.511 |
| 13 | [allegro-restart-1](confirmation/trials/allegro-restart-1/summary.json) / v2 | 1/0 | 1 | 10 | 0 | 63.257 | 0.000 | 63.257 |
| 14 | [allegro-live-1](confirmation/trials/allegro-live-1/summary.json) / v2 | 1/0 | 1 | 7 | 2 | 51.765 | 2.556 | 54.321 |
| 15 | [hug-live-0](confirmation/trials/hug-live-0/summary.json) / v2 | 1/0 | 1 | 6 | 4 | 71.110 | 3.457 | 74.567 |
| 16 | [hug-restart-0](confirmation/trials/hug-restart-0/summary.json) / v2 | 1/0 | 1 | 11 | 0 | 87.144 | 0.000 | 87.144 |
| 17 | [hug-restart-1](confirmation/trials/hug-restart-1/summary.json) / v2 | 3/1 | 3 | 11 | 0 | 80.133 | 0.000 | 80.133 |
| 18 | [hug-live-1](confirmation/trials/hug-live-1/summary.json) / v2 | 1/0 | 1 | 5 | 3 | 54.745 | 11.611 | 66.356 |

All 69 completed candidate records are retained, including 45 that failed the
quality criteria. Calibration uses nine complete 3000-step candidates per trial;
original tasks use complete 1500-step candidates. There are 44 distinct trial
simulation processes and 18 separate verification processes. Live calibration
uses three processes across 27 candidates; restart calibration uses 27. Each live
trial uses one simulation process. Each restart candidate uses a distinct fresh
process. Recursive counts include nested outputs such as
`allegro-restart-1/runs/candidate_01` and `hug-restart-0/candidate01`, while excluding
all verification directories.

All 32 live MCP calls complete successfully. They are describe/execute calls;
none of these agents chose observe or rebuild. Each live candidate uses validated
`apply_config`, reset, fixed-count step, and metrics. Numerical calibration fits
use the supplied reference and the agent's own candidate traces; they do not run
additional forward simulations. Restart commands use the standard rollout entry
point and exit after each candidate. Shared forward-model and public asset reads
are permitted. HUG live0's model mass/armature/damping inspection is read-only.
The 17 file-change tool items target only the agent's own config.py; the remaining
calibration restart1 configuration writes occur inside counted shell commands.
No recorded command searches other trial results or private calibration inputs.

Trial 16 preserves a failed asset-path read (`item_7`, exit 2) before retrying the
correct permitted asset path. Its time and command event are counted. Trial 11
retains its failed initial Allegro candidate; trial 17 retains its failed initial
HUG candidate and both subsequent passing repeats. No agent timeout or failed
agent exit occurred. Process-start timestamps follow the registered order and
start after the preceding eligible trial's summary timestamp. This supports
serial simulation execution without independently measuring every operating
system process lifetime.

## Token accounting

Every usage field equals the sum of raw `turn.completed` usage records. Cached
input is a subset of input, reasoning output is a subset of output, and all
cache-write counts are zero. Uncached is input minus cached; it is shown to avoid
confusing total-input changes with measured cost changes.

| # | Input | Cached input | Uncached input | Output | Reasoning output |
|---:|---:|---:|---:|---:|---:|
| 1 | 310838 | 263168 | 47670 | 2462 | 467 |
| 2 | 496656 | 449024 | 47632 | 4059 | 886 |
| 3 | 369595 | 333568 | 36027 | 3271 | 855 |
| 4 | 194035 | 172928 | 21107 | 1830 | 292 |
| 5 | 319271 | 290176 | 29095 | 2186 | 185 |
| 6 | 434550 | 407040 | 27510 | 4165 | 967 |
| 7 | 137115 | 120832 | 16283 | 899 | 129 |
| 8 | 125401 | 99072 | 26329 | 1028 | 237 |
| 9 | 159723 | 138368 | 21355 | 973 | 192 |
| 10 | 113928 | 102400 | 11528 | 866 | 178 |
| 11 | 140830 | 124928 | 15902 | 1024 | 97 |
| 12 | 214749 | 183040 | 31709 | 1157 | 150 |
| 13 | 163528 | 135296 | 28232 | 1241 | 252 |
| 14 | 137167 | 122624 | 14543 | 1033 | 134 |
| 15 | 232036 | 201472 | 30564 | 1512 | 278 |
| 16 | 327101 | 284288 | 42813 | 1917 | 520 |
| 17 | 312829 | 282496 | 30333 | 1666 | 283 |
| 18 | 150053 | 132352 | 17701 | 1230 | 159 |

The runner's `tool_items` is **command + MCP events**, totaling 182 here, and
excludes 17 completed file-change items. It must not be labeled an exhaustive
tool-action total. This report exposes the components and does not change frozen
measurement code. In calibration variants 0 and 2, live has fewer total input
tokens but slightly more uncached input than restart. No dollar-cost conclusion
follows from total input alone.

## Independent quality checks

For every candidate, recomputed its success decision from the recorded finite
flag, expected frame/sample count, every fixed threshold, and calibration
per-episode thresholds. Every candidate config is within the published bounds.
Parsed submitted config.py values without executing them and matched each to a
complete passing training candidate and the verifier's config. Every verifier
runs in a separate PID after the trial's simulation process starts. Runner code
stops the agent/server before launching it and writes into
`verification/metrics.json`; stale training metrics are not reused.

All six calibration final results combine exact submitted-config training
success, unchanged reference checks, and fresh held-out episode `[2]` success.
Training logs contain episodes `[0,1]` and 3000 samples; final held-out metrics
contain 1500 samples. All 12 original-task fresh verifications exactly match the
submitted candidate's full set of scored metrics, configuration, finite flag,
and frame/sample count. The two agents in a pair may submit different passing
configurations; no equality between their configurations or errors is claimed.

The following primary RMSE values are checksums for reading the linked full
quality records, not a substitute for their complete threshold checks. Calibration
uses held-out trajectory RMSE in rad; Panda/Allegro use tracking RMSE in rad; HUG
uses wrist RMSE in m. All other required criteria also pass.

| Task | Variant | Live RMSE | Restart RMSE | Both final passes |
|---|---:|---:|---:|---|
| panda_calibration | 0 | 0.000201503822716 | 0.000201513184959 | Yes |
| panda_calibration | 1 | 0.0002012094966 | 0.000201208941896 | Yes |
| panda_calibration | 2 | 0.000200418286168 | 0.000200422684585 | Yes |
| panda | 0 | 0.00517681303806 | 0.00781227676651 | Yes |
| panda | 1 | 0.00770085019364 | 0.00407911859992 | Yes |
| allegro | 0 | 0.00577171543945 | 0.00720529897857 | Yes |
| allegro | 1 | 0.00879860965391 | 0.0148213479735 | Yes |
| hug | 0 | 0.0103900710662 | 0.0102454328638 | Yes |
| hug | 1 | 0.00999082358833 | 0.0113991080688 | Yes |

Quality was recomputed from exported scalar/episode measurements and the fixed
rules; this audit did not regenerate trajectories or independently reconstruct
private held-out errors.

## Source amendment and retained infrastructure failure

The first four trials used v1 commit
`7790cd99713f66e98c2ca6cc2b489f12dd9676c2`; the remaining 14 used v2 commit
`cd9bfdd866459a58e26a3ce6687f741c40d1f26b`. Every task and summary source manifest
matches its **per-trial** amended registration; current frozen files match v2.
Both protocol hashes remain unchanged. It would be incorrect to describe the
18 trials as using one unchanged source revision.

Inspected the v2 diff: only scenario-dependent variant CLI validation in
rollout.py and its test changed. Original Panda/Allegro/HUG variants remain
restricted to 0/1; calibration permits nonnegative variant identifiers. The diff
contains no physics, targets, thresholds, model, prompt, scoring, data or budget
change. First-four v1 files and the explicit amendment remain preserved.

The original trial-5 attempt is retained at
[its startup log](confirmation/infrastructure/panda_calibration-live-2/trials/confirmation/panda_calibration-live-2/server.log) and
[runner log](confirmation/infrastructure/panda_calibration-live-2/confirmation-runner-logs/05-panda_calibration-live-2.log). Its parser
rejects variant 2 before any agent transcript exists; the server-log digest
matches the amendment. It is a **pre-agent infrastructure failure**, elapsed
2.681858 seconds in the retained record, and is not one of the 18 eligible agent
outcomes. Its replacement is explicitly named `panda_calibration-live-2-attempt2`,
not an overwrite. The retained attempt, source amendment and correction effort
must remain disclosed alongside results; their time is outside the per-agent
comparison table. The registration points to preserved red/green correction logs
and a real CLI smoke artifact; this audit did not rerun them.

## Interpretation limits

The execution-order record was written after the first live outcome was known.
It explicitly records an earlier claimed order decision and does not claim to
be a pre-outcome registration. The exact earlier message timestamp cannot be
independently established from these files. The observed order and sample size
match that disclosed record, with the one documented pre-agent infrastructure
retry; no outcome-dependent exclusion was found.

The six calibration trials are a separately labeled exploratory extension to
the 12 original-task trials. Many original tasks finish in one candidate; this
limits what they establish about sustained iterative editing. All outcomes pass,
but these small, fixed task sets do not establish a general productivity effect.
Report the actual time/token/candidate distributions and their cache and startup
qualifications. In particular, trial 18's 11.611013950780034-second live startup is
included in its 66.35555855650455-second inclusive time rather than discarded.

The audit did not independently verify all external asset bytes, operating-system
file accesses, complete compilation-cache warmup history, or wall-clock timer
accuracy. Logged process order, reviewed workflows, exact source commitments,
and metric reconciliation support the conclusions within those limits. No
frozen Newton source was changed by this audit.

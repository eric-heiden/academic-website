# Evaluation fairness and accounting audit

Audit time: 2026-09-19T09:40:59.019498+00:00.
Read-only audit during serial confirmation trials; no simulations, tests, source
changes, or evaluator steering were performed. Private truth, seeds, held-out
response files and their contents were not opened.

## Scope and conclusion

The completed first calibration pair, `panda_calibration-live-0` and
`panda_calibration-restart-0`, conforms to the inspected task and workflow rules.
The nine-candidate counts, process counts, usage totals and final-quality records
reconcile with raw evidence. No forbidden input access, extra simulation, shared
source modification or outcome-dependent exclusion was observed in this pair.
This conclusion is bounded by the recorded commands and artifacts; it is not an
operating-system access audit or a conclusion about all 18 outcomes.

Inspected:

- `tools/mcp_evaluation/run_agents.py`, including task preparation, model/settings,
  source commitments, recursive log counting, usage accounting, process cleanup,
  training-result matching and fresh held-out verification.
- [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md),
  [CALIBRATION_PROTOCOL.md](CALIBRATION_PROTOCOL.md), and
  [confirmation-registration.json](confirmation/registration.json).
- Both first-pair TASK.md, task.json, config.py, summary.json, complete agent.jsonl
  command/MCP/file-change/message records, candidate/process JSONL records, and
  verification process metadata. Training reference copies were hashed without
  interpreting their arrays. Private reference contents were not read or hashed.

## Reconciled first-pair measurements

| Measurement | Live | Restart |
|---|---:|---:|
| Completed training candidates | 9 | 9 |
| Training steps per candidate | 3000 | 3000 |
| Distinct simulation processes during trial | 1 | 9 |
| Separate verification processes | 1 | 1 |
| Actual MCP call events | 6 | 0 |
| Command execution events | 6 | 15 |
| `tool_items` in summary | 12 | 15 |
| Separate completed file-change events | 1 | 1 |
| Agent wall time, seconds | 103.9183834661 | 191.1463813568 |
| Separately measured live startup, seconds | 3.5088088652 | 0 |
| Startup-inclusive time, seconds | 107.4271923313 | 191.1463813568 |
| Total input tokens | 310838 | 496656 |
| Cached input tokens | 263168 | 449024 |
| Input minus cached input tokens | 47670 | 47632 |
| Output tokens | 2462 | 4059 |
| Reasoning output tokens (subset) | 467 | 886 |
| Cache-write input tokens | 0 | 0 |
| Final training RMSE, rad | 0.0002004114801882 | 0.0002004007128393 |
| Fresh held-out RMSE, rad | 0.0002015038227157 | 0.0002015131849589 |
| Training and held-out success | true / true | true / true |
| Timed out | false | false |

Sources: [live summary](confirmation/trials/panda_calibration-live-0/summary.json),
[restart summary](confirmation/trials/panda_calibration-restart-0/summary.json),
[live raw events](confirmation/trials/panda_calibration-live-0/agent.jsonl), and
[restart raw events](confirmation/trials/panda_calibration-restart-0/agent.jsonl).
Both raw transcripts contain one `turn.completed` event; every reported usage
field equals its raw value. Cached input and reasoning output are not added a
second time. Startup-inclusive time equals agent time plus live startup exactly.
These timing measures exclude the post-agent independent verification in both
conditions; they should not be relabeled total execution time including scoring.

## Fairness checks

**Same information and action constraints.** The two complete task.json objects
are identical after removing only `condition`, including initial configuration,
bounds, noise level, episode length, thresholds, camera and all source/reference
commitments. Both receive the same training reference digest. Neither task.json
nor TASK.md contains a private truth or held-out file path. The source presents
the same forward model and numerical observations, with the intended transport
and process-lifecycle difference. Discovery is optional. Neither agent was forced
to spend a minimum number of iterations or make mandatory extra inspection calls.

Both agents autonomously used sensitivity measurements and least-squares fitting.
All nine candidate configurations in each condition lie within the published
bounds. Every candidate log records episodes `[0,1]`, 3000 frames and 3000 samples.
Every candidate counted is preserved. The live log contains nine records; the
restart aggregate includes eight nested candidate directories and the final
root-level output, correctly excluding verification.

**Actual allowed workflow.** The live transcript has six completed `newton` MCP
calls: one describe and five execute calls. The execute batches run 1, 3, 1, 3 and
1 candidates respectively, each through `apply_config`, reset, `step(count=3000)`
and metrics. Its sole simulation PID is 134472. The restart agent runs the
standard rollout command nine times in nine distinct processes, including its
per-candidate subprocess batches; no persistent simulator was used. Both sets of
non-simulation numerical scripts operate on supplied training responses and their
own exported candidate traces. Their only recorded file-change tool edits target
their own config.py; additional shell writes also target config.py. The complete
recorded commands contain no reads/searches of other trials, private truth,
withheld responses, feasibility answers, or private generator files, and no extra
forward simulations outside the prescribed workflow.

**Frozen source/protocol consistency.** All task source hashes equal the
registration's source hashes and independently recomputed current hashes. Both
registered protocol digests match the inspected files. Both training reference
copies match the committed training digest. The same held-out digest appears in
the registration, task manifests and fresh verification metrics. The runner's
reference-integrity check reports true. This audit reviewed that check's code
without opening the private reference itself.

**Independent final quality.** The submitted config.py values exactly match a
logged complete passing training candidate and the final verification config in
each condition. The independent verifier reports episode `[2]`, 1500 frames and
samples, finite state and passing quality. Verification PID 135443 for live and
137171 for restart are distinct from their trial simulation PIDs, match their
verification process logs, and start after the respective candidate process
starts. The runner launches verification after stopping the agent/server and
writes it into a separate verification directory. It requires both passing
training quality of the exact submitted configuration and passing held-out
quality, plus unchanged references; self-reported agent success is insufficient.
No private held-out arrays were inspected by this audit.

## Concrete reporting qualifications

1. **Tool-count label:** `tool_items` deliberately sums command-execution and MCP
   events, and omits `file_change`. Each first-pair transcript has one successful
   configuration edit, so the counts of recorded actionable command/MCP/file-change
   items are 13 live and 16 restart, compared with summary `tool_items` 12 and 15.
   This does not alter their difference, but call the existing field
   “command + MCP calls” rather than an exhaustive tool-call total. No frozen
   runner change is requested during trials.

2. **Cache interpretation:** live uses fewer total input tokens in this pair,
   but uncached input is 47670 versus 47632—nearly identical and slightly higher
   for live. Preserve the protocol's cache qualification; total-input reduction
   alone is not a measured cost reduction. Report input/output and cached subsets
   separately, without double counting or extrapolating from this one pair.

3. **Registration timing:** the execution-order file explicitly states that it
   was written after the first live outcome was available. It records an earlier
   claimed order decision and does not claim to predate trial launch. Preserve
   that qualification: the 18-run execution-order artifact is not a prelaunch
   registration. The earlier decision's exact message timestamp cannot be
   independently verified from these files alone. The inspected first pair
   follows the registered order; no change of order is proposed by this audit.

The third completed summary available at audit time,
[panda_calibration-restart-1](confirmation/trials/panda_calibration-restart-1/summary.json),
was checked only for summary-level status: nine candidates, sources/references
unchanged, within budget, no timeout, success true. Its raw workflow has not been
fully audited here. Remaining trials were not assessed or selected by outcome.

## Limits

This is a bounded artifact audit, not a rerun of physics, a full telemetry audit,
a private-data reconstruction, or a statistical productivity claim. It did not
independently verify every external asset byte, operating-system file access,
wall-clock instrumentation, or the entire compilation-cache warmup history.
The protocol fixes equal warmup and serial execution; those execution details
remain subject to the orchestrator's retained run logs. Continue the already
fixed 18-trial order and retain every eligible result without steering or
outcome-dependent replacement. No source edits were made by this audit.


## Infrastructure event reported at handoff

After this bounded audit was written, the orchestrator reported that registered
trial 5 (`panda_calibration-live-2`) aborted before agent launch because rollout.py
accepted only variants 0 and 1. Timed execution was stopped for a bounded parser
fix. This audit did not independently probe that new failure. Preserve its raw
startup/runner evidence as an infrastructure attempt, identify the corrective
revision and affected trial on resumption, and retain the registered order. Do
not silently overwrite it or classify a pre-agent abort as an agent-quality
failure. The orchestrator reports the first four trials completed; the audit
coverage above remains the first pair in full and trial 3 at summary level.

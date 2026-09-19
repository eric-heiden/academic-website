# Evaluation protocol (before agent trials)

Date: 2026-09-19. Model: GPT-6 Astra, reasoning effort xhigh.

## Conditions

Both conditions receive the same task objective, initial scene, target
trajectory, editable parameter ranges, objective measurements, observation
capabilities, time/attempt budget, and source access. Every trial begins in
a fresh agent context and a separate task workspace. No tuned answers,
previous agent transcripts, or opposite-condition outputs are shared.

- Live: the agent connects to a running Newton simulation using MCP tools.
  It may query, observe, edit parameters, reset and roll out in that process.
- Restart: the agent edits a normal Python configuration and runs a script
  that constructs Newton, performs the rollout and exits. It receives the
  same measurements and can inspect the same rendered observations.

The baseline may choose efficient normal debugging strategies. It must not
be artificially forced to make more trials or write verbose code. Both
conditions can read the same documentation and use the same shell. The
baseline may batch parameter tests, with a fresh simulation process for
each trial, but cannot turn itself into the live condition by keeping a
simulation process active across tests.

## Tasks

1. Menagerie Franka Panda tracking under gravity: tune drive settings and
   assess fixed reference joint/end-effector motion.
2. Menagerie Allegro hand posture tracking: tune dynamics for tracking and
   stability. This is not labeled a successful grasp without a grasp test.
3. HUG recorded manipulation setup: use real MANO hand trajectories and a
   scanned object; assess physical replay/tracking and scene setup choices.
   Distinguish MANO fitted motion from raw sensor measurements and simulated
   contacts from measured contact forces.

Scenario feasibility tests choose explicit quality thresholds before agent
optimization starts. Save thresholds and initial parameters in versioned
scenario manifests. Do not redefine success after inspecting trial results.

## Measurements

- Agent wall-clock time from process launch to final response or timeout.
- Input tokens, cached input tokens, cache-write input tokens, output
  tokens, reasoning output tokens from the CLI usage events. Cached and
  reasoning counts are subsets where the API reports them as such, not
  additional tokens to double count.
- Agent tool calls, simulation rollouts, process starts and final verified
  task quality; final success is checked by the harness, not self-reported.
- Simulation setup/compilation/rollout and live request times separately.
- Raw transcripts and parameters retained, with credentials and local
  session connection secrets excluded from public artifacts.

Run compilation warmup outside the timed agent comparison and report its
cost independently. The restart baseline retains the same on-disk kernel
cache as the live condition. Report live application startup separately as
well as startup-inclusive time, avoiding a free-startup speedup claim.

## Development and confirmation

Use development runs to find and fix MCP bugs or poor ergonomics. Label
those runs as development, retain failures, and avoid presenting them as
independent confirmation evidence. Freeze the implementation and task
definitions before confirmation runs. Prefer two initial-condition seeds
per task and alternate condition order within each pair. Do not run timed
GPU rollouts concurrently. New correctness fixes after freezing require
explicitly identifying the revision and any affected reruns.

Report all eligible confirmation trials, including failed and timed-out
runs. Describe small-sample limitations; two trials per condition are case
studies, not a statistically established universal speedup. If a condition
fails the quality threshold, do not rank its short runtime as a win.

## Separate timing benchmark

A scripted repeated-edit benchmark can isolate process/setup overhead.
Use identical configurations and rollout lengths. It is a systems timing
measurement, not evidence about LLM reasoning quality or token efficiency.

## Portability

Test CPU and CUDA on Linux, actual ViewerGL captures under the available
display, and sensor captures without OpenGL initialization. The local workspace
is Linux; at protocol creation there was no local Windows runner. Subsequent
validation on 2026-09-19 used a Windows Server 2025 GitHub-hosted runner in the
user's fork: [Windows CPU CI](https://github.com/eric-heiden/newton/actions/runs/35434465266)
completed 45 runtime/observation tests at commit
`0c814b59772c9cf6bae41f6c77404fa78f299596`, with 43 passed and two intentional
skips (CUDA and attached ViewerGL). This includes official SDK interoperability
and actual CPU sensor rendering; it does not establish Windows portability of
the full evaluation harness. Windows CUDA and attached ViewerGL remain
unverified. Do not claim EGL independence for ViewerGL; the sensor default is
the context-free path. Exact run metadata and logs are under `windows-validation/`.

# Prospective additional study: synthetic Panda dynamics calibration

This additional study retains and reports the original tracking/setup
study separately. It does not replace unfavorable development or
confirmation results. It tests workflow on simulated reference responses
using a real Menagerie asset, not identification of a physical robot.

The frozen machine-readable specification and commitments are in
`calibration-study/manifest.json`. Three parameter instances are sampled
once, independently and uniformly across the declared bounds. A failure
of reference feasibility aborts registration; no instance is resampled
for being inconvenient or easy. Exact seed and parameter values are
retained in `<withheld-parameter-file>`, outside agent workspaces and
the permitted common source. Publish that file only after all independent
trials have finished, matching the prior public digest.

The unchanged Menagerie Panda inertias/geometry carry a new solid-sphere
payload at the source attachment. Its radius and attachment are fixed.
The unknowns are payload mass [0.2,2.0] kg, an authored viscous-damping
multiplier [0.5,2.0], and common Coulomb joint friction [0,0.3] N m.
Control gains, effort limits, timestep and all command trajectories are
fixed. Reference observations are Newton joint positions with independently
sampled, fixed Gaussian noise of standard deviation 0.0002 rad.

Each candidate consists of two prescribed three-second training episodes,
1500 steps each at dt=0.002 s. The same model resets between episodes in
both conditions. One live `step(count=3000)` and one restart script
invocation therefore perform identical experiments. Both agents receive
`reference.npz` with `episodes=[0,1]` and `q` shaped `[2,1500,7]`. The
public source defines command trajectories; the expected third response
and its generating parameters remain withheld. Final fresh verification
uses the third episode, `episodes=[2]`, 1500 steps.

Quality requires finite state, RMSE <=0.00035 rad and p95 absolute error
<=0.0008 rad for each episode and pooled training samples, and maximum
joint speed <=5 rad/s. Final success requires both a completed, passing
training measurement of the exact submitted configuration and passing
fresh held-out verification. Original task limits are unchanged. Criteria
were fixed using the development sensitivity/noise-floor checks before
any independent calibration agent ran.

Both interfaces export every candidate's measured q, qd, command target
and residual at each step, plus body poses every 25 steps, to NPZ. The
metrics report that trace path, pooled and per-episode quality and joint
residual summaries. Both conditions can inspect identical numerical
signals and rendered poses. Agents may derive analytical estimates or
use numerical fitting, batch candidates, or run independent restart
candidates in parallel within available hardware. No minimum iteration
count is required. The ceiling is 12 complete candidates and 600 seconds
per agent, with partial/failed work retained in raw events.

Run three matched pairs, alternating condition order, in fresh Astra
xhigh contexts. Keep timed trials serial across study cases, apply equal
kernel-cache warmup, retain every eligible result, and report all quality
failures/timeouts. Primary measurements are startup-inclusive wall time
and total input/output tokens, separately. Also report agent-only time,
cached and uncached input, and systems setup/rollout time. A lower count
of total input tokens is not a lower-cost claim without accounting for
cache treatment. A short failed run is not a speed win.

The runner accepts explicit training and private verification inputs:

```bash
uv run --no-sync python -m tools.mcp_evaluation.run_agents \
  --scenario panda_calibration --variant 0 --condition live \
  --workspace /path/to/new/trial --seconds 600 --phase confirmation \
  --reference /path/to/calibration-study/variant-0/training.npz \
  --verification-reference <withheld-verification-file> \
  --run
```

The private verification path is held by the orchestrating harness and is
not written into `TASK.md` or `task.json`. Those files include its digest
only. For restart trials replace `--condition live` with `restart`; other
inputs, budgets and quality requirements are identical. This command
launches an independent agent only with `--run`.

This six-run extension is exploratory. It may show that model/tool latency
outweighs saved model construction; such a result is retained and reported.

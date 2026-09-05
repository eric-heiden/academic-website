# MJWarp PR #1535 SHAC reproduction

This directory contains the self-contained probes and SHAC-style training
harness used for the September 2026 report. The experiment is pinned to
[MJWarp PR #1535](https://github.com/google-deepmind/mujoco_warp/pull/1535) at
commit `02d09b139fdf091e1e859d7f41c47a8f71d30574` and Newton at commit
`d37f4d3d341ccce1e06a1dff21e9a054759b4855`.

There are two deliberately separate environments:

- The **PR-lock environment** uses the PR's committed `uv.lock`. It is the
  smallest and most faithful environment for the one-step finite-difference
  gate in `gradient_probe.py`.
- The **Newton integration environment** uses Newton's current PyTorch/CUDA
  stack, then overlays the exact PR checkout as an editable package without
  changing its dependencies. It runs the multi-step checks and
  `train_shac.py`.

The reported runs used uv 0.11.20 and Python 3.12.13 on an NVIDIA RTX PRO 6000
Blackwell Server Edition MIG 1g.24gb partition with driver 595.58.03.

| Environment | MJWarp | MuJoCo | Warp | NumPy | PyTorch |
| --- | --- | --- | --- | --- | --- |
| PR lock | 3.10.0.1 | 3.10.0.dev932327625 (reports 3.10.0) | 1.14.0 | 2.3.4 | not needed |
| Newton integration | 3.10.0.1, editable PR checkout | 3.12.0 | 1.17.0 | 2.5.0 | 2.11.0+cu128 |

## 1. Create the exact checkouts

The commands below use the paths from the reported run. Change the first four
variables together to reproduce elsewhere. `REPORT_ROOT` must point to this
reports checkout.

```bash
REPOS_ROOT=/home/horde/repos
REPORT_ROOT="$REPOS_ROOT/academic-website-reports"
MJW_PR_ROOT="$REPOS_ROOT/mujoco_warp-pr1535"
NEWTON_SHAC_ROOT="$REPOS_ROOT/newton-shac-pr1535"
MJW_PR_SHA=02d09b139fdf091e1e859d7f41c47a8f71d30574
NEWTON_SHA=d37f4d3d341ccce1e06a1dff21e9a054759b4855
```

Starting from empty destination paths:

```bash
git clone --no-checkout https://github.com/google-deepmind/mujoco_warp.git "$MJW_PR_ROOT"
git -C "$MJW_PR_ROOT" fetch origin pull/1535/head:refs/remotes/origin/pr-1535
git -C "$MJW_PR_ROOT" checkout --detach "$MJW_PR_SHA"

git clone --no-checkout https://github.com/newton-physics/newton.git "$NEWTON_SHAC_ROOT"
git -C "$NEWTON_SHAC_ROOT" checkout --detach "$NEWTON_SHA"

test "$(git -C "$MJW_PR_ROOT" rev-parse HEAD)" = "$MJW_PR_SHA"
test "$(git -C "$NEWTON_SHAC_ROOT" rev-parse HEAD)" = "$NEWTON_SHA"
```

The PR checkout must stay at that exact commit for the published results.
`gradient_probe.py` fails closed on a different commit, and both probes fail
closed if Python imports `mujoco_warp` from another location.

## 2. Reproduce the one-step gradient gate

Create the frozen environment from the PR's lockfile:

```bash
(cd "$MJW_PR_ROOT" && uv sync --frozen)
```

Run the Ant and Humanoid analytic-gradient checks. The script uses the
out-of-place `mujoco_warp.step(model, data_in, data_out)` API, checks five
random control directions against central differences, and writes full
provenance into the JSON output.

```bash
"$MJW_PR_ROOT/.venv/bin/python" \
  "$REPORT_ROOT/shac/pr1535/gradient_probe.py" \
  --pr-root "$MJW_PR_ROOT" \
  --models ant humanoid \
  --device cuda:0 \
  --seed 20260904 \
  --directions 5 \
  --epsilon 0.01 \
  --max-relative-error 0.005 \
  --out "$REPORT_ROOT/shac/pr1535/results/gradient_frozen_lock.json"
```

The Ant asset is vendored at `models/ant.xml`. Humanoid comes directly from
`benchmarks/humanoid/humanoid.xml` in the pinned PR checkout. See
[`models/README.md`](models/README.md) for their hashes and provenance.

## 3. Build the Newton/PyTorch integration environment

Sync Newton's locked CUDA 12 extra, then replace its packaged MJWarp with the
exact PR checkout. `--no-deps` is intentional: it preserves Newton's tested
MuJoCo, Warp, NumPy, and PyTorch versions.

```bash
(cd "$NEWTON_SHAC_ROOT" && uv sync --frozen --extra torch-cu12)
uv pip install \
  --python "$NEWTON_SHAC_ROOT/.venv/bin/python" \
  --no-deps \
  --editable "$MJW_PR_ROOT"

"$NEWTON_SHAC_ROOT/.venv/bin/python" -c \
  'import mujoco, mujoco_warp, numpy, torch, warp; print(mujoco_warp.__file__); print(mujoco.__version__, warp.__version__, numpy.__version__, torch.__version__)'
```

The first printed path must be inside `$MJW_PR_ROOT`. A later `uv sync` in the
Newton checkout can restore Newton's packaged MJWarp, so repeat the editable
install before rerunning these experiments.

For a second one-step result under the newer integration stack:

```bash
"$NEWTON_SHAC_ROOT/.venv/bin/python" \
  "$REPORT_ROOT/shac/pr1535/gradient_probe.py" \
  --pr-root "$MJW_PR_ROOT" \
  --models ant humanoid \
  --device cuda:0 \
  --seed 20260904 \
  --directions 5 \
  --epsilon 0.01 \
  --max-relative-error 0.005 \
  --out "$REPORT_ROOT/shac/pr1535/results/gradient_newton_stack.json"
```

## 4. Reproduce the multi-step gradient sweeps

These checks differentiate a fixed action schedule through horizons 1, 2, 4,
8, 16, and 32, then compare directional derivatives with central differences.
They also record a batched active-contact-count trace. The command exits with
status 1 if **any** requested horizon fails the threshold, but it writes the
complete JSON first. A nonzero status is therefore expected for the published
sweeps and should not cause the output to be discarded.

```bash
TRAINER="$REPORT_ROOT/shac/pr1535/train_shac.py"
PYTHON="$NEWTON_SHAC_ROOT/.venv/bin/python"
RESULTS="$REPORT_ROOT/shac/pr1535/results"

"$PYTHON" "$TRAINER" \
  --mode gradcheck --task ant \
  --pr-root "$MJW_PR_ROOT" --newton-root "$NEWTON_SHAC_ROOT" \
  --device cuda:0 --seed 1701 --worlds 2 \
  --gradcheck-horizons 1 2 4 8 16 32 \
  --gradcheck-directions 5 --gradcheck-eps 0.01 \
  --gradcheck-max-relative 0.05 \
  --output "$RESULTS/ant_multistep_gradient.json"

"$PYTHON" "$TRAINER" \
  --mode gradcheck --task humanoid \
  --pr-root "$MJW_PR_ROOT" --newton-root "$NEWTON_SHAC_ROOT" \
  --device cuda:0 --seed 2301 --worlds 2 \
  --gradcheck-horizons 1 2 4 8 16 32 \
  --gradcheck-directions 5 --gradcheck-eps 0.01 \
  --gradcheck-max-relative 0.05 \
  --output "$RESULTS/humanoid_multistep_gradient.json"
```

## 5. Reproduce SHAC-style training

Training is cold-started: no pretrained actor or critic is loaded. Each run
automatically writes a checkpoint next to its JSON result, for example
`ant_seed17.pt`. The three Ant seeds use horizon 8 and 64 parallel worlds:

```bash
for SEED in 17 29 41; do
  "$PYTHON" "$TRAINER" \
    --mode train --task ant \
    --pr-root "$MJW_PR_ROOT" --newton-root "$NEWTON_SHAC_ROOT" \
    --device cuda:0 --seed "$SEED" \
    --worlds 64 --horizon 8 --epochs 100 --hidden 128 \
    --actor-lr 0.0003 --critic-lr 0.001 \
    --adam-beta1 0.7 --adam-beta2 0.95 --critic-iterations 16 \
    --gamma 0.99 --td-lambda 0.95 --target-polyak 0.4 \
    --max-grad-norm 1 --reset-interval 0 \
    --eval-steps 500 --eval-every 10 \
    --output "$RESULTS/ant_seed${SEED}.json"
done
```

The three exploratory Humanoid seeds use the more conservative horizon 4 and
32 worlds:

```bash
for SEED in 17 29 41; do
  "$PYTHON" "$TRAINER" \
    --mode train --task humanoid \
    --pr-root "$MJW_PR_ROOT" --newton-root "$NEWTON_SHAC_ROOT" \
    --device cuda:0 --seed "$SEED" \
    --worlds 32 --horizon 4 --epochs 100 --hidden 128 \
    --actor-lr 0.0003 --critic-lr 0.001 \
    --adam-beta1 0.7 --adam-beta2 0.95 --critic-iterations 16 \
    --gamma 0.99 --td-lambda 0.95 --target-polyak 0.4 \
    --max-grad-norm 1 --reset-interval 0 \
    --eval-steps 400 --eval-every 10 \
    --output "$RESULTS/humanoid_seed${SEED}.json"
done
```

Every training JSON contains the complete CLI configuration, source hashes,
software versions, model fingerprint, device information, periodic evaluation
history, and an independent holdout comparison between the initial actor and
the best periodically selected actor. The checkpoint stores both the final
actor and the selected `best_actor`.

To reevaluate a saved policy explicitly:

```bash
"$PYTHON" "$TRAINER" \
  --mode evaluate --task ant \
  --pr-root "$MJW_PR_ROOT" --newton-root "$NEWTON_SHAC_ROOT" \
  --device cuda:0 --seed 17 --worlds 64 --eval-steps 500 \
  --checkpoint "$RESULTS/ant_seed17.pt" --checkpoint-policy best \
  --output "$RESULTS/ant_seed17_reevaluation.json"
```

## 6. Reproduce the failure diagnosis

The standalone diagnostic keeps the original v1 training artifacts unchanged.
It measures actor and critic saturation for both saved policy snapshots from
all six v1 runs, replays frozen final-policy action tapes through direct MJWarp
and CPU MuJoCo, and checks
finite differences of the complete closed-loop actor objective—including the
terminal target critic—at nominal and available pre-fall states.

```bash
"$PYTHON" "$REPORT_ROOT/shac/pr1535/diagnose_failures.py" \
  --results-dir "$RESULTS" \
  --pr-root "$MJW_PR_ROOT" --newton-root "$NEWTON_SHAC_ROOT" \
  --device cuda:0 \
  --output "$RESULTS/failure_diagnostics.json"
```

The command fails closed on a different PR head or import location. Its full
JSON retains per-step state-error, contact, and alive traces, plus every
analytic and finite-difference direction; the compact `findings` object
contains the report-ready aggregates.

## 7. Reproduce the corrected v2 training runs

`train_shac_v2.py` is the corrective follow-up. It preserves the v1 artifacts
above and writes v2 JSON/checkpoint pairs under `results/fixed/`. These runs
are still SHAC-style experiments rather than canonical SHAC benchmarks.

```bash
V2_TRAINER="$REPORT_ROOT/shac/pr1535/train_shac_v2.py"
FIXED_RESULTS="$RESULTS/fixed"
mkdir -p "$FIXED_RESULTS"
```

Corrected Ant configuration: 64 worlds, horizon 8, one raw physics step per
action, and 200 epochs.

```bash
for SEED in 17 29 41; do
  "$PYTHON" "$V2_TRAINER" \
    --mode train --task ant \
    --pr-root "$MJW_PR_ROOT" --newton-root "$NEWTON_SHAC_ROOT" \
    --device cuda:0 --seed "$SEED" \
    --worlds 64 --horizon 8 --action-repeat 1 --epochs 200 \
    --actor-lr 0.001 --critic-lr 0.001 \
    --critic-iterations 8 --critic-batches 4 --target-polyak 0.2 \
    --stochastic-std 0.1 --reward-profile legacy \
    --train-noise-profile narrow --eval-noise-profile narrow \
    --eval-steps 500 --eval-every 20 \
    --output "$FIXED_RESULTS/ant_seed${SEED}.json"
done
```

Corrected Humanoid configuration: 32 worlds, horizon 32, three raw physics
steps per action, and 150 epochs.

```bash
for SEED in 17 29 41; do
  "$PYTHON" "$V2_TRAINER" \
    --mode train --task humanoid \
    --pr-root "$MJW_PR_ROOT" --newton-root "$NEWTON_SHAC_ROOT" \
    --device cuda:0 --seed "$SEED" \
    --worlds 32 --horizon 32 --action-repeat 3 --epochs 150 \
    --actor-lr 0.001 --critic-lr 0.0005 \
    --critic-iterations 8 --critic-batches 4 --target-polyak 0.995 \
    --stochastic-std 0.1 --reward-profile diffrl \
    --train-noise-profile canonical --eval-noise-profile canonical \
    --eval-steps 200 --eval-every 25 \
    --output "$FIXED_RESULTS/humanoid_seed${SEED}.json"
done
```

Each v2 checkpoint saves the initial, periodically selected best, and final
actors together with their matching observation-normalizer states.

Validate all six runs, their per-task configuration consistency, recorded
repository heads and source hashes, and their checkpoint files, then create the
deterministic aggregate used by the report:

```bash
python "$REPORT_ROOT/shac/pr1535/summarize_fixed_results.py" \
  --results-dir "$FIXED_RESULTS" \
  --output "$FIXED_RESULTS/summary.json"
```

## 8. Generate the ViewerGL videos

Install the pinned encoder helper into the Newton integration environment;
this does not replace the editable MJWarp checkout.

```bash
uv pip install \
  --python "$PYTHON" \
  imageio-ffmpeg==0.6.0

RENDERER="$REPORT_ROOT/shac/pr1535/render_viewergl.py"
VIDEOS="$RESULTS/videos"
mkdir -p "$VIDEOS"
```

Render any v1 or v2 checkpoint policy at the report defaults of 960x540 and
50 fps. The command writes a browser-compatible H.264/yuv420p fast-start MP4,
a JPEG poster, and a JSON manifest with behavior metrics and hashes.

```bash
PYGLET_HEADLESS=1 "$PYTHON" "$RENDERER" \
  --checkpoint "$FIXED_RESULTS/ant_seed17.pt" \
  --policy best \
  --camera-mode track \
  --pr-root "$MJW_PR_ROOT" \
  --newton-root "$NEWTON_SHAC_ROOT" \
  --overwrite \
  --output "$VIDEOS/ant_v2_best.mp4"

PYGLET_HEADLESS=1 "$PYTHON" "$RENDERER" \
  --checkpoint "$FIXED_RESULTS/humanoid_seed17.pt" \
  --policy best \
  --camera-mode track \
  --pr-root "$MJW_PR_ROOT" \
  --newton-root "$NEWTON_SHAC_ROOT" \
  --overwrite \
  --output "$VIDEOS/humanoid_v2_best.mp4"
```

Use `--policy initial`, `--policy best`, or `--policy final` with the same
checkpoint and reset to make direct visual comparisons. The renderer advances
one fixed, non-noisy lane through MJWarp and freezes it at the first terminal
state. ViewerGL replays that recorded MJWarp state trajectory. Native MuJoCo
is used only for forward kinematics from each recorded `qpos` to body poses;
it does **not** resimulate the trajectory.

## 9. Reproduce the final v3 gait checks

The 5 September full-gait update is in `results/gaits_v3/`. It promotes a
hybrid Ant dynamic-trot controller and a filtered/calibrated Humanoid PPO
policy, then subjects both to guarded SHAC, three independent 1024-lane
full-horizon audits, a nominal audit, and an exact-recorded-trajectory
ViewerGL gate. The final measured ranges are:

| Task | Noisy audit | Horizon | Speed | Final alive | Contact signature |
| --- | --- | ---: | ---: | ---: | --- |
| Ant | 3 seeds × 1024 lanes | 20 s | 1.010–1.015 m/s | 98.44–99.12% | 8.91–9.02% diagonal support; 58.77–59.05% flight |
| Humanoid | 3 seeds × 1024 lanes | 15 s | 1.195–1.199 m/s | 99.61–100% | 83.18–83.27% single support; 7.17–7.27% flight |

The Ant is an aerial dynamic trot, not a grounded walk. The complete gates,
unrounded metrics, source-checkpoint hashes, and video hashes are in
[`results/gaits_v3/summary.json`](results/gaits_v3/summary.json).

Validate the published bundle first. This fails if either task gate is false,
the PR/Newton revisions differ, a SHAC source or output checkpoint hash does
not match, or a video/poster hash differs from its manifest.

```bash
V3="$RESULTS/gaits_v3"
"$PYTHON" "$REPORT_ROOT/shac/pr1535/summarize_gaits_v3.py" \
  --results-dir "$V3"
```

Re-run the definitive audits without overwriting the published JSON:

```bash
"$PYTHON" "$REPORT_ROOT/shac/pr1535/audit_gaits_v3.py" \
  --checkpoint "$V3/ant_checkpoint.pt" --checkpoint-policy best \
  --worlds 1024 --steps 400 --seeds 9851 9863 9877 \
  --output /tmp/ant_final_audit20s.json

"$PYTHON" "$REPORT_ROOT/shac/pr1535/audit_gaits_v3.py" \
  --checkpoint "$V3/humanoid_checkpoint.pt" --checkpoint-policy best \
  --worlds 1024 --steps 600 --seeds 9801 9811 9829 \
  --output /tmp/humanoid_final_audit15s.json
```

The final guarded-SHAC calls are exactly reproducible from the published
source checkpoints. Each signed candidate receives a randomized and a
nominal uninterrupted full-horizon evaluation; both gates must pass, and the
weaker selection score must improve.

```bash
"$PYTHON" "$REPORT_ROOT/shac/pr1535/fine_tune_gaits_shac_v3.py" \
  --checkpoint "$V3/ant_shac_source_checkpoint.pt" \
  --checkpoint-policy best --output /tmp/ant_final_shac.json \
  --seed 997 --worlds 8 --selection-worlds 256 --horizon 1 --epochs 1 \
  --warmup-steps 32 --selection-steps 400 --holdout-steps 400 \
  --holdout-repeats 3 --direction-check-epsilon 0.0001 \
  --line-search-steps 0 0.00001 -0.00001 0.00003 -0.00003 0.0001 -0.0001

"$PYTHON" "$REPORT_ROOT/shac/pr1535/fine_tune_gaits_shac_v3.py" \
  --checkpoint "$V3/humanoid_shac_source_checkpoint.pt" \
  --checkpoint-policy best --output /tmp/humanoid_final_shac.json \
  --seed 993 --worlds 8 --selection-worlds 256 --horizon 1 --epochs 1 \
  --warmup-steps 32 --selection-steps 600 --holdout-steps 600 \
  --holdout-repeats 3 --direction-check-epsilon 0.0001 \
  --line-search-steps 0 0.00001 -0.00001 0.00003 -0.00003 0.0001 -0.0001
```

Render the promoted checkpoints. `render_gaits_v3.py` keeps the exact
control-rate qpos/qvel/action trace sent to ViewerGL, recomputes the complete
gait gate on that trace, and refuses to write a manifest if it fails. It also
runs a separate nominal sibling evaluation to expose long-horizon contact
sensitivity.

```bash
PYOPENGL_PLATFORM=egl PYGLET_HEADLESS=1 "$PYTHON" \
  "$REPORT_ROOT/shac/pr1535/render_gaits_v3.py" \
  --checkpoint "$V3/ant_checkpoint.pt" --policy best --steps 400 \
  --width 960 --height 540 --fps 50 --camera-mode track \
  --camera-offset -1.8 -2.7 1.2 --camera-target-height .42 --camera-fov 36 \
  --output /tmp/ant_viewergl.mp4 --overwrite

PYOPENGL_PLATFORM=egl PYGLET_HEADLESS=1 "$PYTHON" \
  "$REPORT_ROOT/shac/pr1535/render_gaits_v3.py" \
  --checkpoint "$V3/humanoid_checkpoint.pt" --policy best --steps 600 \
  --width 960 --height 540 --fps 50 --camera-mode track \
  --camera-offset -1.8 -2.8 1.3 --camera-target-height .9 --camera-fov 35 \
  --output /tmp/humanoid_viewergl.mp4 --overwrite
```

GPU contact execution is not claimed to be bitwise deterministic, so a
re-run need not reproduce trajectory hashes. The large ensemble and exact
recorded-trace gates, rather than a single bit pattern, define success.

## What this does—and does not—test

`mjwarp_torch_bridge.py` is necessary because PR #1535 records its analytic
adjoint only for the out-of-place three-argument `step` call. At the pinned
Newton commit, `SolverMuJoCo` still calls the in-place
`mujoco_warp.step(model, data)` form. Consequently, `--newton-root` is used for
provenance and the environment is Newton's, but the physics step is called
directly through MJWarp. These runs do **not** demonstrate gradients through
Newton's public solver adapter.

The harness calls `mujoco_warp.enable_grad()` before constructing the bridge.
The bridge calls `put_model`, replays one out-of-place step during PyTorch
backward, and seeds the `qpos` and `qvel` output adjoints. Its intentionally
narrow contract matters:

- The standalone one-step gate is a local API test, not the first step of a
  training rollout: it uses 20 solver iterations and settles Humanoid for 10
  CPU MuJoCo steps. Training and multi-step checks retain the models' default
  100 iterations, and Humanoid starts at `qpos0`.
- Only `qpos`, `qvel`, and `ctrl` are differentiated. Solver
  `qacc_warmstart` is detached/reset at every step, and other `MjData` state is
  not carried through the PyTorch graph.
- Both test models have stateless motor actuators (`na == 0`); the harness
  rejects models with actuator state.
- Model setup forces Euler integration, MuJoCo's Newton solver, and an
  elliptic friction cone, disables Euler damping, and verifies that the
  harness did not alter `geom_solimp`.
- Resets and healthy-state masks are non-differentiable rollout boundaries.
- Contact and closest-feature selection are piecewise smooth. A fixed local
  contact witness can yield a useful gradient while finite differences across
  a contact transition disagree, especially over longer horizons. Equal
  contact counts alone do not prove identical contact identity or feature.
- PR #1535 explicitly rejects connect/weld equality constraints, tendons, and
  flex. Its smooth reverse also rejects fluid forces, actuator-routed gravity
  compensation, non-joint transmissions, unsupported actuator gain/bias
  types, and actuators on free/ball joints. Consult the pinned PR source before
  substituting another model.
- Fixed seeds make the experiment repeatable, but GPU contact atomics are not
  claimed to be bitwise deterministic.

Finally, `train_shac.py` is a compact **SHAC-style** experiment, not a canonical
reproduction of the full SHAC implementation. It retains the defining short
differentiable actor rollout, differentiable terminal target-critic value,
TD(lambda) critic targets, and Polyak target update, while using the simplified
state bridge and evaluation protocol described above. Report the transient and
holdout results as evidence about this PR/harness combination, not as a new
SHAC benchmark score.

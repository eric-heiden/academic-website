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
| PR lock | 3.10.0.1 | 3.10.0 | 1.14.0 | 2.3.4 | not needed |
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

## What this does—and does not—test

`mjwarp_torch_bridge.py` is necessary because PR #1535 records its analytic
adjoint only for the out-of-place three-argument `step` call. At the pinned
Newton commit, `SolverMuJoCo` still calls the in-place
`mujoco_warp.step(model, data)` form. Consequently, `--newton-root` is used for
provenance and the environment is Newton's, but the physics step is called
directly through MJWarp. These runs do **not** demonstrate gradients through
Newton's public solver adapter.

The bridge calls `mujoco_warp.enable_grad()` before `put_model`, replays one
out-of-place step during PyTorch backward, and seeds the `qpos` and `qvel`
output adjoints. Its intentionally narrow contract matters:

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

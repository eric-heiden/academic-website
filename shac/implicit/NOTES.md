# SHAC with implicit integration on MJWarp PR #1535 (head 357a75d)

Study directory for the updated SHAC report (reports.eric-heiden.com/shac/).

## Setup
- MJWarp PR #1535 head `357a75d` worktree: `/home/horde/repos/mujoco_warp-pr1535-357a75d`
  (earlier report used `02d09b1` in `/home/horde/repos/mujoco_warp-pr1535`).
- Python env: `/home/horde/repos/mujoco_warp-pr1535-357a75d/.venv` (uv sync from the PR lock:
  warp-lang 1.15.0, mujoco 3.12.1.dev, plus torch 2.11.0+cu128).
- Menagerie sparse clone (Go1, H1, ...): `third_party/mujoco_menagerie` @ 2cd6b7b.
- GPU: 1x RTX PRO 6000 Blackwell; 4 CPU cores.

## Code
- `diffsim.py` graph-captured forward + analytic backward per physics step (after
  `contrib/diffsim/_rollout.py`), PyTorch autograd per control step, K substeps.
  Masks control gradient for force-saturated actuators (PR's `_smooth_ctrl_vjp` ignores forcerange).
- `locomotion.py` SHAC-reference rewards/termination/reset, z-up.
- `shac.py` SHAC per DiffRL reference.
- `integrator_sweep.py` (E1), `policy_gradcheck.py` (E3), `smoke.py`.

## Findings so far
- Head supports EULER and IMPLICITFAST in step_backward (tests removed in refactor).
- For robots whose only velocity-dependent force is joint damping (Ant, Go1, Humanoid, H1),
  implicitfast == Euler with implicit damping (MuJoCo default eulerdamp) bit-for-bit in E1.
  The earlier study disabled eulerdamp (explicit damping) -> this is the unstable config:
  Go1 diverges 100% at 20 ms, Humanoid 100% at 10 ms. go1_kv variant (damping moved into
  actuator kv) isolates the regime where only implicitfast is implicit.
- Per-world FD: 73-89% of worlds match to <5% over 5 control steps; rest are nonsmooth events.
- Policy-level gradients (random actor, Ant, 10 ms): match FD at T<=4 steps; at T=16 FD
  disagrees and half-batch cosine ~0.04; T=64 |g|~2e5 (explosion). Free-floating w/o limits
  matches to T=16; with joint limits, analytic misses limit-activation jump contributions.
  => SHAC horizon must be short; T=32 pilot stalls at ~0.05 m/s.

## Plan
E1 integrator sweep -> E2 throughput -> E3 gradient quality vs dt/horizon -> E4 SHAC training
(small-dt baseline vs large-dt implicit; evaluation in a common fine-step simulator) -> report
(Aperture template, template-gallery/) -> publish gh-pages.

## Gradient quality vs timestep (random actor, policy_gradcheck)
- Humanoid: informative horizon ~16 steps at 2 ms (explicit or implicit, cos 0.70-0.73),
  ~8 at 10 ms, <8 at 20 ms. Seed 1 confirms (T8: cos .89 @2ms vs .46 @10ms).
- Softer contacts+limits at 10 ms (timeconst .05/.1) do NOT restore T16 (cos .09/.15).
- 4x solver iterations at 10 ms: no change -> not convergence.
- Hypothesis (unconfirmed): per-step event discontinuities scale with dt.

## Explicit-damping stability bound (h_crit = 2 I_eff / b, I_eff = 1/(M^-1)_ii at nominal pose)
ant 2010 ms; go1 14.7 ms (4 calf/knee dofs <20 ms); go1_kv 14.7 ms (but force clamp bounds it:
no divergence, 15.7 deg error @20 ms); humanoid 10.1 ms; h1 209 ms. Matches E1 divergence exactly.
Humanoid 20 ms: ~1.9% divergence even with implicit damping (random flailing protocol).
- H1 20 ms: ~70% divergence under ALL integrators (not damping; stiff contact/actuation). 10 ms ok.
- Humanoid SHAC (2 ms, T16, 500 ep): forward-dive local optimum (rigid plank falls forward), survival 3%.
  Standing diagnostic (velocity_weight=0) running to test whether balance is learnable.
- Ant explicit 2 ms s0 (1000 ep): ref-sim 6.17 m/s, survival 94%, lateral 8.6 m / 20 s.

## Humanoid investigation (all standing-task variants fail within 300 epochs, survival ~1.0-1.2 s)
- zero action falls in 33-41 steps; joint PD cannot stand (MuJoCo humanoid and H1).
- pilots: 10ms T8 (700 ep) flat; 2ms T16 (500 ep) forward dive; humanoid_ref (1.75x gear) dives faster.
- standing (velocity_weight=0): S0 base, S1 lr 5e-4, S2 fast critic, S3 low reset noise,
  S4/S5 soft contact tc .05/.1, S6 T32, S7 T64 (too slow), S8 up 1.0 + angvel .05 (slow creep 45->55),
  S9/S10 termination at 0.4 m: none stands (eval with standard 0.74 m criterion: 0.96-1.1 s).
- pyramidal vs elliptic cone: same gradient quality.
- open-loop trajopt (2ms, 64 worlds, full 100-step BPTT): 22% of worlds stand the full 2 s after
  130 Adam iterations (survival 0.72 -> ~1.35 s); TBPTT-16 slower. => long-horizon gradient info exists.
- Hypothesis: early termination is a discrete event, analytic gradient cannot see cost of falling;
  saturated height reward gives no balance gradient while upright; critic must carry it.
- Next: long runs HA (ref reward 2ms T16 512 env 2000 ep), HB (shaped 5ms T16 512 env 2000 ep).
- Critic informative: Spearman(V, remaining lifetime) 0.74 (HA) / 0.99 (S2).
- Per-world state-adjoint clipping (3x median) keeps half-batch cosine 0.65-0.70 at T=32/64 (vs ~0-0.45).
  But S11 (T32 clip) and S12 (BPTT 50-step episodes, clip, logstd -3) do not learn standing.
- H1-PD (position servos kp 40-300, kv 2-6, torque limits = motor ranges): zero action falls in 1.3 s
  (vs 0.5 s torque). No constant pose stands: ankle kp 40 N m/rad << m g l ~ 500 -> must balance
  actively / step. Open-loop stepping falls faster (0.7-1.2 s).
- Argument: stepping = contact-mode change; IFT gradient only sees current contact set.
- Final attempts: H1PD gait clock (0.8 s, weight 2, speed cap 1.0, up 1.0), 5ms T16 512 env 1000 ep;
  MuJoCo humanoid soft contact (timeconst 0.1) ref reward 5ms T16 512 env 1000 ep.
- Final attempts: H1PD gait clock plateau 1.32 s episodes (epochs 100-240, idle baseline 1.3 s);
  humanoid soft contact (tc 0.1) 0.55 s at 240 epochs. Stopped. Humanoid = documented negative result.

## E4 results (ref sim = implicitfast 1 ms; 3 seeds; 1000 epochs, 256 envs, T=8)
Ant ref speed: explicit2 6.26, impl2 6.37, impl5 5.18, impl10 2.72, impl20 3.47 (m/s); survival ref >=0.89.
Go1 ref speed: explicit2 1.58, impl2 1.56, impl5 ~1.56, impl10 1.56 (all transfer), impl20 0.84+-0.55 (train 1.55) -> 20 ms policies
exploit integration artifacts (static crouch in ref sim). E5 time-matched 10 ms 4000 epochs running.

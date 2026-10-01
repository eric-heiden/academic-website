"""SHAC trainer for the research loop (configurable, synchronisation-free rollout).

Same algorithm as ``shac.py`` (Xu et al. 2022): the actor minimises the negative
discounted return of a short window back-propagated through the analytic
simulator adjoint plus a bootstrapped terminal value; the critic regresses
TD(lambda) targets with a Polyak-averaged target network. Differences:

* environments come from ``tasks.py`` (configurable Isaac-Lab-style terms);
* the rollout, backward, and actor update never synchronise with the host;
  statistics are read once per epoch;
* the critic update can be captured as a single CUDA graph;
* optional extensions (default off) for the research loop are marked "ext".

Usage: python shac2.py --task go1 --out runs/x/name [--set key=value ...]
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from diffsim import SimConfig
from robots import ROBOTS
from tasks import TASKS, Env, make_task


@dataclasses.dataclass
class Config:
    task: str = "go1"
    # simulator
    integrator: str = "implicitfast"
    dt: float = 0.002
    control_dt: float = 0.02
    iterations: int = 100
    ls_iterations: int = 50
    cone: str = "elliptic"
    solref_timeconst: float | None = None
    geom_solimp: tuple | None = None
    geom_margin: float | None = None
    model_edit: str | None = None
    # Backward-only contact softening (needs stored_forward=False): solimp / margin used by the backward model.
    grad_solimp: tuple | None = None
    grad_margin: float | None = None
    # Contact / constraint buffer sizes per world (None: robot defaults); overflow is logged per epoch.
    nconmax: int | None = None
    njmax: int | None = None
    # SHAC
    num_envs: int = 256
    horizon: int = 8
    epochs: int = 1000
    gamma: float = 0.99
    lam: float = 0.95
    actor_lr: float = 2e-3
    critic_lr: float = 2e-3
    lr_schedule: str = "linear"  # linear | constant
    lr_final: float = 1e-5
    betas: tuple = (0.7, 0.95)
    grad_norm: float = 1.0
    critic_iterations: int = 16
    critic_batches: int = 4
    target_alpha: float = 0.2
    actor_units: tuple = (256, 128)
    critic_units: tuple = (128, 128)
    layer_norm: bool = True
    logstd_init: float = -1.0
    critic_graph: bool = True
    # Keep every physics step's forward data of the window resident so the backward does not re-run
    # the forward (identical gradients; more memory).
    stored_forward: bool = True
    # ext: per-world cap on the state adjoint norm passed between control steps (x median).
    state_grad_clip: float | None = None
    # ext: per-world cap on each control step's control-gradient norm (x median over worlds).
    ctrl_grad_clip: float | None = None
    # ext: bundled contact gradients (bcg.py, after arXiv 2609.30951): each env is simulated as bcg_branches branches,
    # perturbed in joint space when the previous step's vertical contact load exceeded bcg_tau_mg * m g (or always /
    # never), and averaged after every control step. 1 = off. Evaluation scripts always build unbundled envs.
    bcg_branches: int = 1
    bcg_trigger: str = "force"
    bcg_tau_mg: float = 1.2
    bcg_sigma_q: float = 0.02
    bcg_sigma_qd: float = 0.05
    bcg_antithetic: bool = True
    # ext: warm start (actor, critic, target critic and observation statistics) from a checkpoint, e.g. for a
    # command curriculum; the optimiser state starts fresh.
    init_from: str | None = None
    # ext: entropy bonus weight on the Gaussian policy (SAPO-style max-entropy objective).
    entropy_coef: float = 0.0
    # ext: SAPO-style maximum-entropy return: reward + alpha * (-log pi(a|s)) of the tanh-squashed policy,
    # differentiable through the reparameterised sample; alpha is tuned towards target_entropy * act_dim
    # when ent_alpha_lr > 0 (0 = fixed alpha). ent_alpha = 0 disables it.
    ent_alpha: float = 0.0
    ent_alpha_lr: float = 0.0
    target_entropy: float = -0.5
    # ext: RPO-style reuse (Zhong et al. 2025): after the SHAC step, reuse the cached per-sample action
    # gradients dL/du for reuse_epochs extra policy-only updates. Each update re-parameterises the stored
    # pre-tanh actions under the new policy (noise recovered so the action is unchanged), weights each sample by
    # the importance ratio (zeroed outside [rpo_clip_lo, rpo_clip_hi]) and adds rpo_kl * KL(old || new).
    reuse_epochs: int = 0
    rpo_clip_lo: float = 0.2
    rpo_clip_hi: float = 2.0
    rpo_kl: float = 0.5
    # ext: lambda-return actor objective: each window segment contributes
    # sum_k w_k [R_{0:k} + gamma^{k+1} V(s_{k+1})] with w_k = (1 - l) l^k and the remaining weight on the
    # segment's last step (actor_lambda = None: SHAC's terminal-only bootstrap).
    actor_lambda: float | None = None
    # ext: inverse-variance-weighted hybrid gradient (IVW-H): per control step and action dimension, the
    # per-sample analytic gradient dL/du and a likelihood-ratio gradient on TD(lambda) advantages are
    # mixed with weights inversely proportional to their variances over worlds.
    ivw: bool = False
    ivw_eps: float = 1e-12
    # ext: weight of a likelihood-ratio (score-function) policy-gradient term on normalised
    # TD(lambda) advantages, added to the analytic SHAC loss (0 = pure SHAC).
    lr_coef: float = 0.0
    # misc
    seed: int = 0
    time_budget: float | None = None
    save_every: int = 50
    log_every: int = 20
    task_overrides: dict = dataclasses.field(default_factory=dict)


PRESETS = {
    "ant": dict(actor_units=(128, 64, 32), critic_units=(64, 64), actor_lr=2e-3, critic_lr=2e-3, target_alpha=0.2),
    "go1": dict(actor_units=(256, 128), critic_units=(128, 128), actor_lr=2e-3, critic_lr=2e-3, target_alpha=0.2),
    "humanoid": dict(actor_units=(256, 128), critic_units=(128, 128), actor_lr=2e-3, critic_lr=5e-4,
                     target_alpha=0.995),
}
for _k in ("humanoid_clean", "humanoid_surv", "humanoid_gait"):
    PRESETS[_k] = PRESETS["humanoid"]
# Humanoid velocity tasks: a faster target critic (alpha 0.9 instead of DiffRL's 0.995) learned 2-3x faster on
# H1 and was required for G1 balance (research/loop.md, iterations 7-9).
PRESETS["h1_vel"] = dict(actor_units=(256, 128), critic_units=(128, 128), actor_lr=2e-3, critic_lr=5e-4,
                         target_alpha=0.9)
PRESETS["go1_vel"] = PRESETS["go1_vel_fwd"] = PRESETS["go1_gait"] = PRESETS["go1"]
PRESETS["g1_vel"] = PRESETS["h1_gait"] = PRESETS["g1_gait"] = PRESETS["g1_gait_strong"] = PRESETS["h1_vel"]

# ----------------------------------------------------------------------------- networks


def mlp(sizes, out, layer_norm=True):
    layers, d = [], sizes[0]
    for h in sizes[1:]:
        layers += [nn.Linear(d, h), nn.ELU()]
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, units, logstd_init=-1.0, layer_norm=True):
        super().__init__()
        self.mu = mlp([obs_dim, *units], act_dim, layer_norm)
        self.logstd = nn.Parameter(torch.full((act_dim,), float(logstd_init)))

    def forward(self, obs, deterministic=False):
        mu = self.mu(obs)
        if deterministic:
            return mu
        return mu + torch.exp(self.logstd) * torch.randn_like(mu)


class Critic(nn.Module):
    def __init__(self, obs_dim, units, layer_norm=True):
        super().__init__()
        self.v = mlp([obs_dim, *units], 1, layer_norm)
        for m in self.v.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(m.bias)

    def forward(self, obs):
        return self.v(obs).squeeze(-1)


class RunningMeanStd:
    def __init__(self, shape, device):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = torch.full((), 1e-4, device=device)

    def update(self, x):
        x = x.detach()
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + delta * bc / tot
        m2 = self.var * self.count + bv * bc + delta.square() * self.count * bc / tot
        self.var = m2 / tot
        self.count = tot

    def normalize(self, x):
        return (x - self.mean) / torch.sqrt(self.var + 1e-5)

    def snapshot(self):
        s = RunningMeanStd.__new__(RunningMeanStd)
        s.mean, s.var, s.count = self.mean.clone(), self.var.clone(), self.count.clone()
        return s

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": float(self.count)}


# ----------------------------------------------------------------------------- critic update


class CriticUpdate:
    """TD(lambda) regression with minibatch Adam; optionally one CUDA graph for all iterations."""

    def __init__(self, critic, cfg: Config, n, obs_dim, dev):
        self.critic, self.cfg = critic, cfg
        self.graph = None
        self.obs = torch.zeros(n, obs_dim, device=dev)
        self.tgt = torch.zeros(n, device=dev)
        self.loss = torch.zeros((), device=dev)
        self.use_graph = cfg.critic_graph
        if self.use_graph:
            self.lr = torch.tensor(cfg.critic_lr, device=dev)
            self.opt = torch.optim.Adam(critic.parameters(), lr=self.lr, betas=tuple(cfg.betas), capturable=True,
                                        fused=True)
        else:
            self.opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr, betas=tuple(cfg.betas), fused=True)

    def set_lr(self, lr):
        if self.use_graph:
            self.lr.fill_(lr)
        else:
            for g in self.opt.param_groups:
                g["lr"] = lr

    def _run(self):
        c = self.cfg
        n = self.obs.shape[0]
        bs = n // c.critic_batches
        total = torch.zeros((), device=self.obs.device)
        for _ in range(c.critic_iterations):
            for b in range(c.critic_batches):
                sl = slice(b * bs, (b + 1) * bs)
                loss = (self.critic(self.obs[sl]) - self.tgt[sl]).square().mean()
                self.opt.zero_grad(set_to_none=not self.use_graph)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), c.grad_norm)
                self.opt.step()
                total = loss.detach()
        return total

    def _capture(self):
        params = [p.detach().clone() for p in self.critic.parameters()]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self._run()
        torch.cuda.current_stream().wait_stream(s)
        # Undo the warm-up updates (parameters and optimiser state).
        with torch.no_grad():
            for p, p0 in zip(self.critic.parameters(), params):
                p.copy_(p0)
            for st in self.opt.state.values():
                for k, v in st.items():
                    if torch.is_tensor(v):
                        v.zero_()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.loss = self._run()

    def __call__(self, obs, tgt):
        # Targets are sanitised once here (the only possible source of non-finite critic gradients).
        self.obs.copy_(obs)
        self.tgt.copy_(torch.nan_to_num(tgt, nan=0.0, posinf=0.0, neginf=0.0))
        if not self.use_graph:
            self.loss = self._run()
            return self.loss
        if self.graph is None:
            self._capture()
        self.graph.replay()
        return self.loss


# ----------------------------------------------------------------------------- training


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="Config overrides key=value (JSON values)")
    ap.add_argument("--tset", nargs="*", default=[], help="TaskCfg overrides key=value (JSON values)")
    a = ap.parse_args()
    cfg = Config(task=a.task)
    robot = TASKS[a.task].robot
    for k, v in PRESETS.get(a.task, PRESETS.get(robot, {})).items():
        setattr(cfg, k, v)
    for kv in a.set:
        k, v = kv.split("=", 1)
        assert hasattr(cfg, k), f"unknown config key {k}"
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        setattr(cfg, k, tuple(v) if isinstance(v, list) else v)
    for kv in a.tset:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        cfg.task_overrides[k] = tuple(tuple(x) if isinstance(x, list) else x for x in v) if isinstance(v, list) else v
    return cfg, Path(a.out)


def build_env(cfg: Config, num_envs=None, seed=None, backward=True, bundle=False, **sim_over):
    task = make_task(cfg.task, **cfg.task_overrides)
    r = ROBOTS[task.robot]
    substeps = round(cfg.control_dt / cfg.dt)
    assert abs(substeps * cfg.dt - cfg.control_dt) < 1e-9
    sc = dict(xml=r.xml, integrator=cfg.integrator, timestep=cfg.dt, substeps=substeps, keyframe=r.keyframe,
              nconmax=cfg.nconmax or r.nconmax, njmax=cfg.njmax or r.njmax, iterations=cfg.iterations, ls_iterations=cfg.ls_iterations,
              cone=cfg.cone, solref_timeconst=cfg.solref_timeconst,
              geom_solimp=tuple(cfg.geom_solimp) if cfg.geom_solimp else None, geom_margin=cfg.geom_margin,
              model_edit=cfg.model_edit or r.model_edit,
              grad_solimp=tuple(cfg.grad_solimp) if cfg.grad_solimp else None, grad_margin=cfg.grad_margin)
    sc.update(sim_over)
    bcg = None
    if bundle and cfg.bcg_branches > 1:
        from bcg import BCGConfig
        bcg = BCGConfig(branches=cfg.bcg_branches, trigger=cfg.bcg_trigger, tau_mg=cfg.bcg_tau_mg,
                        sigma_q=cfg.bcg_sigma_q, sigma_qd=cfg.bcg_sigma_qd, antithetic=cfg.bcg_antithetic,
                        seed=cfg.seed + 1000003)
    env = Env(task, SimConfig(**sc), num_envs or cfg.num_envs, seed=cfg.seed if seed is None else seed,
              backward=backward, bcg=bcg)
    env.sim.state_grad_clip = cfg.state_grad_clip
    env.sim.ctrl_grad_clip = cfg.ctrl_grad_clip
    return env


def train(cfg: Config, out: Path):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    env = build_env(cfg, bundle=True)
    dev = env.dev
    obs_dim, act_dim = env.num_obs, env.nu
    actor = Actor(obs_dim, act_dim, cfg.actor_units, cfg.logstd_init, cfg.layer_norm).to(dev)
    critic = Critic(obs_dim, cfg.critic_units, cfg.layer_norm).to(dev)
    target = copy.deepcopy(critic)
    for prm in target.parameters():
        prm.requires_grad_(False)
    a_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr, betas=tuple(cfg.betas), fused=True)
    T, N, gamma, lam = cfg.horizon, cfg.num_envs, cfg.gamma, cfg.lam
    c_update = CriticUpdate(critic, cfg, T * N, obs_dim, dev)
    if cfg.stored_forward:
        env.sim.enable_window(T)
    log_alpha = torch.tensor(math.log(cfg.ent_alpha) if cfg.ent_alpha > 0 else 0.0, device=dev, requires_grad=True)
    alpha_opt = torch.optim.Adam([log_alpha], lr=cfg.ent_alpha_lr) if cfg.ent_alpha_lr > 0 else None
    logp_sum = torch.zeros((), device=dev)
    obs_rms = RunningMeanStd(obs_dim, dev)
    if cfg.init_from:
        ck = torch.load(cfg.init_from, map_location=dev, weights_only=False)
        actor.load_state_dict(ck["actor"])
        if "critic" in ck:
            critic.load_state_dict(ck["critic"])
            target.load_state_dict(ck["critic"])
        r = ck["obs_rms"]
        obs_rms.mean.copy_(r["mean"]), obs_rms.var.copy_(r["var"])
        obs_rms.count = torch.full((), float(r["count"]), device=dev)

    obs_buf = torch.zeros(T, N, obs_dim, device=dev)
    rew_buf = torch.zeros(T, N, device=dev)
    done_buf = torch.zeros(T, N, device=dev)
    nv_buf = torch.zeros(T, N, device=dev)
    u_buf = torch.zeros(T, N, act_dim, device=dev)
    eps_buf = torch.zeros(T, N, act_dim, device=dev)
    gk_buf = torch.zeros(T, N, device=dev)
    ivw_alpha = torch.zeros(T, device=dev)
    skipped = torch.zeros((), device=dev)

    history = []
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"config": dataclasses.asdict(cfg), "task": dataclasses.asdict(env.task), "obs_dim": obs_dim,
            "act_dim": act_dim}
    t_start = time.time()
    torch.cuda.synchronize()

    def save(epoch):
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "obs_rms": obs_rms.state_dict(),
                    "meta": meta, "epoch": epoch}, out.with_suffix(".pt"))
        out.with_suffix(".json").write_text(json.dumps({**meta, "history": history,
                                                        "total_wall_s": time.time() - t_start}))

    for epoch in range(cfg.epochs):
        t0 = time.time()
        frac = epoch / cfg.epochs
        if cfg.lr_schedule == "linear":
            a_lr = cfg.actor_lr + (cfg.lr_final - cfg.actor_lr) * frac
            c_lr = cfg.critic_lr + (cfg.lr_final - cfg.critic_lr) * frac
        else:
            a_lr, c_lr = cfg.actor_lr, cfg.critic_lr
        for g in a_opt.param_groups:
            g["lr"] = a_lr
        c_update.set_lr(c_lr)

        # ------------------------------------------------------------ actor rollout
        env.detach()
        if cfg.stored_forward:
            env.sim.begin_window()
        rms = obs_rms.snapshot()
        obs = env.obs()
        obs_rms.update(obs)
        rew_acc = torch.zeros(N, device=dev)
        gamma_k = torch.ones(N, device=dev)
        lam_k = torch.ones(N, device=dev)
        loss = torch.zeros((), device=dev)
        u_list = []
        for i in range(T):
            nobs = rms.normalize(obs)
            obs_buf[i] = nobs.detach()
            mu = actor.mu(nobs)
            eps = torch.randn_like(mu)
            u = mu + torch.exp(actor.logstd) * eps
            u_buf[i] = u.detach()
            eps_buf[i] = eps
            gk_buf[i] = gamma_k
            if cfg.ivw or cfg.reuse_epochs:
                u_list.append(u)
            action = torch.tanh(u)
            obs, rew, done, info = env.step(action)
            if cfg.ent_alpha > 0:
                # log-density of the squashed sample; its negative is a one-sample entropy estimate.
                logp = (-0.5 * eps.square() - actor.logstd - 0.5 * math.log(2 * math.pi)).sum(-1) \
                    - torch.log(1.0 - action.square() + 1e-6).sum(-1)
                logp_sum = logp_sum + logp.detach().mean()
                rew = rew - log_alpha.detach().exp() * logp
            obs_rms.update(obs)
            term, trunc = info["terminated"], info["truncated"]
            # One target evaluation: truncated worlds bootstrap from their pre-reset observation.
            next_v = target(rms.normalize(torch.where(trunc[:, None], info["obs_before_reset"], obs)))
            next_v = torch.where(term, 0.0, next_v)
            rew_acc = rew_acc + gamma_k * rew
            contrib = -(rew_acc + gamma * gamma_k * next_v)
            if cfg.actor_lambda is not None:
                la = cfg.actor_lambda
                final = done | (i == T - 1)
                w = torch.where(final, lam_k, (1.0 - la) * lam_k)
                loss = loss + (w * contrib).sum()
                lam_k = torch.where(done, 1.0, lam_k * la)
            elif i == T - 1:
                loss = loss + contrib.sum()
            else:
                loss = loss + torch.where(done, contrib, 0.0).sum()
            gamma_k = torch.where(done, 1.0, gamma_k * gamma)
            rew_acc = torch.where(done, 0.0, rew_acc)
            rew_buf[i] = rew.detach()
            done_buf[i] = 1.0 if i == T - 1 else done.float()
            nv_buf[i] = next_v.detach()
        loss = loss / (T * N)
        if cfg.entropy_coef:
            # Gaussian entropy per action dimension (pre-tanh); ext.
            loss = loss - cfg.entropy_coef * actor.logstd.sum()
        t_roll = time.time()

        # TD(lambda) targets (needed by the critic and by the optional likelihood-ratio terms).
        with torch.no_grad():
            target_vals = torch.zeros(T, N, device=dev)
            Ai = torch.zeros(N, device=dev)
            Bi = torch.zeros(N, device=dev)
            lam_i = torch.ones(N, device=dev)
            for i in reversed(range(T)):
                d = done_buf[i]
                lam_i = lam_i * lam * (1.0 - d) + d
                Ai = (1.0 - d) * (lam * gamma * Ai + gamma * nv_buf[i] + (1.0 - lam_i) / (1.0 - lam) * rew_buf[i])
                Bi = gamma * (nv_buf[i] * d + Bi * (1.0 - d)) + rew_buf[i]
                target_vals[i] = (1.0 - lam) * Ai + lam_i * Bi
        lr_loss = torch.zeros((), device=dev)
        if cfg.lr_coef:
            # ext: score-function gradient on detached states and sampled pre-tanh actions.
            with torch.no_grad():
                adv = target_vals - critic(obs_buf.reshape(T * N, obs_dim)).reshape(T, N)
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            mu_b = actor.mu(obs_buf.reshape(T * N, obs_dim)).reshape(T, N, act_dim)
            logp = (-0.5 * ((u_buf - mu_b) / torch.exp(actor.logstd)).square() - actor.logstd).sum(-1)
            lr_loss = -(adv * logp).mean()
            loss = loss + cfg.lr_coef * lr_loss

        a_opt.zero_grad(set_to_none=True)
        g_cache = None
        if cfg.reuse_epochs and not cfg.ivw:
            g1 = torch.stack(torch.autograd.grad(loss, u_list))
            g_cache = g1
            std0 = torch.exp(actor.logstd).detach()
            u_re = actor.mu(obs_buf.reshape(T * N, obs_dim)).reshape(T, N, act_dim) + std0 * eps_buf
            u_re = u_re + (torch.exp(actor.logstd) - std0) * eps_buf
            torch.autograd.backward(u_re, g1)
        elif cfg.ivw:
            # Per-sample analytic gradients w.r.t. the pre-tanh actions (runs the simulator backward).
            g1 = torch.stack(torch.autograd.grad(loss, u_list))
            with torch.no_grad():
                std = torch.exp(actor.logstd)
                adv = target_vals - critic(obs_buf.reshape(T * N, obs_dim)).reshape(T, N)
                # Likelihood-ratio gradient of the same objective w.r.t. mu: -gamma^k A (u - mu) / sigma^2 / (T N).
                g0 = -(gk_buf * adv)[..., None] * (eps_buf / std) / (T * N)
                v1 = g1.var(1, keepdim=True)
                v0 = g0.var(1, keepdim=True)
                alpha = v0 / (v0 + v1 + cfg.ivw_eps)
                G = alpha * g1 + (1.0 - alpha) * g0
                ivw_alpha = alpha.mean((1, 2))
            u_re = actor.mu(obs_buf.reshape(T * N, obs_dim)).reshape(T, N, act_dim) + std.detach() * eps_buf
            u_re = u_re + (torch.exp(actor.logstd) - std.detach()) * eps_buf
            torch.autograd.backward(u_re, G)
        else:
            loss.backward()
        t_bwd = time.time()
        gnorm = torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.grad_norm)
        ok = torch.isfinite(gnorm) & (gnorm < 1e6)
        skipped += (~ok).float()
        if bool(ok):  # one host synchronisation per epoch; a non-finite or exploding gradient skips the step
            a_opt.step()
            if g_cache is not None and cfg.reuse_epochs:
                # ext: RPO reuse of the cached action gradients (policy-only updates, no simulator work).
                with torch.no_grad():
                    flat_obs = obs_buf.reshape(T * N, obs_dim)
                    mu_old = mu_old_all = None
                    old_std = std0
                    mu_old = (u_buf - std0 * eps_buf)  # mean of the rollout policy
                    logp_old = (-0.5 * eps_buf.square() - torch.log(old_std)).sum(-1)
                for _ in range(cfg.reuse_epochs):
                    mu_new = actor.mu(flat_obs).reshape(T, N, act_dim)
                    std_new = torch.exp(actor.logstd)
                    eps_new = ((u_buf - mu_new) / std_new).detach()
                    u_new = mu_new + std_new * eps_new  # equals u_buf; gradient through the new policy
                    logp_new = (-0.5 * eps_new.square() - torch.log(std_new)).sum(-1)
                    ratio = torch.exp((logp_new - logp_old).detach())
                    w = torch.where((ratio > cfg.rpo_clip_lo) & (ratio < cfg.rpo_clip_hi), ratio, 0.0)
                    surrogate = (w[..., None] * g_cache * u_new).sum()
                    kl = (torch.log(std_new / old_std) + (old_std.square() + (mu_old - mu_new).square())
                          / (2 * std_new.square()) - 0.5).sum(-1).mean()
                    a_opt.zero_grad(set_to_none=True)
                    (surrogate + cfg.rpo_kl * kl).backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.grad_norm)
                    a_opt.step()
        if alpha_opt is not None:
            alpha_loss = -log_alpha * (logp_sum / T + cfg.target_entropy * act_dim).detach()
            alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            alpha_opt.step()
        logp_mean = logp_sum / T
        logp_sum = torch.zeros((), device=dev)

        # ------------------------------------------------------------ critic
        c_loss = c_update(obs_buf.reshape(T * N, obs_dim), target_vals.reshape(T * N))
        with torch.no_grad():
            torch._foreach_lerp_(list(target.parameters()), list(critic.parameters()), 1.0 - cfg.target_alpha)

        # ------------------------------------------------------------ logging (one sync per epoch)
        scal = torch.stack((loss.detach(), gnorm.detach(), c_loss.detach(), rew_buf.mean(), skipped,
                            env.sim.nonfinite_grad_worlds.float(), env.nonfinite_resets.float(),
                            actor.logstd.detach().mean(), lr_loss.detach(), log_alpha.detach().exp(),
                            logp_mean.detach(), ivw_alpha.mean(),
                            (env.sim.t_overflow & 15).ne(0).float().mean(),
                            env.sim.state_grad_clip_worlds.float(), env.sim.ctrl_grad_clip_worlds.float(),
                            getattr(env.sim, "n_bundles", torch.zeros((), device=dev)).float(),
                            getattr(env.sim, "n_bad_branches", torch.zeros((), device=dev)).float())).cpu().tolist()
        env.sim.clear_overflow()
        t_end = time.time()
        st = env.pop_stats()
        wall = t_end - t_start
        row = {"epoch": epoch, "wall_s": wall, "epoch_s": t_end - t0, "rollout_s": t_roll - t0,
               "backward_s": t_bwd - t_roll, "rest_s": t_end - t_bwd, "env_steps": (epoch + 1) * T * N,
               "actor_loss": scal[0], "grad_norm": scal[1], "critic_loss": scal[2], "mean_reward": scal[3],
               "skipped_updates": int(scal[4]), "nonfinite_grad_worlds": int(scal[5]),
               "nonfinite_resets": int(scal[6]), "logstd": scal[7], "lr_loss": scal[8], "alpha": scal[9],
               "logp": scal[10], "ivw_alpha": scal[11], "overflow_frac": scal[12],
               "state_clip_worlds_cum": int(scal[13]), "ctrl_clip_worlds_cum": int(scal[14]),
               "bcg_bundles_cum": int(scal[15]), "bcg_bad_branches_cum": int(scal[16]),
               **{"ep_" + k: v for k, v in st.items()}}
        history.append(row)
        if epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1:
            print(f"ep {epoch:5d} t={wall:7.1f}s ({row['epoch_s']*1e3:5.0f}ms) R={st.get('return')} "
                  f"len={st.get('length')} v={st.get('speed_x')} trk={st.get('track_err')} "
                  f"r/step={scal[3]:.3f} |g|={scal[1]:.3g} skip={int(scal[4])} nanG={int(scal[5])}", flush=True)
        stop = cfg.time_budget is not None and wall > cfg.time_budget
        if epoch % cfg.save_every == 0 or epoch == cfg.epochs - 1 or stop:
            save(epoch)
            if epoch % cfg.save_every == 0:
                torch.save({"actor": actor.state_dict(), "obs_rms": obs_rms.state_dict(), "meta": meta,
                            "epoch": epoch, "wall_s": wall}, out.parent / f"{out.name}_ep{epoch:05d}.pt")
        if stop:
            break
    save(epoch)
    return history


if __name__ == "__main__":
    cfg, out = parse()
    train(cfg, out)

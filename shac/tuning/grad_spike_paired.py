import sys, torch, warp as wp
sys.path.insert(0, "/home/horde/repos/shac-implicit")
import diffsim
from shac2 import Config, build_env
task, variant, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
T = 4
cfg = Config(task=task, num_envs=64, dt=0.005, horizon=T, seed=3)
env = build_env(cfg)
sim = env.sim
sim.enable_window(T)
if variant == "sync":
    _cl = wp.capture_launch
    def cl(g, stream=None):
        _cl(g, stream=stream); wp.synchronize()
    diffsim.wp.capture_launch = cl
if variant == "refwd":
    W = diffsim._WindowControlStep
    _bw = W.backward
    def bw(ctx, gq, gv, gw):
        s, t = ctx.sim, ctx.t
        for k in range(s.cfg.substeps):  # restore the stored forward data of this control step
            wp.capture_launch(s.win_fwd[t][k], stream=s.stream)
        return _bw(ctx, gq, gv, gw)
    W.backward = staticmethod(bw)
snap = {k: v.clone() for k, v in vars(env).items() if torch.is_tensor(v)}
gstate = env.gen.get_state() if hasattr(env, "gen") else None
def restore():
    for k, v in snap.items():
        setattr(env, k, v.clone())
    if gstate is not None:
        env.gen.set_state(gstate)
def run(window):
    restore()
    sim.window = T if window else None
    if window:
        sim.begin_window()
    g = torch.Generator(device="cuda").manual_seed(0)
    acts = (0.5 * torch.randn(T, 64, env.nu, device="cuda", generator=g)).requires_grad_(True)
    q0 = env.q.detach().clone().requires_grad_(True); env.q = q0
    loss = 0
    for t in range(T):
        obs, r, d, info = env.step(torch.tanh(acts[t]))
        loss = loss + r.sum() + env.q[:, 0].sum() + env.v[:, 0].sum()
    loss.backward()
    return acts.grad.clone()
bad = 0
for i in range(N):
    r = run(False); w = run(False if variant == "rr" else True)
    per_w = (w - r).norm(dim=(0, 2)) / r.norm(dim=(0, 2)).clamp(min=1e-9)
    rel = float((w - r).norm() / r.norm())
    if rel > 0.1:
        bad += 1
        print(i, "rel %.3f" % rel, "norms r %.1f w %.1f" % (r.norm(), w.norm()), "world59 r %.1f w %.1f" % (r[:, 59].norm(), w[:, 59].norm()), "worlds", (per_w > 0.5).nonzero().flatten().tolist(), flush=True)
print(variant, "bad", bad, "of", N, flush=True)

import sys, runpy
sys.argv = ["x", sys.argv[1], "none", "0"]
ns = runpy.run_path("/home/horde/repos/shac-implicit/grad_spike_paired.py")
run = ns["run"]; import torch
N = 40
for mode in (False, True, False, True):
    n59 = []
    spikes = 0
    for i in range(N):
        g = run(mode)
        pw = g.norm(dim=(0, 2))
        med = pw.median()
        if (pw > 10 * med).any():
            spikes += 1
    print("window" if mode else "recompute", "passes with a world > 10x median:", spikes, "of", N, flush=True)

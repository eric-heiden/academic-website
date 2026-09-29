"""Builds the report's figure data (data/figures.json) from the result files."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = Path("/home/horde/repos/academic-website-reports/shac/data/figures.json")


def load(p):
    return json.loads(Path(p).read_text())


def integrator_rows():
    return load(ROOT / "results" / "e1_summary.json")


def gradient_rows():
    out = defaultdict(lambda: defaultdict(dict))
    for f in sorted((ROOT / "results" / "e3").glob("*.json")):
        stem = f.stem
        robot_cfg, seed = stem.rsplit("_s", 1)
        cut = max(robot_cfg.find("_explicit_"), robot_cfg.find("_implicit_"))
        robot, cfg = robot_cfg[:cut], robot_cfg[cut + 1:]
        for r in load(f)["rows"]:
            fd = statistics.median(r["fd_slope"].values())
            ratio = r["analytic_slope"] / fd if fd > 0 else float("nan")
            d = out[robot][cfg].setdefault(r["horizon"], {"cos": [], "ratio": [], "gnorm": []})
            d["cos"].append(r["half_batch_cosine"])
            d["ratio"].append(math.log10(ratio) if ratio > 0 else None)
            d["gnorm"].append(r["grad_norm"])
    res = {}
    for robot, cfgs in out.items():
        res[robot] = {}
        for cfg, hs in cfgs.items():
            rows = []
            for h in sorted(hs):
                d = hs[h]
                ratios = [x for x in d["ratio"] if x is not None and math.isfinite(x)]
                rows.append({"horizon": h, "cosine_mean": statistics.fmean(d["cos"]), "cosine_values": d["cos"],
                             "log10_ratio_mean": statistics.fmean(ratios) if ratios else None,
                             "grad_norm_median": statistics.median(d["gnorm"]), "n": len(d["cos"])})
            res[robot][cfg] = rows
    return res


def learning_rows(tag="e4", suffix=""):
    s = load(ROOT / "results" / f"{tag}_summary.json")
    timing = {}
    tf = ROOT / "results" / "epoch_timing.json"
    if tf.exists():
        for r in load(tf):
            if r["num_envs"] == 256:
                timing[(r["robot"], r["config"])] = r["median_epoch_s"]
    res = defaultdict(dict)
    for row in s.values():
        c = row["curve"]
        g = lambda k, f: [p[k][f] if p[k] else None for p in c]  # noqa: E731
        res[row["robot"]][row["config"] + suffix] = {
            "epoch": [p["epoch"] for p in c],
            "speed_mean": g("speed", "mean"), "speed_sd": g("speed", "sd"),
            "length_mean": [x * 0.02 if x is not None else None for x in g("length", "mean")],
            "length_sd": [x * 0.02 if x is not None else None for x in g("length", "sd")],
            "epoch_seconds": timing.get((row["robot"], row["config"])),
        }
    return res


def main():
    fig = {"integrator": integrator_rows()}
    if (ROOT / "results" / "e3").exists():
        fig["gradient"] = gradient_rows()
    if (ROOT / "results" / "e4_summary.json").exists():
        fig["learning"] = learning_rows()
    if (ROOT / "results" / "e5_summary.json").exists():
        for robot, cfgs in learning_rows("e5", "_long").items():
            for cfg, v in cfgs.items():
                # The long runs share the 10 ms per-epoch cost measured for the matrix.
                v["epoch_seconds"] = fig["learning"].get(robot, {}).get("implicit_10ms", {}).get("epoch_seconds")
                fig["learning"].setdefault(robot, {})[cfg] = v
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fig, separators=(",", ":")))
    print("wrote", OUT, OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    main()

"""Gather every experiment log into one summary file."""

from __future__ import annotations

import json
import re
from pathlib import Path

OUT = Path("/home/horde/repos/fpgs-study/out")
SUM = Path("/home/horde/repos/fpgs-study/summary.json")


def tail_json(name):
    p = OUT / name
    if not p.exists():
        return None
    txt = p.read_text(errors="ignore")
    if "===JSON===" not in txt:
        return None
    blob = txt.split("===JSON===", 1)[1].strip()
    # anything printed after the payload (shell markers, warp teardown) is ignored
    try:
        return json.JSONDecoder().raw_decode(blob)[0]
    except json.JSONDecodeError:
        return None


def line_json(name, prefixes):
    """Pull per-case JSON printed one line at a time."""
    p = OUT / name
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text(errors="ignore").splitlines():
        for pre in prefixes:
            m = re.match(rf"^({pre}\S*)\s+(\{{.*)$", line)
            if m:
                try:
                    out[m.group(1)] = json.loads(m.group(2))
                except json.JSONDecodeError:
                    pass
            m2 = re.match(rf"^({pre}\S*)\s+timing\s+([\d.]+)$", line)
            if m2 and m2.group(1) in out:
                ms = float(m2.group(2))
                out[m2.group(1)]["ms_per_frame"] = ms
                out[m2.group(1)]["physics_rtf"] = (1000.0 / 60.0) / ms
    return out


def graph_capture():
    p = OUT / "graph.log"
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text(errors="ignore").splitlines():
        m = re.match(r"^(streams=\S+,double_buffer=\S+) -> (.*)$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def kv_lines(name, keys):
    p = OUT / name
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text(errors="ignore").splitlines():
        for k in keys:
            if line.startswith(k + " "):
                rest = line[len(k) + 1:]
                if rest.startswith("{"):
                    try:
                        out[k] = json.loads(rest)
                    except json.JSONDecodeError:
                        out[k] = rest
                else:
                    out[k] = rest
    return out


def tracking():
    p = OUT / "track.log"
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text(errors="ignore").splitlines():
        m = re.match(r"^(\S+) max arm joint error ([\d.]+) deg$", line)
        if m:
            out[m.group(1)] = {"final_err_max_deg": float(m.group(2))}
    return out


if __name__ == "__main__":
    s = {}
    s["graph_capture"] = graph_capture()
    s["tracking"] = tracking()
    s["phase"] = (tail_json("phase.log") or {}).get("phase", {})
    s["configs"] = line_json("gc_all.log", ["fpgs_gc"])
    sub = line_json("substeps_v1.log", ["fpgs_ss"])
    sub.update(line_json("substeps_mj.log", ["mujoco_ss"]))
    s["substeps"] = sub
    rows = tail_json("rows.log") or {}
    s["watermarks"] = rows.get("watermarks", {})
    s["budget"] = rows.get("budget", {})
    s["repeat"] = {k: v for k, v in rows.items() if k.startswith("repeat")}
    s["speculative"] = tail_json("spec.log") or line_json("spec.log", ["base_", "velit"])
    coup = {}
    coup["static"] = (tail_json("coupon_static.log") or {}).get("static", {})
    coup["driven"] = (tail_json("coupon_driven.log") or {}).get("driven", {})
    coup["knobs"] = (tail_json("coupon_knobs.log") or {}).get("knobs", {})
    s["coupon"] = coup
    s["compliance"] = tail_json("compliance.log") or {}
    s["coupon_mech"] = tail_json("coupon_mech.log") or {}
    s["mu_sweep"] = tail_json("mu.log") or {}
    s["tuning"] = tail_json("tuning.log") or {}
    # timings measured with nothing else on the GPU supersede the sweep's own
    clean = tail_json("iters_time.log") or {}
    s["iters_time"] = clean
    s["iters"] = tail_json("iters.log") or {}
    s["inner"] = tail_json("inner.log") or line_json("inner.log", ["outer"])
    s["grip"] = tail_json("grip.log") or line_json("grip.log", ["mujoco", "fpgs_"])
    misc = tail_json("misc.log") or {}
    s["misc"] = misc
    s["defaults"] = tail_json("defaults.log") or line_json("defaults.log", ["library_", "defaults_", "split_", "matrixfree_"])
    s["mesh2"] = tail_json("mesh2.log") or {}
    s["mesh"] = tail_json("mesh.log") or line_json("mesh.log", ["box", "mesh"])
    for k, v in (s.get("iters_time") or {}).items():
        if k.startswith("it") and k in s.get("iters", {}):
            s["iters"][k]["ms_per_frame"] = v["ms_per_frame"]
            s["iters"][k]["physics_rtf"] = v["physics_rtf"]
        if k.startswith("ss") and f"fpgs_{k}" in s.get("substeps", {}):
            s["substeps"][f"fpgs_{k}"]["ms_per_frame"] = v["ms_per_frame"]
            s["substeps"][f"fpgs_{k}"]["physics_rtf"] = v["physics_rtf"]
    SUM.write_text(json.dumps(s, indent=2, default=str))
    for k, v in s.items():
        n = len(v) if hasattr(v, "__len__") else 0
        print(f"{k:16s} {n} entries")

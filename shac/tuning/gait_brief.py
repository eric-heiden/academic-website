"""Prints a compact per-command summary of gait_eval.py results."""
import json
import sys

for f in sys.argv[1:]:
    print("==", f)
    for r in json.load(open(f)):
        ft = r.get("feet", {})
        line = (f"cmd {r['cmd'][0]:+.1f} {r['cmd'][1]:+.1f} {r['cmd'][2]:+.1f} surv {r['survival']:.2f} "
                f"err {r.get('err_xy', float('nan')):.3f}/{r.get('err_wz', float('nan')):.2f} "
                f"alt {r.get('alternation', float('nan')):.2f} sym(swing {r.get('swing_symmetry', float('nan')):.2f} "
                f"clr {r.get('clearance_symmetry', float('nan')):.2f}) ds {r.get('double_support', float('nan')):.2f} sep {r.get('min_foot_sep', float('nan')):.2f}")
        print(line)
        for n, s in ft.items():
            print(f"    {n[:12]:12s} td/s {s['touchdowns_per_s']:.2f} duty {s['duty']:.2f} swing {s['swing_s']:.2f} "
                  f"clr {100 * s['clearance_m']:.1f}cm stride {s['stride_m']:.2f} slip {s['slip_mps']:.2f} drag {s['drag_frac']:.2f}")

"""Verify + compare window-passage references with coal (am_sampling).

For each results/*.npz it loads (base, arm, theta) per knot, poses the coal body (base box + 8 arm
capsules + payload box) and the 5 window obstacles (4 wall borders + tilted sash), and reports the
SIGNED min distance (negative = penetration) per body part, plus base tilt and the box pose where it
crosses the wall. Lets us state, per method, how close the body comes to the window and whether the
keyframe pushes the passage lower / more tilted than the unguided soft_box.

Run: conda run -n am_sampling python src/planner/verify_window.py
"""
import os
import sys

import numpy as np
import coal

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))
from src.planner.sampling_compare import Geometry

RDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "results"))


def part_min_dists(g, q14, dreq, dres):
    """Signed min distance (m) from base / arms / box to ANY window obstacle at config q14."""
    g.fk14(q14)
    g._update_body()
    base_bt = g._body[0]
    box_bt = g._body[-1]
    arm_bts = g._body[1:-1]

    def md(parts):
        m = 1e9
        for bg, bT in parts:
            for og_, oT in g._win_obs:
                dres.clear()
                d = coal.distance(bg, bT, og_, oT, dreq, dres)
                if d < m:
                    m = d
        return m

    return md([base_bt]), md(arm_bts), md([box_bt])


def report(path, g, dreq, dres):
    if not os.path.exists(path):
        print(f"  (missing) {os.path.basename(path)}")
        return
    d = np.load(path)
    base, arm, box = d["base"], d["arm"], d["theta"]  # theta = base rotvec
    boxp = d["box"]
    N = len(base)
    bmin, amin, xmin = 1e9, 1e9, 1e9
    tilt_max = 0.0
    fails = 0
    win_lo, win_hi = g.win_wx[0] - 0.3, g.win_wx[1] + 0.3   # box crossing the wall x-slab (+/-30cm)
    tilt_at_win, boxz_at_win, boxx_at_win = [], [], []
    for k in range(N):
        q14 = np.concatenate([base[k], box[k], arm[k]])
        bd, ad, xd = part_min_dists(g, q14, dreq, dres)
        bmin, amin, xmin = min(bmin, bd), min(amin, ad), min(xmin, xd)
        tilt = np.degrees(np.linalg.norm(box[k]))
        tilt_max = max(tilt_max, tilt)
        if min(bd, ad, xd) < g.win_margin - 1e-6:
            fails += 1
        bx = boxp[k][0]
        if win_lo <= bx <= win_hi:
            tilt_at_win.append(tilt)
            boxz_at_win.append(boxp[k][2])
            boxx_at_win.append(bx)
    name = os.path.basename(path).replace("window_reference_", "").replace(".npz", "")
    print(f"\n=== {name}  ({N} knots) ===")
    print(f"  min signed clearance (cm):  BASE {100*bmin:+6.1f}   ARM {100*amin:+6.1f}   BOX {100*xmin:+6.1f}")
    print(f"  body-window penetrations:   {fails}/{N} knots")
    print(f"  base tilt (deg):            max {tilt_max:5.1f}")
    if tilt_at_win:
        i = int(np.argmin(boxz_at_win))  # lowest box pass
        print(f"  at the wall crossing ({len(tilt_at_win)} knots, box_x in [{win_lo:.2f},{win_hi:.2f}]):")
        print(f"      tilt   {min(tilt_at_win):4.1f} .. {max(tilt_at_win):4.1f} deg")
        print(f"      box z  {min(boxz_at_win):+.3f} .. {max(boxz_at_win):+.3f} m  (lowest at tilt {tilt_at_win[i]:.1f} deg)")
    print(f"  base x range:               {base[:,0].min():+.2f} .. {base[:,0].max():+.2f} m")
    print(f"  base z range:               {base[:,2].min():+.2f} .. {base[:,2].max():+.2f} m")


def main():
    g = Geometry()   # window auto-enabled from config/task.yaml
    dreq = coal.DistanceRequest()
    dreq.enable_signed_distance = True
    dres = coal.DistanceResult()
    print(f"window obstacles: {len(g._win_obs)}   body parts: {len(g._body)} (1 base + "
          f"{len(g._body)-2} caps + 1 box)   win_margin={g.win_margin}")
    # CLI: pass one or more npz paths (absolute, or basename resolved under results/). Defaults to the
    # two canonical g2 references (sampler-seeded vs keyframe-guided).
    files = sys.argv[1:] or ["window_reference_sampler_g2.npz", "window_reference_keyframe_g2.npz"]
    for f in files:
        report(f if os.path.isabs(f) else os.path.join(RDIR, f), g, dreq, dres)


if __name__ == "__main__":
    main()

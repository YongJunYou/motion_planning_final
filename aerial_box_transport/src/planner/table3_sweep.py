"""Table III: keyframe-pitch robustness (am_dualarm). Re-solves the keyframe-guided window PASSAGE at
pitch 50 / 60 / 70 deg with identical settings to the Table I keyframe row (window soft_box,
use_cylinders=False, transport_dur=9, track interp route + keyframe waypoint w_kf=40, NO placement
costs). The base min clearance at the wall is set by the passage, not by the downstream placement
refinement, so the 60 deg row here reproduces keyframe_g2's wall clearance (cross-check) while
isolating the keyframe-pitch effect on the passage homotopy.

Each pitch needs its own seed (build_kf_seed.py KF_PITCH=.. SEED_OUT=/tmp/window_kf_seed_p{pitch}.npz,
am_sampling) built beforehand. Saves results/table3_p{pitch}.npz; coal clearance via verify_window.py.

Run: conda run -n am_dualarm python src/planner/table3_sweep.py
Out: prints converged / solve time per pitch + writes the npz; saves results/table3_summary.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from planner.ocp import solve_ocp  # noqa: E402

DEG = np.pi / 180.0
TRANSPORT_DUR = 9.0
RDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "results"))
KF_BASE = np.array([-0.74, 0.0, 0.65])
PITCHES = [int(p) for p in os.environ.get("PITCHES", "50,60,70").split(",")]


def main():
    kf_arm = np.load("/tmp/keyframe_closed_arm.npy")
    out = {}
    for pitch in PITCHES:
        seed_path = f"/tmp/window_kf_seed_p{pitch}.npz"
        if not os.path.exists(seed_path):
            print(f"  pitch {pitch}: MISSING seed {seed_path} (build it in am_sampling first)")
            continue
        kf = np.load(seed_path)
        kf_q14 = np.concatenate([KF_BASE, [0.0, pitch * DEG, 0.0], kf_arm])
        r = solve_ocp(window=True, window_mode="soft_box", use_cylinders=False,
                      transport_dur=TRANSPORT_DUR, verbose=False,
                      seed={"q_route": kf["q_route"], "box_route": kf["box_route"]},
                      keyframe={"q14": kf_q14, "box_x": float(KF_BASE[0])}, w_kf=40.0)
        np.savez(os.path.join(RDIR, f"table3_p{pitch}.npz"), times=r["times"], base=r["base"],
                 arm=r["arm"], box=r["box"], theta=r["theta"], phase_bounds=r["phase_bounds"],
                 lam=r["lam"], fn_set=r["fn_set"], box_ref_z=r["box_ref_z"],
                 box_ref=r["box_ref"], grite_ref=r["grite_ref"])
        out[pitch] = {"converged": bool(r["converged"]), "solve_time": float(r["solve_time"]),
                      "iters": int(r["iter_count"]), "inf_pr": float(r["inf_pr"]),
                      "status": r["return_status"], "max_tilt_deg": float(r["max_tilt_deg"])}
        print(f"  pitch {pitch}: conv={r['converged']} t={r['solve_time']:.1f}s iters={r['iter_count']} "
              f"inf_pr={r['inf_pr']:.1e} max_tilt={r['max_tilt_deg']:.1f}deg  {r['return_status']}")
    with open(os.path.join(RDIR, "table3_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {RDIR}/table3_summary.json")


if __name__ == "__main__":
    main()

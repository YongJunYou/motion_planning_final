"""Probe: cap the keyframe window-passage solve at ~400 s and see if the trajectory is already good.

Table I shows the keyframe (ours) cold solve runs to the 1500-iter cap (~1700 s) while the sampler
exits at ~480 s. Question: is the keyframe trajectory already good early (and the rest is adaptive-mu
dual dithering past a feasible point), or does it actually need the full time? This runs the SAME
keyframe passage solve as table1_sweep.py but capped (COLD_MAXIT, default 400 iters ~= 400 s), saves the
trajectory, and prints when primal feasibility (inf_pr) crossed each threshold + how the objective moved.

Run: COLD_MAXIT=400 conda run -n am_dualarm python src/planner/probe_keyframe.py
Out: results/window_reference_keyframe_p400.npz  (coal-check + compare vs the full 1500-iter table1_keyframe.npz)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from planner.ocp import solve_ocp  # noqa: E402

DEG = np.pi / 180.0
RDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "results"))
KF_BASE = np.array([-0.74, 0.0, 0.65])


def main():
    cap = os.environ.get("COLD_MAXIT", "400")
    os.environ["COLD_MAXIT"] = cap
    kf = np.load("/tmp/window_kf_seed.npz")
    kf_arm = np.load("/tmp/keyframe_closed_arm.npy")
    kf_q14 = np.concatenate([KF_BASE, [0.0, 60.0 * DEG, 0.0], kf_arm])
    print(f"[probe] keyframe passage solve, COLD_MAXIT={cap} (~400 s target)", flush=True)
    r = solve_ocp(window=True, window_mode="soft_box", use_cylinders=False, transport_dur=9.0,
                  verbose=False, seed={"q_route": kf["q_route"], "box_route": kf["box_route"]},
                  keyframe={"q14": kf_q14, "box_x": float(KF_BASE[0])}, w_kf=40.0)

    out = os.path.join(RDIR, "window_reference_keyframe_p400.npz")
    np.savez(out, times=r["times"], base=r["base"], arm=r["arm"], box=r["box"], theta=r["theta"],
             phase_bounds=r["phase_bounds"], lam=r["lam"], fn_set=r["fn_set"],
             box_ref_z=r["box_ref_z"], box_ref=r["box_ref"], grite_ref=r["grite_ref"])

    print(f"\n[probe] solve_time {r['solve_time']:.1f}s  iters {r['iter_count']}  "
          f"inf_pr {r['inf_pr']:.2e}  status {r['return_status']}  max_tilt {r['max_tilt_deg']:.1f}deg")
    h = r["inf_pr_hist"]
    for thr in (1e-2, 1e-3, 1e-4, 1e-6):
        k = next((i for i, v in enumerate(h) if v < thr), None)
        print(f"  inf_pr < {thr:.0e} first reached at iter {k}")
    ob = r["obj_hist"]
    if ob:
        # how much did the objective still improve AFTER feasibility (inf_pr<1e-6)?
        kf6 = next((i for i, v in enumerate(h) if v < 1e-6), len(ob) - 1)
        print(f"  objective: start {ob[0]:.4g} -> at-feasible(iter {kf6}) {ob[kf6]:.4g} -> "
              f"final(iter {len(ob)-1}) {ob[-1]:.4g}")
        if ob[kf6] != 0:
            print(f"  objective change after feasibility: {100*(ob[-1]-ob[kf6])/abs(ob[kf6]):+.2f}%")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()

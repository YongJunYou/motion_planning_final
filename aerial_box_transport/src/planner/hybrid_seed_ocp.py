"""Hybrid (B): sampler -> OCP warm-start. The sampling-based planner finds the narrow-passage
grasp-constrained carry (where sampling excels); the OCP then REFINES it (smooth alignment / effort
/ dynamics, where gradient optimization excels). Demonstrates the standard pipeline -- "use sampling
to find the homotopy, optimization to minimize the cost" -- and that the OCP drives the box<->base
offset (the soft w_align cost) far below what the sampler + local post-opt could.

Two stages, in two conda envs (the sampler needs OMPL = am_sampling; the OCP needs CasADi/IPOPT =
am_dualarm), passing the seed through a file:

  conda run -n am_sampling python src/planner/hybrid_seed_ocp.py --stage build
  conda run -n am_dualarm  python src/planner/hybrid_seed_ocp.py --stage solve

Stage build  -> /tmp/sampler_seed.npz   (QR (109,14) + PB (109,3) warm-start)
Stage solve  -> results/hybrid_reference.npz   (OCP-refined tracker reference; tilt reported)
Then track it: conda run -n am_isaac python src/sim/track_reference.py --ref results/hybrid_reference.npz --loop
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np  # noqa: E402

SEED_FILE = "/tmp/sampler_seed.npz"


def stage_build():
    """am_sampling: plan the full sampler trajectory and save it as the OCP warm-start."""
    from planner.sampling_compare import Geometry
    from planner.sampling_to_reference import build_trajectory
    geom = Geometry()
    Q = build_trajectory(geom)                       # (109,14) [base3, theta3=0, arm8]
    # box (PB) consistent with the OCP's per-phase box constraints: pick (approach+grasp),
    # pad midpoint (transport), place (release). knots: appr 0..29, grasp 30..45, transp 46..95.
    PB = np.zeros((len(Q), 3))
    for k in range(len(Q)):
        if k < 46:
            PB[k] = geom.pick
        elif k < 96:
            PB[k] = 0.5 * (np.add(*geom.fk14(Q[k])))
        else:
            PB[k] = geom.place
    off = np.array([np.linalg.norm(PB[k][:2] - Q[k][:2]) for k in range(46, 96)])
    np.savez(SEED_FILE, QR=Q, PB=PB)
    print(f"[build] wrote {SEED_FILE}: QR {Q.shape}, PB {PB.shape}")
    print(f"[build] sampler transport box<->base offset: mean {off.mean()*100:.1f} cm, "
          f"max {off.max()*100:.1f} cm  (the OCP will drive this down)")


def stage_solve():
    """am_dualarm: warm-start the OCP from the sampler seed, refine, save the tracker reference."""
    import time
    from planner.ocp import solve_ocp, plot_phases
    d = np.load(SEED_FILE)
    # GENTLE handoff: seed only the box ROUTE (the sampler's narrow-passage homotopy). The OCP builds
    # its own consistent symmetric arm + velocity guess around it (a raw full-config seed fails).
    seed = {"box": d["PB"]}

    # box<->base offset of the SEED (sampler), for the before/after comparison
    base_s, box_s = d["QR"][:, 0:3], d["PB"]
    tr = slice(46, 96)
    off_seed = np.linalg.norm(box_s[tr, :2] - base_s[tr, :2], axis=1)

    # use_cylinders=False: the sampler's box route already gives the up-over-down homotopy, so the
    # OCP drops its own cylinder path-shaping device (which conflicts with the sampler route) and
    # keeps only keep_out (real collision) + the smooth costs it refines.
    t0 = time.time()
    res = solve_ocp(verbose=False, seed=seed, use_cylinders=False)
    dt = time.time() - t0
    base_o, box_o = res["base"], res["box"]
    off_ocp = np.linalg.norm(box_o[tr, :2] - base_o[tr, :2], axis=1)

    rdir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "results"))
    out = os.path.join(rdir, "hybrid_reference.npz")
    np.savez(out, times=res["times"], base=res["base"], arm=res["arm"], box=res["box"],
             lam=res["lam"], fn_set=res["fn_set"], box_ref_z=res["box_ref_z"],
             box_ref=res["box_ref"], grite_ref=res["grite_ref"])
    plot_phases(res, os.path.join(rdir, "hybrid_phases.png"))

    print(f"\n[solve] OCP status: {res['status']}  ({dt:.1f}s)")
    print(f"[solve] transport box<->base offset: sampler seed mean {off_seed.mean()*100:.1f} cm "
          f"-> OCP-refined mean {off_ocp.mean()*100:.1f} cm")
    print(f"[solve] OCP max base tilt over whole trajectory: {res['max_tilt_deg']:.2f} deg")
    print(f"[solve] wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["build", "solve"], required=True)
    args = ap.parse_args()
    if args.stage == "build":
        stage_build()
    else:
        stage_solve()


if __name__ == "__main__":
    main()

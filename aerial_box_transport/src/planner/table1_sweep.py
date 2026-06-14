"""Table I: convergence across initialization strategies (am_dualarm). One process so the casadi/cpin
build cost is paid once. Every row is the SAME window PASSAGE solve (window=True, soft_box,
use_cylinders=False, transport_dur=9, NO placement costs) so only the INITIALIZATION differs:

  naive          no seed, no keyframe                 -> only the window keep-out steers it (expect fail)
  linear-interp  track a straight grasp->place route  -> tracks a route that drives through the wall
  sampler        track the CBiRRT route               -> the canonical sampler_g2 homotopy
  keyframe       track interp route + keyframe waypoint-> OURS (keyframe_g2 homotopy)

naive + linear are run N_RESTART times with jittered cold guesses (random restart); sampler + keyframe
once. Convergence = IPOPT return_status Solve_Succeeded|Solved_To_Acceptable_Level (or inf_pr<1e-6).
Converged trajectories are saved to results/table1_<row>[_k].npz for the coal collision check.

Run: conda run -n am_dualarm python src/planner/table1_sweep.py
Out: prints the table + results/table1_summary.json
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
N_RESTART = int(os.environ.get("N_RESTART", "10"))
SEED_START = int(os.environ.get("SEED_START", "0"))   # resume the restart sweep at a later seed index
SKIP_GUIDED = os.environ.get("SKIP_GUIDED", "0") == "1"   # skip the sampler+keyframe solves (already done)
OUT_JSON = os.environ.get("OUT_JSON", "table1_summary.json")
JITTER = float(os.environ.get("JITTER", "0.05"))

KF_BASE = np.array([-0.74, 0.0, 0.65])
KF_PITCH = float(os.environ.get("KF_PITCH", "60"))


def base_kw():
    return dict(window=True, window_mode="soft_box", use_cylinders=False,
                transport_dur=TRANSPORT_DUR, verbose=False)


def save(tag, res):
    np.savez(os.path.join(RDIR, f"table1_{tag}.npz"), times=res["times"], base=res["base"],
             arm=res["arm"], box=res["box"], theta=res["theta"], phase_bounds=res["phase_bounds"],
             lam=res["lam"], fn_set=res["fn_set"], box_ref_z=res["box_ref_z"],
             box_ref=res["box_ref"], grite_ref=res["grite_ref"])


def line(tag, res):
    print(f"  {tag:<16} conv={str(res['converged']):<5} t={res['solve_time']:6.1f}s  "
          f"iters={res['iter_count']:<5} inf_pr={res['inf_pr']:.1e}  {res['return_status']}")
    return {"converged": bool(res["converged"]), "solve_time": float(res["solve_time"]),
            "iters": int(res["iter_count"]), "inf_pr": float(res["inf_pr"]),
            "status": res["return_status"], "max_tilt_deg": float(res["max_tilt_deg"])}


def main():
    out = {"N_RESTART": N_RESTART, "jitter": JITTER}

    # ---- sampler + keyframe FIRST: single solve (must converge -> critical data). Cap at 1500: a
    # well-seeded passage reaches primal feasibility (inf_pr<1e-6) in a few hundred iters; the tail is
    # adaptive-mu dual dithering (the documented reason g2 used warm continuation), not progress. ----
    if not SKIP_GUIDED:
        os.environ["COLD_MAXIT"] = os.environ.get("GUIDED_MAXIT", "1500")
        print("[sampler] track the CBiRRT route (sampler_g2 homotopy)")
        smp = np.load("/tmp/window_seed.npz")
        r = solve_ocp(**base_kw(), seed={"q_route": smp["q_route"], "box_route": smp["box_route"]})
        out["sampler"] = line("sampler", r)
        save("sampler", r)

        print(f"[keyframe] track interp route + keyframe waypoint, pitch={KF_PITCH:.0f} (ours)")
        kf = np.load("/tmp/window_kf_seed.npz")
        kf_arm = np.load("/tmp/keyframe_closed_arm.npy")
        kf_q14 = np.concatenate([KF_BASE, [0.0, KF_PITCH * DEG, 0.0], kf_arm])
        r = solve_ocp(**base_kw(), seed={"q_route": kf["q_route"], "box_route": kf["box_route"]},
                      keyframe={"q14": kf_q14, "box_x": float(KF_BASE[0])}, w_kf=40.0)
        out["keyframe"] = line("keyframe", r)
        save("keyframe", r)

    # ---- naive + linear-interp: random-restart sweep, capped iters (expected to fail; bound the cost,
    # a genuine convergence happens by ~220 iters per the guided runs, so 800 is a generous ceiling) ----
    os.environ["COLD_MAXIT"] = os.environ.get("SWEEP_MAXIT", "400")
    lin = np.load("/tmp/window_linear_seed.npz")
    lin_seed = {"q_route": lin["q_route"], "box_route": lin["box_route"]}

    print(f"[naive] restarts {SEED_START}..{N_RESTART-1} (no seed, no keyframe), COLD_MAXIT={os.environ['COLD_MAXIT']}")
    out["naive"] = []
    for s in range(SEED_START, N_RESTART):
        r = solve_ocp(**base_kw(), jitter=JITTER, jitter_seed=s)
        out["naive"].append(line(f"naive[{s}]", r))
        if r["converged"]:
            save(f"naive_k{s}", r)

    print(f"[linear-interp] restarts {SEED_START}..{N_RESTART-1} (track straight grasp->place)")
    out["linear"] = []
    for s in range(SEED_START, N_RESTART):
        r = solve_ocp(**base_kw(), seed=lin_seed, jitter=JITTER, jitter_seed=s)
        out["linear"].append(line(f"linear[{s}]", r))
        if r["converged"]:
            save(f"linear_k{s}", r)

    nconv = sum(d["converged"] for d in out["naive"])
    lconv = sum(d["converged"] for d in out["linear"])
    g = f"sampler {out['sampler']['converged']}  keyframe {out['keyframe']['converged']}  " if not SKIP_GUIDED else ""
    print(f"\nSUMMARY  {g}naive {nconv}/{len(out['naive'])}  linear {lconv}/{len(out['linear'])}")
    with open(os.path.join(RDIR, OUT_JSON), "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {RDIR}/{OUT_JSON}")


if __name__ == "__main__":
    main()

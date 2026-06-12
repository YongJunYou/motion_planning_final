"""Keyframe-seeded OCP: hand the optimizer ONE full config (a keyframe) to warm-start from,
then let it find the global-optimal whole-body trajectory freely (the keyframe is a SEED, not a
constraint -- no waypoint is enforced, so the result is still the unconstrained OCP optimum).

The keyframe is a single robot config the carry should pass THROUGH near the transport apex:
  base position (HOME frame), a base pitch, and a 4-DoF arm config [dof1, dof2, dof3, dof4].
solve_ocp() blends it smoothly (Gaussian bump) into the base/attitude/arm initial guess over the
transport knots and rebuilds the velocity/accel guess so IPOPT starts dynamically consistent.

Default keyframe (this run):
  position (world) (-0.74, 0, 2.15) -> HOME frame (-0.74, 0, 0.65)  [home = world - z1.5]
  orientation pitch +60 deg
  joints (0, 150, 90, 60) deg per arm  (dof1 mirrored L/R; dof2-dof3-dof4 = 0 -> parallel jaws)

Run: conda run -n am_dualarm python src/planner/keyframe_seed_ocp.py
Outputs: results/keyframe_reference.npz, results/keyframe_phases.png
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np  # noqa: E402

from planner.ocp import plot_phases, solve_ocp  # noqa: E402

SPAWN_Z = 1.5    # drone spawn height [m]; home frame = world - spawn (matches config pick/place)
DEG = np.pi / 180.0


def main():
    # --- keyframe config (edit here) ---------------------------------------------------------
    pos_world = np.array([-0.74, 0.0, 2.15])             # base position, WORLD frame [m]
    pitch_deg = 60.0                                     # base pitch (about world y) [deg]
    arm_deg = np.array([0.0, 150.0, 90.0, 60.0])         # [dof1, dof2, dof3, dof4] per arm [deg]
    # ----------------------------------------------------------------------------------------

    kf_base = pos_world - np.array([0.0, 0.0, SPAWN_Z])  # -> home frame: (-0.74, 0, 0.65)
    keyframe = {"base": kf_base, "pitch": pitch_deg * DEG, "arm": arm_deg * DEG}

    print(f"[keyframe] base world {pos_world.tolist()} -> home {np.round(kf_base, 3).tolist()} m")
    print(f"[keyframe] pitch {pitch_deg:.0f} deg, arm(deg) {arm_deg.tolist()} "
          f"(parallel-jaw check dof2-dof3-dof4 = {arm_deg[1]-arm_deg[2]-arm_deg[3]:.0f} deg)")

    # use_cylinders=False: the cylinders are the OCP's OWN hand-crafted up-over-down path device
    # (confine the box to a narrow vertical column over pick/place below a height). A keyframe that
    # sits on the place SIDE but LOW (here base z=0.65 -> box ~0.38, below z_top_place) collides
    # head-on with that constraint -- the alignment cost pulls the box under the keyframe base while
    # the cylinder forces it back over x=place, so IPOPT's step computation fails. Like the sampler
    # hybrid (hybrid_seed_ocp.py), we drop the cylinders and keep only keep_out (real collision) +
    # the smooth costs, so the keyframe defines the carry shape and the OCP refines it.
    t0 = time.time()
    res = solve_ocp(verbose=True, keyframe=keyframe, use_cylinders=False)
    dt = time.time() - t0

    rdir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "results"))
    os.makedirs(rdir, exist_ok=True)
    out = os.path.join(rdir, "keyframe_reference.npz")
    np.savez(out, times=res["times"], base=res["base"], arm=res["arm"], box=res["box"],
             lam=res["lam"], fn_set=res["fn_set"], box_ref_z=res["box_ref_z"],
             box_ref=res["box_ref"], grite_ref=res["grite_ref"])
    plot_phases(res, os.path.join(rdir, "keyframe_phases.png"))

    lam, ph = res["lam"], res["phase_of"]
    gm = lam[[k for k, p in enumerate(ph) if p == "grasp"]].mean()
    tm = lam[[k for k, p in enumerate(ph) if p == "transport"]].mean()
    print(f"\n[keyframe-OCP] status: {res['status']}  ({dt:.1f}s)")
    print(f"[keyframe-OCP] peak lambda: {lam.max():.2f} N, mean grasp={gm:.2f} transport={tm:.2f} N")
    print(f"[keyframe-OCP] base flies: x {res['base'][:, 0].min():.2f}..{res['base'][:, 0].max():.2f}, "
          f"z {res['base'][:, 2].min():.2f}..{res['base'][:, 2].max():.2f} m")
    print(f"[keyframe-OCP] max base tilt over whole trajectory: {res['max_tilt_deg']:.2f} deg")
    print(f"[keyframe-OCP] wrote {out}")
    print(f"[keyframe-OCP] figure -> {rdir}/keyframe_phases.png")


if __name__ == "__main__":
    main()

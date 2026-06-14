"""Keyframe-guided window OCP (method 2: interpolation warm-start + soft waypoint cost, NO sampler).

Instead of running the CBiRRT sampler to make a seed, we use a single human-provided keyframe (the
drone configuration at the window) to (a) build the warm-start by interpolating grasp -> keyframe ->
place, and (b) add a soft waypoint cost that pulls the window knot toward the keyframe. The sampler
stage is skipped entirely. This isolates "does a human keyframe alone guide the OCP to the good
passage" for the paper's comparison vs the sampler-guided (soft_box) and unguided runs.

Run: conda run -n am_dualarm python src/planner/keyframe_window.py
Out: results/window_reference_keyframe.npz
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))  # src/

TRANSPORT_DUR = 9.0
RDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "results"))
DEG = np.pi / 180.0

# the confirmed window keyframe (teammate pose, grip closed to a01): see GUI verification.
KF_BASE = np.array([-0.74, 0.0, 0.65])          # HOME frame (world 2.15 - spawn 1.5)
KF_PITCH = float(os.environ.get("KF_PITCH", "60"))   # keyframe base pitch (deg); Table III sweeps 50/60/70
KF_ROTVEC = np.array([0.0, KF_PITCH * DEG, 0.0])     # pitch about y (must match build_kf_seed.py KF_PITCH)
KF_BOX = np.array([-0.74, -0.585, 0.65])        # box (grip midpoint), home frame
KF_BOX_X = -0.74                                # x where the keyframe waypoint applies


def _lerp(a, b, n):
    return np.array([a + (b - a) * t for t in np.linspace(0.0, 1.0, n)])


def main():
    kf_arm = np.load("/tmp/keyframe_closed_arm.npy")
    kf_q14 = np.concatenate([KF_BASE, KF_ROTVEC, kf_arm])

    # CONSTRAINT-CONSISTENT warm-start through the keyframe (built by build_kf_seed.py in am_sampling):
    # symmetric arms (dof1_l=-dof1_r, dof2-4 L=R) + box DERIVED from FK so box = base + R(theta)*mid_arm
    # holds at every knot. This starts IPOPT on the grasp manifold (vs the inconsistent linear-interp
    # seed that hit local infeasibility). The keyframe waypoint then pulls the window knot to the pose.
    # continuation/homotopy on w_kf: start gentle, then ramp up warm-starting each from the previous
    # solution. W_KF (weight) and KF_SEED (warm-start file) are env-controlled so the ramp is scriptable.
    seed_path = os.environ.get("KF_SEED", "/tmp/window_kf_seed.npz")
    w_kf = float(os.environ.get("W_KF", "40"))
    w_place = float(os.environ.get("W_PLACE", "300"))
    w_level = float(os.environ.get("W_LEVEL", "200"))
    w_padlevel = float(os.environ.get("W_PADLEVEL", "400"))   # level the EE pads at the rack so the
    #                          rigidly-gripped box is set down on its bottom face (box orient = EE orient)
    w_rise = float(os.environ.get("W_RISE", "800"))           # up-and-over the rack before descending
    seed = np.load(seed_path)
    # CONTINUATION: optionally warm-start the FULL trajectory from a previously-converged result
    # (WARM_FROM). The place/level costs are too sensitive to add from the cold interp seed (the solver
    # loses the good basin), so first solve plain (w_place=w_level=0) to make the converged reference,
    # then re-solve WARM_FROM=that with the costs on -- IPOPT starts feasible and only nudges the place.
    warm = None
    warm_from = os.environ.get("WARM_FROM", "")
    if warm_from and os.path.exists(warm_from):
        w = np.load(warm_from)
        warm = {"base": w["base"], "theta": w["theta"], "arm": w["arm"], "box": w["box"]}
    print(f"[KEYFRAME-OCP] w_kf={w_kf}  w_place={w_place}  w_level={w_level}  w_padlevel={w_padlevel}  "
          f"seed={seed_path}  warm={warm_from or 'none'}")

    from planner.ocp import solve_ocp
    res = solve_ocp(window=True, window_mode="soft_box", verbose=True, transport_dur=TRANSPORT_DUR,
                    use_cylinders=False,   # CRITICAL: the pick/place vertical cylinders conflict with the
                    #                        window passage (hybrid_window passes this; we had defaulted True)
                    # w_place + w_level: soft, rack-gated. Pull the box STRAIGHT DOWN onto the rack and
                    # level the base (tilt+yaw->0) at the rack so the rigidly-gripped box lands upright.
                    w_place=w_place, w_level=w_level, w_padlevel=w_padlevel, w_rise=w_rise, warm=warm,
                    seed={"q_route": seed["q_route"], "box_route": seed["box_route"]},
                    keyframe={"q14": kf_q14, "box_x": KF_BOX_X}, w_kf=w_kf)

    out = os.path.join(RDIR, os.environ.get("KF_OUT", "window_reference_keyframe.npz"))
    np.savez(out, times=res["times"], base=res["base"], arm=res["arm"], box=res["box"],
             theta=res["theta"], phase_bounds=res["phase_bounds"], lam=res["lam"],
             fn_set=res["fn_set"], box_ref_z=res["box_ref_z"], box_ref=res["box_ref"],
             grite_ref=res["grite_ref"])
    print(f"\n[KEYFRAME-OCP] status: {res['status']}")
    print(f"[KEYFRAME-OCP] base x {res['base'][:,0].min():.2f}..{res['base'][:,0].max():.2f}, "
          f"z {res['base'][:,2].min():.2f}..{res['base'][:,2].max():.2f} m")
    print(f"[KEYFRAME-OCP] max base tilt (rotvec norm) {res['max_tilt_deg']:.2f} deg")
    print(f"[KEYFRAME-OCP] wrote {out}")


if __name__ == "__main__":
    main()

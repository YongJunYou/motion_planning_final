"""Build a CONSTRAINT-CONSISTENT warm-start seed for the keyframe-guided OCP (am_sampling, needs FK).

The keyframe-guided OCP must start IPOPT on the OCP's own grasp manifold, or it parks at local
infeasibility. So we interpolate the CONFIG (base, attitude, arm) grasp -> keyframe -> place keeping
the constraints the OCP enforces, and DERIVE the box route from forward kinematics:
  - symmetric arms: dof1_l = -dof1_r, dof2-4 L = R (linear interp of mirror-symmetric anchors stays so)
  - box[k] = 0.5*(EE_l + EE_r) = base + R(theta)*mid_arm  (so the grasp constraint holds at every knot)
Anchors use the FULL r_off (level-grasp EE midpoint, NOT y-zeroed) so the anchor box lands exactly on
pick / place. keyframe_window.py (am_dualarm) loads the result and solves.

Run: conda run -n am_sampling python src/planner/build_kf_seed.py
Out: /tmp/window_kf_seed.npz  {q_route (M,14), box_route (M,3)}
"""
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))  # repo root
from src.planner.sampling_compare import Geometry

DEG = np.pi / 180.0
KF_BASE = np.array([-0.74, 0.0, 0.65])          # home frame (world 2.15 - spawn 1.5)
KF_PITCH = float(os.environ.get("KF_PITCH", "60"))   # keyframe base pitch (deg); Table III sweeps 50/60/70
KF_ROTVEC = np.array([0.0, KF_PITCH * DEG, 0.0])     # pitch about y
SEED_MODE = os.environ.get("SEED_MODE", "keyframe")  # "keyframe" (g->kf->over->place) | "linear" (g->place)
SEED_OUT = os.environ.get("SEED_OUT", "/tmp/window_kf_seed.npz")


def main():
    g = Geometry()
    cfg = yaml.safe_load(open("config/task.yaml"))
    pick = np.array(cfg["ocp"]["pick"], float)
    place = np.array(cfg["ocp"]["place"], float)
    ag = np.array(cfg["ocp"]["arm_grasp"], float)
    if len(ag) == 4:
        ag = np.concatenate([ag, [-ag[0], ag[1], ag[2], ag[3]]])   # mirror dof1 -> full 8-dof arm

    pl0, pr0 = g.fk14(np.concatenate([[0, 0, 0.0], [0, 0, 0.0], ag]))   # EE midpoint at identity base
    r_off = 0.5 * (np.array(pl0) + np.array(pr0))                       # FULL (anchor box == pick/place)
    base_grasp, base_place = pick - r_off, place - r_off

    A_g = np.concatenate([base_grasp, [0, 0, 0.0], ag])
    A_o = np.concatenate([base_place + np.array([0, 0, 0.561]), [0, 0, 0.0], ag])  # box high over the slot
    A_p = np.concatenate([base_place, [0, 0, 0.0], ag])

    def lerp(a, b, n):
        return np.array([a + (b - a) * t for t in np.linspace(0.0, 1.0, n)])

    if SEED_MODE == "linear":
        # TrajOpt/CHOMP-style baseline: a NAIVE straight C-space interpolation grasp -> place, no keyframe
        # and no up-and-over. The straight route drives the box through the SOLID wall (it ignores the
        # window opening), so the OCP that tracks it should stay trapped against the wall (Table I).
        q_route = lerp(A_g, A_p, 32)
        kf_idx = -1
    else:
        kf_arm = np.load("/tmp/keyframe_closed_arm.npy")
        A_k = np.concatenate([KF_BASE, KF_ROTVEC, kf_arm])
        # grasp -> keyframe -> OVER (box ~z 1.05, clears the rack frame) -> place (descend into the slot).
        # The straight kf->place segment dived into the rack frame too low (z 0.49 < keep-out 0.68) and
        # jammed the box at the frame; the OVER anchor makes the seed satisfy the rack keep-out.
        q_route = np.vstack([lerp(A_g, A_k, 12), lerp(A_k, A_o, 12)[1:], lerp(A_o, A_p, 10)[1:]])
        kf_idx = 14
    box_route = np.array([0.5 * (np.add(*g.fk14(q))) for q in q_route])   # box = FK(config)

    np.savez(SEED_OUT, q_route=q_route, box_route=box_route)
    clear = sum(int(g.window_clear_body(q)) for q in q_route)
    kf_txt = "n/a" if kf_idx < 0 else str(np.round(box_route[kf_idx], 3))
    print(f"mode={SEED_MODE} pitch={KF_PITCH:.0f}  box grasp {np.round(box_route[0],3)}  "
          f"kf {kf_txt}  place {np.round(box_route[-1],3)}")
    print(f"seed knots window-clear {clear}/{len(q_route)}; wrote {SEED_OUT}")


if __name__ == "__main__":
    main()

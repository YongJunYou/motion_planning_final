"""Derive reach-DOWN grasp + pre-grasp configs: the arm drops the pads to box height
while the base hovers higher, so the base clears the desk/rack surface. dof1 (opposite
sign on the two arms) tilts the arms down symmetrically; it preserves the pad x (grip
width) and the world-x face normal (vertical pads pressing the box +-x faces). dof2/3/4
keep their proven values (gap + parallel faces, dof2-dof3-dof4 = 0). We solve only dof1
per config to hit a target vertical DROP of the pads below the base.

base clearance above the desk top = (box_center_z - desk_top) + DROP. With the box
5.7 cm above the desk, DROP = 0.27 m puts the base ~0.33 m above the desk surface.

Run: conda run -n am_dualarm python src/model/reach_down.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np  # noqa: E402
import pinocchio as pin  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

from model.whole_body import WholeBody  # noqa: E402

DROP = 0.27   # how far below the base the pads sit [m] (sets the base clearance)


def main():
    wb = WholeBody()
    model, data = wb.model, wb.data
    fid_l, fid_r = wb.ee_frame_ids["ee_l"], wb.ee_frame_ids["ee_r"]

    def pad(arm, fid):
        q = np.zeros(model.nq)
        q[:7] = [0, 0, 0, 0, 0, 0, 1]
        q[7:] = arm
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        M = data.oMf[fid]
        return np.array(M.translation), np.array(M.rotation)

    def arm_of(d1, d2, d3, d4):                 # opposite dof1 on the two arms
        return np.array([d1, d2, d3, d4, -d1, d2, d3, d4], float)

    def solve_dof1(d2, d3, d4):                 # dof1 so the left pad sits DROP below base
        f = lambda d1: pad(arm_of(d1, d2, d3, d4), fid_l)[0][2] + DROP
        return brentq(f, 0.0, 1.4)

    def report(label, d2, d3, d4):
        d1 = solve_dof1(d2, d3, d4)
        a = arm_of(d1, d2, d3, d4)
        pl, Rl = pad(a, fid_l)
        pr, Rr = pad(a, fid_r)
        gap = (pr[0] - pl[0]) * 100
        nl, nr = Rl[:, 0], Rr[:, 0]             # local x-axis = pad face normal
        print(f"{label}: dof1=+-{d1:.4f}  [d1,d2,d3,d4]_L=[{d1:.4f},{d2},{d3},{d4}]")
        print(f"   PL=({pl[0]:+.4f},{pl[1]:+.4f},{pl[2]:+.4f})  "
              f"PR=({pr[0]:+.4f},{pr[1]:+.4f},{pr[2]:+.4f})  gap={gap:.2f} cm")
        print(f"   L normal={nl.round(3).tolist()} R normal={nr.round(3).tolist()} "
              f"(both along world x -> parallel, vertical faces)")
        print(f"   pad drop below base = {-pl[2]*100:.1f} cm (both arms symmetric in z)")
        return d1

    print(f"target pad DROP below base = {DROP} m\n")
    d1_g = report("GRASP   ", 2.56, 1.16, 1.4)
    print()
    d1_p = report("PREGRASP", 2.4353, 1.0866, 1.3487)

    # base clearance check (home frame: world - 1.5)
    a = arm_of(d1_g, 2.56, 1.16, 1.4)
    pl, _ = pad(a, fid_l)
    pr, _ = pad(a, fid_r)
    r_off = 0.5 * (pl + pr)
    pick = np.array([2.0, 0.0, -0.662])
    desk_top_home = 0.781 - 1.5
    base_grasp = pick - r_off
    print(f"\nr_off = {r_off.round(4).tolist()} (base->pad-midpoint offset)")
    print(f"base_grasp = {base_grasp.round(4).tolist()} (home frame)")
    print(f"base clearance above desk top = {(base_grasp[2]-desk_top_home)*100:.1f} cm")
    print(f"\nyaml: arm_grasp:    [{d1_g:.4f}, 2.56, 1.16, 1.4]")
    print(f"yaml: arm_pregrasp: [{d1_p:.4f}, 2.4353, 1.0866, 1.3487]")


if __name__ == "__main__":
    main()

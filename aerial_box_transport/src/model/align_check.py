"""Feasibility check for the box-under-base alignment cost.

The box centre during a grip = the EE pad midpoint. Its horizontal offset from the
base centre is r_off.x, r_off.y (FK at the arm config, base at origin/level). r_off.x
is ~0 by symmetry; r_off.y is the FORWARD reach. "Box on the vertical line through the
base" means r_off.y -> 0 (arm reaches STRAIGHT down). Can the arm do that while keeping
a box-width gap and parallel jaws (dof2-dof3-dof4 = 0)? Solve for it and report the
achievable min |pad midpoint y| and the corresponding vertical drop.

Run: conda run -n am_dualarm python src/model/align_check.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np  # noqa: E402
import pinocchio as pin  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

from model.whole_body import WholeBody  # noqa: E402

GAP = 0.105   # box width [m]


def main():
    wb = WholeBody()
    model, data = wb.model, wb.data
    fid_l, fid_r = wb.ee_frame_ids["ee_l"], wb.ee_frame_ids["ee_r"]

    def pads(d1, d2, d3, d4):
        q = np.zeros(model.nq)
        q[:7] = [0, 0, 0, 0, 0, 0, 1]
        q[7:] = np.array([d1, d2, d3, d4, -d1, d2, d3, d4], float)
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        return np.array(data.oMf[fid_l].translation), np.array(data.oMf[fid_r].translation)

    # current grasp config for reference
    pl, pr = pads(0.6789, 2.56, 1.16, 1.4)
    mid = 0.5 * (pl + pr)
    print(f"current grasp config [0.6789,2.56,1.16,1.4]:")
    print(f"   pad midpoint = ({mid[0]:+.4f}, {mid[1]:+.4f}, {mid[2]:+.4f})  "
          f"-> forward offset r_off.y = {mid[1]:+.3f} m, gap={(pr[0]-pl[0])*100:.1f} cm")
    print()

    # solve for the arm config that puts the pad midpoint DIRECTLY BELOW the base
    # (mid.x = 0, mid.y = 0), with a box-width gap and parallel jaws.
    def resid(p):
        d1, d2, d3 = p
        d4 = d2 - d3                       # parallel jaws (dof2 - dof3 - dof4 = 0)
        pl, pr = pads(d1, d2, d3, d4)
        m = 0.5 * (pl + pr)
        return [m[0], m[1], (pr[0] - pl[0]) - GAP]   # mid.x=0, mid.y=0, gap=box width

    best = None
    for d1_0 in (0.4, 0.7, 1.0, 1.3):     # several starts (nonconvex)
        for d2_0 in (1.8, 2.2, 2.56, 2.9):
            sol = least_squares(resid, [d1_0, d2_0, 1.0],
                                bounds=([0, 0, 0], [np.pi, np.pi, np.pi]))
            r = np.linalg.norm(sol.fun)
            if best is None or r < best[0]:
                best = (r, sol.x)
    r, x = best
    d1, d2, d3 = x
    d4 = d2 - d3
    pl, pr = pads(d1, d2, d3, d4)
    m = 0.5 * (pl + pr)
    print(f"straight-down solve (target mid.x=mid.y=0, gap={GAP*100:.1f} cm, parallel):")
    print(f"   residual = {r:.2e}  (near 0 => straight-down grasp is FEASIBLE)")
    print(f"   config [d1,d2,d3,d4] = [{d1:.4f}, {d2:.4f}, {d3:.4f}, {d4:.4f}]")
    print(f"   pad midpoint = ({m[0]:+.4f}, {m[1]:+.4f}, {m[2]:+.4f})  "
          f"gap={(pr[0]-pl[0])*100:.1f} cm")
    print(f"   forward offset r_off.y = {m[1]:+.4f} m, vertical drop = {-m[2]:.3f} m")
    in_lim = all(0 <= v <= np.pi for v in (d2, d3, d4))
    print(f"   dof2,3,4 in [0,pi]? {in_lim}")


if __name__ == "__main__":
    main()

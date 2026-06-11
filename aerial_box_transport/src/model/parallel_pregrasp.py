"""Solve for a PARALLEL-jaw pre-grasp: pads facing along world x (same orientation
as the grasp config) but a wider x-gap, so closing is a pure parallel squeeze.

The pad facing angle is a linear combination of the joint angles. Empirically the
invariant that holds the face along world x is D = dof2 - dof3 - dof4 (NOT the sum:
the joint z-axes have mixed signs). At the grasp config D = 2.56 - 1.16 - 1.4 = 0
(phi = 0, faces parallel along x). Holding D = 0 keeps the faces parallel; we solve
the remaining 2 DOF to push the pad outward in x while keeping its y (reach) matched
to the grasp, i.e. a pure lateral opening (only the gap changes).

Run: conda run -n am_dualarm python src/model/parallel_pregrasp.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np  # noqa: E402
import pinocchio as pin  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

from model.whole_body import WholeBody  # noqa: E402


def main():
    wb = WholeBody()
    model, data = wb.model, wb.data
    fid_l, fid_r = wb.ee_frame_ids["ee_l"], wb.ee_frame_ids["ee_r"]

    def arm_of(d2, d3, d4):
        return np.array([0, d2, d3, d4, 0, d2, d3, d4], float)

    def pad(arm, fid):
        q = np.zeros(model.nq)
        q[:7] = [0, 0, 0, 0, 0, 0, 1]   # base at origin, identity quat (xyzw)
        q[7:] = arm
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        M = data.oMf[fid]
        return np.array(M.translation), np.array(M.rotation)

    def phi_of(R):
        return np.degrees(np.arctan2(R[1, 0], R[0, 0]))

    gp = (2.56, 1.16, 1.4)
    pl_g, Rl_g = pad(arm_of(*gp), fid_l)
    pr_g, Rr_g = pad(arm_of(*gp), fid_r)
    D_g = gp[0] - gp[1] - gp[2]
    print(f"grasp {gp}  D=dof2-dof3-dof4={D_g:.3f}  gap={(pr_g[0]-pl_g[0])*100:.2f} cm  "
          f"left pad=({pl_g[0]:+.4f},{pl_g[1]:+.4f})  phi_l={phi_of(Rl_g):+.2f} "
          f"phi_r={phi_of(Rr_g):+.2f} deg")
    print()

    for gap_cm in (16.0, 18.0, 20.0, 22.0):
        x_open = -gap_cm / 200.0   # left pad target x (half the gap, negative side)

        def resid(p):
            d2, d3, d4 = p
            pl, _ = pad(arm_of(d2, d3, d4), fid_l)
            return [d2 - d3 - d4,            # keep faces parallel along x (D = 0)
                    pl[0] - x_open,          # widen the gap (target x)
                    pl[1] - pl_g[1]]         # keep grasp y (pure lateral opening)

        sol = least_squares(resid, [2.56, 1.16, 1.4],
                            bounds=([0.0, 0.0, 0.0], [np.pi, np.pi, np.pi]))
        d2, d3, d4 = sol.x
        pl, Rl = pad(arm_of(d2, d3, d4), fid_l)
        pr, Rr = pad(arm_of(d2, d3, d4), fid_r)
        res = np.linalg.norm(sol.fun)
        print(f"target {gap_cm:4.1f} cm -> [{d2:.4f}, {d3:.4f}, {d4:.4f}]  "
              f"D={d2-d3-d4:+.4f}  gap={(pr[0]-pl[0])*100:6.2f} cm  "
              f"pad=({pl[0]:+.4f},{pl[1]:+.4f})  phi_l={phi_of(Rl):+.2f} "
              f"phi_r={phi_of(Rr):+.2f} deg  resid={res:.2e}")


if __name__ == "__main__":
    main()

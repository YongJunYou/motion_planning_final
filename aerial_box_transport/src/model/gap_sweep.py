"""Quick sweep: how the pre-grasp arm config sets the gripper pad gap.

The two pads face each other across world x; the gap is the x-separation of the
pad inner-face centres (fk_ee). Vary dof3 (elbow) around the current pre-grasp to
find a WIDER open config, keeping the pads level (same z) and facing (same y).

Run: conda run -n am_dualarm python src/model/gap_sweep.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np  # noqa: E402

from model.whole_body import WholeBody  # noqa: E402


def gap_for(wb, dof2, dof3, dof4):
    arm = np.array([0, dof2, dof3, dof4, 0, dof2, dof3, dof4], float)
    q = np.concatenate([[0, 0, 0], [0, 0, 0, 1], arm])  # quat xyzw identity
    pl, pr = wb.fk_ee(q)
    pl, pr = np.array(pl).ravel(), np.array(pr).ravel()
    gap_x = pr[0] - pl[0]
    return gap_x, pl, pr


def main():
    wb = WholeBody()
    pre = [2.56, 1.34, 1.4]    # current pre-grasp
    grasp = [2.56, 1.16, 1.4]  # current grasp (box face = 10.5 cm)
    for label, cfg in [("grasp", grasp), ("pregrasp", pre)]:
        g, pl, pr = gap_for(wb, *cfg)
        print(f"{label:9s} {cfg}  gap_x={g*100:6.2f} cm  "
              f"pl={pl.round(4).tolist()} pr={pr.round(4).tolist()}")
    print("\n--- sweep dof3 (elbow), dof2=2.56 dof4=1.4 fixed ---")
    for dof3 in np.arange(1.10, 2.31, 0.10):
        g, pl, pr = gap_for(wb, 2.56, float(dof3), 1.4)
        print(f"dof3={dof3:4.2f}  gap_x={g*100:6.2f} cm  "
              f"pad z={pl[2]:+.3f}/{pr[2]:+.3f}  pad y={pl[1]:+.3f}/{pr[1]:+.3f}")
    print("\n--- sweep dof2 (shoulder), dof3=1.34 dof4=1.4 fixed ---")
    for dof2 in np.arange(2.20, 2.91, 0.10):
        g, pl, pr = gap_for(wb, float(dof2), 1.34, 1.4)
        print(f"dof2={dof2:4.2f}  gap_x={g*100:6.2f} cm  "
              f"pad z={pl[2]:+.3f}/{pr[2]:+.3f}  pad y={pl[1]:+.3f}/{pr[1]:+.3f}")


if __name__ == "__main__":
    main()

"""Print the base roll/pitch/yaw range over the wall-crossing slab for any window reference npz.

Used to classify the passage homotopy: "pitch-forward" = pitch leads (large) while yaw stays small;
"yaw-sideways" = yaw swings large while pitch stays small. Box x in [-1.40, -0.70] is the slab.

Run: conda run -n am_sampling python src/planner/crossing_rpy.py results/table3_p50.npz ...
"""
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

RDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "results"))
XLO, XHI = -1.40, -0.70


def main():
    files = sys.argv[1:] or ["window_reference_sampler_g2.npz", "window_reference_keyframe_g2.npz"]
    for f in files:
        p = f if os.path.isabs(f) else os.path.join(RDIR, f)
        if not os.path.exists(p):
            print(f"{os.path.basename(f):<34}  (missing)")
            continue
        d = np.load(p)
        rpy = R.from_rotvec(d["theta"]).as_euler("xyz", degrees=True)
        bx = d["box"][:, 0]
        m = (bx >= XLO) & (bx <= XHI)
        if not m.any():
            print(f"{os.path.basename(p):<34}  (no knots in slab)")
            continue
        roll, pitch, yaw = rpy[m, 0], rpy[m, 1], rpy[m, 2]
        homotopy = "pitch-forward" if abs(pitch).max() > abs(yaw).max() else "yaw-sideways"
        name = os.path.basename(p).replace("window_reference_", "").replace(".npz", "")
        print(f"{name:<30} pitch [{pitch.min():+5.0f},{pitch.max():+5.0f}]  "
              f"yaw [{yaw.min():+5.0f},{yaw.max():+5.0f}]  roll [{roll.min():+5.0f},{roll.max():+5.0f}]"
              f"  -> {homotopy}")


if __name__ == "__main__":
    main()

"""Table II path-length metrics for the window references (post-processing only, no solver).

For each npz (base, theta=base rotvec, arm) computes:
  - base translational path length (m)   = sum ||base[k+1] - base[k]||
  - base rotational path length (deg)    = sum geodesic angle of R[k]^T R[k+1]
  - arm joint path length (deg, L2)      = sum ||arm[k+1] - arm[k]||_2  (text-only metric)

The rotational length is the discriminating metric: a yaw twist-and-untwist inflates it even when the
net base rotation is small, so the sideways-yaw sampler reference reads much larger than the
pitch-forward keyframe one.

Run: conda run -n am_sampling python src/planner/path_lengths.py [npz ...]
Defaults to the two canonical g2 references.
"""
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

RDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "results"))


def lengths(path):
    d = np.load(path)
    base, theta, arm = d["base"], d["theta"], d["arm"]   # theta = base rotvec (N,3)
    N = len(base)
    trans = float(np.sum(np.linalg.norm(np.diff(base, axis=0), axis=1)))
    Rk = R.from_rotvec(theta)
    rel = Rk[:-1].inv() * Rk[1:]                          # R[k]^T R[k+1]
    rot_deg = float(np.degrees(np.sum(np.linalg.norm(rel.as_rotvec(), axis=1))))
    arm_deg = float(np.degrees(np.sum(np.linalg.norm(np.diff(arm, axis=0), axis=1))))
    return N, trans, rot_deg, arm_deg


def main():
    files = sys.argv[1:] or ["window_reference_sampler_g2.npz", "window_reference_keyframe_g2.npz"]
    print(f"{'reference':<34}{'N':>5}{'trans (m)':>12}{'rot (deg)':>12}{'arm (deg)':>12}")
    for f in files:
        p = f if os.path.isabs(f) else os.path.join(RDIR, f)
        if not os.path.exists(p):
            print(f"{os.path.basename(f):<34}  (missing)")
            continue
        N, trans, rot_deg, arm_deg = lengths(p)
        name = os.path.basename(p).replace("window_reference_", "").replace(".npz", "")
        print(f"{name:<34}{N:>5}{trans:>12.3f}{rot_deg:>12.1f}{arm_deg:>12.1f}")


if __name__ == "__main__":
    main()

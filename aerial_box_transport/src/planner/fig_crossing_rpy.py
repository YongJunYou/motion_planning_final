"""Fig 1: base roll/pitch/yaw through the wall crossing, sampler vs keyframe (post-processing only).

theta[k] (base rotvec) is decomposed to euler xyz = (roll, pitch, yaw) in degrees and plotted against
the box x position so the two homotopies are directly comparable: the sampling-seeded reference swings
YAW toward ~ -90 deg to squeeze the body through sideways while PITCH stays small, whereas the
keyframe-guided reference keeps YAW near 0 and leads with PITCH (the human-intended pitch-forward
passage). The wall-crossing slab (box x in [-1.40, -0.70]) is shaded.

Run: conda run -n am_sampling python src/planner/fig_crossing_rpy.py
Out: results/fig_crossing_rpy.png  (+ printed pitch/yaw extrema over the crossing slab)
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

RDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "results"))
XLO, XHI = -1.40, -0.70   # wall-crossing slab in box x


def load(name):
    d = np.load(os.path.join(RDIR, name))
    rpy = R.from_rotvec(d["theta"]).as_euler("xyz", degrees=True)  # (N,3): roll, pitch, yaw
    return d["box"][:, 0], rpy


def main():
    refs = [("window_reference_sampler_g2.npz", "sampling-seeded"),
            ("window_reference_keyframe_g2.npz", "keyframe-guided (ours)")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (name, title) in zip(axes, refs):
        bx, rpy = load(name)
        roll, pitch, yaw = rpy[:, 0], rpy[:, 1], rpy[:, 2]
        ax.axvspan(XLO, XHI, color="0.85", zorder=0, label="wall slab")
        ax.plot(bx, pitch, "-", color="C0", label="pitch")
        ax.plot(bx, yaw, "-", color="C3", label="yaw")
        ax.plot(bx, roll, "-", color="C2", label="roll")
        ax.set_title(title)
        ax.set_xlabel("box x (m)")
        ax.set_xlim(1.2, -2.2)   # box travels +x (desk) -> -x (rack); reverse so motion reads L->R
        ax.grid(alpha=0.3)
        m = (bx >= XLO) & (bx <= XHI)
        print(f"{title:<26} crossing: pitch [{pitch[m].min():+5.0f},{pitch[m].max():+5.0f}]  "
              f"yaw [{yaw[m].min():+5.0f},{yaw[m].max():+5.0f}]  deg")
    axes[0].set_ylabel("base euler angle (deg)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out = os.path.join(RDIR, "fig_crossing_rpy.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

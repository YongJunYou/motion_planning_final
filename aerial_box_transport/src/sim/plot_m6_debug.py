"""Diagnose the M6 demo: why do the box and the grippers move separately?

Loads results/track_log.npz (from a single track_reference pass) and plots:
  1. base position actual vs reference (per axis) -> base tracking error
  2. planned box position vs the ACTUAL gripper midpoint (per axis)
  3. the box-to-gripper gap |box_set - ee_mid| -> if large during transport, the
     open-loop box path has drifted away from where the grippers actually are.

Run: conda run -n am_dualarm python src/sim/plot_m6_debug.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_R = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "results"))
d = np.load(os.path.join(_R, "track_log.npz"))
t = d["t"]
p, p_d = d["p"], d["p_d"]
ee_mid, box_set = d["ee_mid"], d["box_set"]
gap = np.linalg.norm(box_set - ee_mid, axis=1)

fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
lbl = ["x", "y", "z"]
for i in range(3):
    ax[0].plot(t, p[:, i], f"C{i}-", lw=1.5, label=f"base {lbl[i]}")
    ax[0].plot(t, p_d[:, i], f"C{i}:", lw=1.5)
ax[0].set_ylabel("base position [m]")
ax[0].set_title("Base: actual (solid) vs reference (dotted)")
ax[0].legend(fontsize=8, ncol=3)
ax[0].grid(alpha=0.3)

for i in range(3):
    ax[1].plot(t, box_set[:, i], f"C{i}-", lw=1.5, label=f"box {lbl[i]}")
    ax[1].plot(t, ee_mid[:, i], f"C{i}--", lw=1.5, label=f"grip {lbl[i]}")
ax[1].set_ylabel("position [m]")
ax[1].set_title("ACTUAL box (solid) vs actual gripper midpoint (dashed) -- grip holds if they overlap")
ax[1].legend(fontsize=7, ncol=3)
ax[1].grid(alpha=0.3)

ax[2].plot(t, gap * 100, "r-", lw=2)
ax[2].set_ylabel("box-to-gripper gap [cm]")
ax[2].set_xlabel("time [s]")
ax[2].set_title(f"|box - gripper midpoint|  (grip holds if small in transport; max {gap.max()*100:.1f} cm)")
ax[2].grid(alpha=0.3)

fig.tight_layout()
out = os.path.join(_R, "m6_debug.png")
fig.savefig(out, dpi=130)
print(f"wrote {out}")
print(f"base RMSE = {np.sqrt(np.mean(np.sum((p - p_d)**2, axis=1)))*1e3:.1f} mm")
print(f"box-to-gripper gap: mean {gap.mean()*100:.1f} cm, max {gap.max()*100:.1f} cm")
# where is the gap worst?
k = int(gap.argmax())
print(f"worst gap at t={t[k]:.1f}s: planned box={np.round(box_set[k],2).tolist()}, "
      f"actual grip={np.round(ee_mid[k],2).tolist()}")

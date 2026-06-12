"""Top-down before/after figure of the gripper grasp vs the 12 cm box.

Before: arm = [dof2,dof3]=[2.75,1.35], wrist dof4 = 0 (what the GUI showed).
After:  arm = [dof2,dof3,dof4]=[2.64,1.36,1.28] (wrist engaged, pads facing).
Each panel projects the two actual link4 pad meshes onto the base x-y plane and
overlays the 12 cm box, so the pad separation and facing direction are visible.

Run: conda run -n am_dualarm python src/model/plot_grasp_fix.py
"""
import os
import sys

import numpy as np
import pinocchio as pin
from pxr import Usd, UsdGeom

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_THIS, os.pardir)))
from model.whole_body import build_planning_model  # noqa: E402

USD_PATH = os.path.abspath(os.path.join(_THIS, os.pardir, os.pardir, os.pardir, "dual_arm_final.usd"))


def world_pts(prim):
    a = UsdGeom.Mesh(prim).GetPointsAttr().Get()
    M = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    P = np.array([[v[0], v[1], v[2]] for v in a])
    R = np.array([[float(M[i][j]) for j in range(3)] for i in range(3)])
    t = np.array([float(M[3][0]), float(M[3][1]), float(M[3][2])])
    return P @ R + t


def main():
    stage = Usd.Stage.Open(USD_PATH)
    it = Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
    meshes = [p for p in it if p.IsA(UsdGeom.Mesh)]
    pads = {}
    for side in ("l", "r"):
        cs = [(len(UsdGeom.Mesh(p).GetPointsAttr().Get()), p) for p in meshes
              if f"/{side}_link4_01/{side}_link4_01/" in str(p.GetPath())]
        cs.sort(key=lambda c: c[0])
        pads[side] = world_pts(cs[0][1])

    model = build_planning_model()
    data = model.createData()
    pin.forwardKinematics(model, data, pin.neutral(model))
    pin.updateFramePlacements(model, data)
    padb = {s: np.array([data.oMf[model.getFrameId(f"{s}_link4_01")].actInv(p) for p in pads[s]])
            for s in ("l", "r")}
    aidx = {model.names[j]: model.joints[j].idx_q for j in range(model.njoints)
            if model.names[j].startswith("dof")}

    def pad_xy(d2, d3, d4):
        q = pin.neutral(model)
        for jn, val in {"dof_l2": d2, "dof_l3": d3, "dof_l4": d4,
                        "dof_r2": d2, "dof_r3": d3, "dof_r4": d4}.items():
            q[aidx[jn]] = val
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        out = {}
        for s in ("l", "r"):
            oMf = data.oMf[model.getFrameId(f"{s}_link4_01")].copy()
            out[s] = np.array([oMf.act(p) for p in padb[s]])
        return out

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharex=True, sharey=True)
    panels = [("Before  (wrist dof4 = 0)", (2.75, 1.35, 0.0), axes[0]),
              ("After  (wrist dof4 = 1.28, pads facing)", (2.64, 1.36, 1.28), axes[1])]
    for title, (d2, d3, d4), ax in panels:
        g = pad_xy(d2, d3, d4)
        cy = 0.5 * (g["l"][:, 1].mean() + g["r"][:, 1].mean())
        ax.add_patch(Rectangle((-0.06, cy - 0.06), 0.12, 0.12, fill=False,
                               edgecolor="k", lw=2, ls="--", label="12 cm box"))
        for s, col in (("l", "C0"), ("r", "C3")):
            ax.scatter(g[s][:, 0], g[s][:, 1], s=6, c=col, alpha=0.6)
        l_in = g["l"][:, 0].max()
        r_in = g["r"][:, 0].min()
        ax.annotate("", xy=(r_in, cy), xytext=(l_in, cy),
                    arrowprops=dict(arrowstyle="<->", color="green", lw=1.8))
        ax.text(0.5, 0.93, f"pad gap = {(r_in - l_in) * 100:.1f} cm", transform=ax.transAxes,
                ha="center", color="green", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_xlabel("base x [m]  (squeeze axis)")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.16, 0.16)
    axes[0].set_ylabel("base y [m]  (reach direction)")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Gripper pads vs the 12 cm box, top-down (base x-y plane)", fontsize=12)
    fig.tight_layout()
    out = os.path.abspath(os.path.join(_THIS, os.pardir, os.pardir, "results", "grasp_fix_comparison.png"))
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

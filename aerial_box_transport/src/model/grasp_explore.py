"""Explore the reachable grasp envelope to find a DOWNWARD grasp.

For a box resting on a desk, a horizontal reach puts the drone body at the box
(desk) height and collides. We want the EE below the base (drone hovers above the
box, arms reach down-and-forward) while the pads still face along x and are one
box-width apart. This scans (dof2, dof3, dof4) with dof1 = 0 (so the left/right
pads stay mirror-symmetric about x = 0) and reports, for facing-x grasps at the
box gap, how far DOWN (cen_z) and forward (cen_y) the EE can be placed.

Run: conda run -n am_dualarm python src/model/grasp_explore.py
"""
import os
import sys

import numpy as np
import pinocchio as pin
from pxr import Usd, UsdGeom

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_THIS, os.pardir)))
from model.whole_body import build_planning_model, ARM_JOINTS  # noqa: E402

USD = "/home/jaewoo/Research/motion_planning_final/dual_arm_final.usd"
GAP = 0.105


def main():
    stage = Usd.Stage.Open(USD)
    it = Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
    meshes = [p for p in it if p.IsA(UsdGeom.Mesh)]
    pads = {}
    for s in ("l", "r"):
        cs = [(len(UsdGeom.Mesh(p).GetPointsAttr().Get()), p) for p in meshes
              if f"/{s}_link4_01/{s}_link4_01/" in str(p.GetPath())]
        cs.sort(key=lambda c: c[0])
        m = UsdGeom.Mesh(cs[0][1])
        a = m.GetPointsAttr().Get()
        M = UsdGeom.Xformable(cs[0][1]).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        P = np.array([[v[0], v[1], v[2]] for v in a])
        R = np.array([[float(M[i][j]) for j in range(3)] for i in range(3)])
        t = np.array([float(M[3][k]) for k in range(3)])
        pads[s] = P @ R + t

    model = build_planning_model()
    data = model.createData()
    pin.forwardKinematics(model, data, pin.neutral(model))
    pin.updateFramePlacements(model, data)
    padb = {s: np.array([data.oMf[model.getFrameId(f"{s}_link4_01")].actInv(p) for p in pads[s]])
            for s in ("l", "r")}
    aidx = {jn: model.joints[model.getJointId(jn)].idx_q for jn in ARM_JOINTS}

    def fk(d2, d3, d4):
        q = pin.neutral(model)
        for jn, v in {"dof_l2": d2, "dof_l3": d3, "dof_l4": d4,
                      "dof_r2": d2, "dof_r3": d3, "dof_r4": d4}.items():
            q[aidx[jn]] = v
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        out = {}
        for s in ("l", "r"):
            oMf = data.oMf[model.getFrameId(f"{s}_link4_01")].copy()
            W = padb[s] @ oMf.rotation.T + oMf.translation     # vectorized act()
            out[s] = (W, oMf.rotation @ np.array([1.0, 0, 0]))
        return out

    cand = []
    for d2 in np.arange(0.4, 3.1, 0.05):
        for d3 in np.arange(0.3, 2.6, 0.05):
            for d4 in np.arange(0.0, 3.1, 0.1):
                r = fk(d2, d3, d4)
                L, nL = r["l"]
                R, _ = r["r"]
                l_in, r_in = L[:, 0].max(), R[:, 0].min()
                gap = r_in - l_in
                if abs(gap - GAP) > 0.006 or nL[0] < 0.9:
                    continue
                cen = 0.5 * (L.mean(0) + R.mean(0))
                cand.append((cen[2], d2, d3, d4, cen, nL[0], gap))
    if not cand:
        print("no facing-x grasp at gap found")
        return
    cand.sort()  # most negative cen_z first (most downward)
    print(f"facing-x grasps at {GAP*100:.1f} cm gap, sorted by EE height below base (cen_z):")
    print(f"{'d2':>6}{'d3':>6}{'d4':>6} | {'cen_x':>7}{'cen_y':>7}{'cen_z':>7} {'n_x':>5}")
    for c in cand[:6]:
        _, d2, d3, d4, cen, nx, gap = c
        print(f"{d2:6.2f}{d3:6.2f}{d4:6.2f} | {cen[0]:+7.3f}{cen[1]:+7.3f}{cen[2]:+7.3f} {nx:5.2f}")
    print("  ... (most downward above)")
    print(f"\ndeepest downward reach: cen_z = {cand[0][0]:.3f} m below the base")
    print("horizontal reference (current grasp 2.56,1.16,1.4):")
    r = fk(2.56, 1.16, 1.4)
    cen = 0.5 * (r["l"][0].mean(0) + r["r"][0].mean(0))
    print(f"  cen = {np.round(cen, 3).tolist()}  (cen_z ~ 0 = same height as base)")


if __name__ == "__main__":
    main()

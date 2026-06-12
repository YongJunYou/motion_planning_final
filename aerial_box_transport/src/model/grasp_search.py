"""Derive the grasp arm configuration from the ACTUAL gripper pad mesh.

The EE that touches the box is the thin rubber contact pad on link4 (239-vertex
mesh), not the wrist body frame. This script pulls that pad mesh from the USD
(headless usd-core, instance-proxy traversal), expresses it in the link4 body
frame via the Pinocchio model at q=0, then searches arm joints (shoulder dof2,
elbow dof3, wrist dof4; dof1 = 0) for a posture where:
  - the pad flat face is parallel to the box face (face normal along base x, so
    the two pads face each other across the box, not along the reach direction),
  - the pad's nearest point to the box (the actual contact) is ~6 cm off centre,
    i.e. the two pads grasp a 12 cm box.
It then defines the EE as that nearest contact point (so the OCP's contact = the
visible pad surface) and prints the EE offset, grasp/pregrasp configs, and
box_center to paste into whole_body.py / task.yaml.

Run: conda run -n am_dualarm python src/model/grasp_search.py
"""
import os
import sys

import numpy as np
import pinocchio as pin
from pxr import Usd, UsdGeom

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_THIS, os.pardir)))
from model.whole_body import build_planning_model, ARM_JOINTS  # noqa: E402

USD_PATH = os.path.abspath(os.path.join(_THIS, os.pardir, os.pardir, os.pardir, "dual_arm_final.usd"))
BOX_EDGE = 0.105   # teammate's cubebox_a01 is a 10.5 cm cube (measured from the USD)


def world_pts(prim):
    a = UsdGeom.Mesh(prim).GetPointsAttr().Get()
    M = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    P = np.array([[v[0], v[1], v[2]] for v in a])
    R = np.array([[float(M[i][j]) for j in range(3)] for i in range(3)])
    t = np.array([float(M[3][0]), float(M[3][1]), float(M[3][2])])
    return P @ R + t


def pad_meshes(stage):
    it = Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
    meshes = [p for p in it if p.IsA(UsdGeom.Mesh)]
    out = {}
    for side in ("l", "r"):
        cands = [(len(UsdGeom.Mesh(p).GetPointsAttr().Get()), p) for p in meshes
                 if f"/{side}_link4_01/{side}_link4_01/" in str(p.GetPath())]
        cands.sort(key=lambda c: c[0])
        out[side] = world_pts(cands[0][1])
    return out


def main():
    stage = Usd.Stage.Open(USD_PATH)
    pads_world = pad_meshes(stage)

    model = build_planning_model()
    data = model.createData()
    pin.forwardKinematics(model, data, pin.neutral(model))
    pin.updateFramePlacements(model, data)
    pad_body = {}
    for side in ("l", "r"):
        oMf0 = data.oMf[model.getFrameId(f"{side}_link4_01")].copy()
        pad_body[side] = np.array([oMf0.actInv(p) for p in pads_world[side]])

    aidx = {jn: model.joints[model.getJointId(jn)].idx_q for jn in ARM_JOINTS}

    def fk_pads(d2, d3, d4):
        q = pin.neutral(model)
        for jn, val in {"dof_l2": d2, "dof_l3": d3, "dof_l4": d4,
                        "dof_r2": d2, "dof_r3": d3, "dof_r4": d4}.items():
            q[aidx[jn]] = val
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        res = {}
        for side in ("l", "r"):
            oMf = data.oMf[model.getFrameId(f"{side}_link4_01")].copy()
            res[side] = (np.array([oMf.act(p) for p in pad_body[side]]),
                         oMf.rotation @ np.array([1.0, 0.0, 0.0]))
        return res

    # search: pad faces parallel to box face (max n_x), nearest contact pts ~ +/-0.06
    cands = []
    for d4 in np.arange(0.6, 2.4, 0.04):
        for d2 in np.arange(2.0, 3.0, 0.04):
            for d3 in np.arange(0.6, 1.9, 0.04):
                res = fk_pads(d2, d3, d4)
                L, nL = res["l"]
                R, _ = res["r"]
                l_in = L[:, 0].max()        # left pad nearest point to centre (contact)
                r_in = R[:, 0].min()
                gap = r_in - l_in
                cen_y = 0.5 * (L[:, 1].mean() + R[:, 1].mean())
                if nL[0] < 0.97:            # require nearly-parallel pad faces
                    continue
                if not (-0.50 < cen_y < -0.36):   # moderate forward reach
                    continue
                score = abs(gap - BOX_EDGE) + 0.2 * (1.0 - nL[0])
                cands.append((score, d2, d3, d4, l_in, r_in, gap, cen_y, nL[0]))
    cands.sort()
    print(f"[search] parallel-face 12 cm grasp (n_x>0.97, reach -0.36..-0.50)")
    print(f"{'d2':>6} {'d3':>6} {'d4':>6} | {'gap':>7} {'cen_y':>7} {'n_x':>6}")
    for c in cands[:6]:
        print(f"{c[1]:6.2f} {c[2]:6.2f} {c[3]:6.2f} | {c[6]:7.3f} {c[7]:+.3f} {c[8]:6.3f}")
    _, GD2, GD3, GD4, l_in, r_in, gap, cen_y, nx = cands[0]
    GD2, GD3, GD4 = round(GD2, 3), round(GD3, 3), round(GD4, 3)

    # EE = centre of the pad's inner contact face (the surface that meets the box).
    # With n_x = 1 the face is parallel to the box face, so all inner-face vertices
    # share one base-x; averaging them gives a clean, symmetric contact point.
    ee_off = {}
    res = fk_pads(GD2, GD3, GD4)
    for side, want_max in (("l", True), ("r", False)):
        W = res[side][0]
        edge = W[:, 0].max() if want_max else W[:, 0].min()
        on_face = np.abs(W[:, 0] - edge) < 3e-3
        ee_off[side] = pad_body[side][on_face].mean(0)
    print(f"\n[grasp] [d2,d3,d4]=[{GD2},{GD3},{GD4}]  contact gap={gap:.4f} m  n_x={nx:.3f}")
    print("[recipe] EE_GRIPPER_OFFSET (pad nearest-contact point, body frame):")
    print(f'    "ee_l": np.array({np.round(ee_off["l"], 4).tolist()}),')
    print(f'    "ee_r": np.array({np.round(ee_off["r"], 4).tolist()}),')
    box_z = 0.5 * (res["l"][0][:, 2].mean() + res["r"][0][:, 2].mean())
    print(f"[recipe] box_center=[0.0, {cen_y:.4f}, {box_z:.4f}]  half_extent=0.06")

    # pregrasp: open the contact gap to ~14.5 cm, same wrist/shoulder
    for d3 in np.arange(GD3 - 0.7, GD3 + 0.7, 0.01):
        r = fk_pads(GD2, d3, GD4)
        g = r["r"][0][:, 0].min() - r["l"][0][:, 0].max()
        if abs(g - 0.145) < 0.004 and r["l"][1][0] > 0.95:
            print(f"[recipe] arm_pregrasp=[{GD2}, {round(float(d3),3)}, {GD4}]  "
                  f"(contact gap {g:.3f} m, just outside)")
            break
    print(f"[recipe] arm_grasp=[{GD2}, {GD3}, {GD4}]")


if __name__ == "__main__":
    main()

"""Extract the robot kinematic and inertial structure from the USD.

Headless USD parse using usd-core (no Isaac Sim). Reads link masses, centers of
mass, inertias, and the revolute-joint tree with axes and limits, then writes a
JSON description that M1 (whole_body.py) consumes to build the Pinocchio model.
The robot model comes ONLY from this USD (the vlm_drone_ws hardware differs).

Run:  conda run -n am_dualarm python src/model/extract_usd.py
Output: config/usd_model.json plus a printed summary.
"""
import json
import os
import re

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics

USD_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir,
                                        os.pardir, "dual_arm_final.usd"))
OUT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, "config", "usd_model.json"))


def quat_wxyz(q):
    if q is None:
        return [1.0, 0.0, 0.0, 0.0]
    im = q.GetImaginary()
    return [float(q.GetReal()), float(im[0]), float(im[1]), float(im[2])]


def vec3(v, default=(0.0, 0.0, 0.0)):
    if v is None:
        return list(default)
    return [float(v[0]), float(v[1]), float(v[2])]


def world_pose(prim):
    """World rest pose at default time: translation [m] and column-vector rotation R."""
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = [float(m[3][0]), float(m[3][1]), float(m[3][2])]
    R = np.array([[float(m[i][j]) for j in range(3)] for i in range(3)]).T
    return t, R.tolist()


def classify(name):
    if re.fullmatch(r"dof_[lr][1-4]", name):
        return "arm"
    if re.fullmatch(r"dof_[lr][bf][12]", name):
        return "tilt"
    return "other"


def main():
    stage = Usd.Stage.Open(USD_PATH)
    prims = list(stage.Traverse())

    bodies = {}
    for p in prims:
        if not p.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        path = str(p.GetPath())
        m = UsdPhysics.MassAPI(p)
        t, R = world_pose(p)
        mass = m.GetMassAttr().Get()
        bodies[path] = {
            "name": p.GetName(),
            "path": path,
            "mass": float(mass) if mass is not None else 0.0,
            "com": vec3(m.GetCenterOfMassAttr().Get()),
            "diag_inertia": vec3(m.GetDiagonalInertiaAttr().Get()),
            "principal_axes": quat_wxyz(m.GetPrincipalAxesAttr().Get()),
            "world_t": t,
            "world_R": R,
        }

    joints = []
    for p in prims:
        if not p.IsA(UsdPhysics.RevoluteJoint):
            continue
        j = UsdPhysics.RevoluteJoint(p)
        b0 = j.GetBody0Rel().GetTargets()
        b1 = j.GetBody1Rel().GetTargets()
        lo = j.GetLowerLimitAttr().Get()
        hi = j.GetUpperLimitAttr().Get()
        joints.append({
            "name": p.GetName(),
            "path": str(p.GetPath()),
            "kind": classify(p.GetName()),
            "axis": j.GetAxisAttr().Get(),
            "lower_deg": float(lo) if lo is not None else None,
            "upper_deg": float(hi) if hi is not None else None,
            "body0": str(b0[0]) if b0 else None,
            "body1": str(b1[0]) if b1 else None,
            "local_pos0": vec3(j.GetLocalPos0Attr().Get()),
            "local_rot0": quat_wxyz(j.GetLocalRot0Attr().Get()),
            "local_pos1": vec3(j.GetLocalPos1Attr().Get()),
            "local_rot1": quat_wxyz(j.GetLocalRot1Attr().Get()),
        })

    art = [str(p.GetPath()) for p in prims if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    children = {j["body1"] for j in joints}
    roots = [path for path in bodies if path not in children]

    model = {
        "usd_path": USD_PATH,
        "articulation_roots": art,
        "base_candidates": roots,
        "bodies": list(bodies.values()),
        "joints": joints,
        "n_arm_joints": sum(1 for j in joints if j["kind"] == "arm"),
        "n_tilt_joints": sum(1 for j in joints if j["kind"] == "tilt"),
        "total_mass": sum(b["mass"] for b in bodies.values()),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(model, f, indent=2)

    print(f"bodies: {len(bodies)} | joints: {len(joints)} | articulation roots: {art}")
    print(f"arm joints: {model['n_arm_joints']} | tilt joints: {model['n_tilt_joints']}")
    print(f"total mass: {model['total_mass']:.4f} kg")
    print(f"base candidates (no parent joint): {[bodies[r]['name'] for r in roots]}")
    print("\narm joint chain  (name: parent -> child | axis | limits[deg]):")
    for j in joints:
        if j["kind"] == "arm":
            pn = bodies.get(j["body0"], {}).get("name", j["body0"])
            cn = bodies.get(j["body1"], {}).get("name", j["body1"])
            print(f"  {j['name']}: {pn} -> {cn} | axis {j['axis']} "
                  f"| [{j['lower_deg']}, {j['upper_deg']}]")
    print("\nper-body mass [kg]:")
    for b in bodies.values():
        print(f"  {b['name']:<22} m={b['mass']:.4f}  com={np.round(b['com'],4).tolist()}")
    print(f"\nwrote {OUT}")
    if model["n_arm_joints"] != 8:
        print(f"\n[WARN] expected 8 arm joints, found {model['n_arm_joints']}")


if __name__ == "__main__":
    main()

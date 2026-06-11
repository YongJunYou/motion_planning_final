"""Validate the M1 Pinocchio model against the Isaac/PhysX data.

The decisive check: set the model to the exact joint configuration Isaac was in
when dumped, run forward kinematics, and compare every link frame placement to
Isaac's body_pos_w / body_quat_w. This validates the anchor-based kinematic
construction (placements + axes), not just the trivial zero config. Also checks
total mass, whole-body center of mass, and the cpin EE Jacobian self-consistency.

Run:  conda run -n am_dualarm python src/model/validate_m1.py
"""
import numpy as np
import pinocchio as pin

from model.whole_body import (
    IsaacBodies, WholeBody, build_full_model, load_extracted, quat_wxyz_to_R,
)


def dump_config(model, isaac):
    q = pin.neutral(model)
    rp = np.asarray(isaac["root_pos_w"], float).ravel()
    rq = np.asarray(isaac["root_quat_w"], float).ravel()  # wxyz
    q[0:3] = rp
    q[3:7] = [rq[1], rq[2], rq[3], rq[0]]  # pinocchio free-flyer uses xyzw
    jpos = {n: float(p) for n, p in zip(isaac["joint_names"], isaac["joint_pos"])}
    for jid in range(1, model.njoints):
        name = model.names[jid]
        if name in jpos:
            q[model.joints[jid].idx_q] = jpos[name]
    return q


def validate_kinematics():
    usd, isaac = load_extracted()
    full, _ = build_full_model(usd, isaac)
    data = full.createData()
    bodies = IsaacBodies(isaac)
    q = dump_config(full, isaac)
    pin.forwardKinematics(full, data, q)
    pin.updateFramePlacements(full, data)

    print("per-body FK vs Isaac at the dump config:")
    max_pe = max_re = 0.0
    for name in bodies.names:
        if not full.existFrame(name):
            continue
        oMf = data.oMf[full.getFrameId(name)]
        p_iso = bodies.pos[bodies.idx[name]]
        R_iso = quat_wxyz_to_R(bodies.quat[bodies.idx[name]])
        pe = float(np.linalg.norm(oMf.translation - p_iso))
        re = float(np.linalg.norm(pin.log3(oMf.rotation.T @ R_iso)))
        max_pe, max_re = max(max_pe, pe), max(max_re, re)
        print(f"  {name:<16} pe={pe * 1e3:7.2f} mm  re={np.degrees(re):6.2f} deg")
    print(f"MAX pos err = {max_pe * 1e3:.2f} mm | MAX rot err = {np.degrees(max_re):.2f} deg")

    total = sum(I.mass for I in full.inertias)
    com = pin.centerOfMass(full, data, q)
    print(f"\ntotal mass model = {total:.4f} kg (Isaac {float(np.sum(bodies.masses)):.4f})")
    print(f"whole-body CoM (model, dump config) = {np.array(com).round(4).tolist()}")
    return max_pe, max_re


def validate_jacobian():
    wb = WholeBody()
    q = pin.neutral(wb.model)
    q[7:] = 0.3
    J_l = np.array(wb.J_ee(q)[0])[:3, :]
    p0 = np.array(wb.fk_ee(q)[0]).ravel()
    fd = np.zeros((3, wb.nv))
    eps = 1e-6
    for i in range(wb.nv):
        dv = np.zeros(wb.nv)
        dv[i] = eps
        qp = pin.integrate(wb.model, q, dv)
        fd[:, i] = (np.array(wb.fk_ee(qp)[0]).ravel() - p0) / eps
    err = float(np.max(np.abs(fd - J_l)))
    print(f"\nEE_l Jacobian finite-diff max err = {err:.2e}")
    return err


if __name__ == "__main__":
    pe, re = validate_kinematics()
    jerr = validate_jacobian()
    ok = pe < 5e-3 and re < 1e-2 and jerr < 1e-4
    print("\n=== M1 validation ===")
    print(f"kinematics match Isaac (pos<5mm, rot<0.57deg): {pe < 5e-3 and re < 1e-2}")
    print(f"jacobian self-consistent (<1e-4): {jerr < 1e-4}")
    print("RESULT:", "PASS" if ok else "NEEDS REVIEW")

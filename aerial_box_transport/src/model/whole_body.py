"""M1: whole-body model (Pinocchio + pinocchio.casadi).

Built from two extracted artifacts:
  - config/usd_model.json   kinematics: joint tree, axes, limits, joint local frames
  - config/isaac_model.json inertials (mass, com, inertia) from PhysX

Important property of this USD: at the zero configuration all rigid-body link
frames coincide at the base origin (Isaac body_pos_w is ~0 for every link), and
the real arm geometry (link lengths) is encoded in the joint LOCAL frames
(local_pose0 == local_pose1 = the joint anchor). So we build the kinematics from
the joint anchors, not from body world poses:

  - each revolute joint frame is placed at its USD anchor A = SE3(local_pos1, local_rot1)
  - jointPlacement_i = A_parent^-1 * A_i   (parent anchor -> this anchor)
  - the child link frame sits at A_i^-1 relative to the joint (so at q=0 it is at
    the base origin, matching Isaac)
  - the revolute axis is the joint-frame Z (the anchor orientation already aligns Z
    with the USD axis)

Then the 8 tilt joints are locked with pin.buildReducedModel, leaving a floating
base (6) + 8 arm joints. Inertias come from PhysX (what the simulator uses).

Exposes CasADi (cpin) functions for EE forward kinematics, frame Jacobians, the
mass matrix M(q) (crba), and the nonlinear terms h(q, v) (rnea). The robot model
comes ONLY from this USD.
"""
import json
import os

import casadi as ca
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin

_CONFIG = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, "config"))

TILT_JOINTS = [f"dof_{s}{p}{i}" for s in ("l", "r") for p in ("b", "f") for i in (1, 2)]
ARM_JOINTS = [f"dof_{s}{i}" for s in ("l", "r") for i in range(1, 5)]
EE_BODIES = {"ee_l": "l_link4_01", "ee_r": "r_link4_01"}
BASE_BODY = "base_link_01"
# Gripper PAD inner-face contact point relative to the link4 body frame, measured
# from the USD (the thin 239-vertex pad mesh; the inner face is the surface that
# touches the box, not the whole-assembly centroid). The link4 frame is the wrist;
# the EE is the pad surface that presses on the box. Derived by grasp_search.py.
EE_GRIPPER_OFFSET = {"ee_l": np.array([-0.3969, -0.067, 0.0]),
                     "ee_r": np.array([0.397, -0.067, 0.0])}


def short(path):
    return path.rsplit("/", 1)[-1] if path else path


def quat_wxyz_to_R(q):
    w, x, y, z = [float(v) for v in q]
    n = (w * w + x * x + y * y + z * z) ** 0.5
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def se3_from_local(pos, quat_wxyz):
    return pin.SE3(quat_wxyz_to_R(quat_wxyz), np.asarray(pos, float))


def load_extracted():
    with open(os.path.join(_CONFIG, "usd_model.json")) as f:
        usd = json.load(f)
    with open(os.path.join(_CONFIG, "isaac_model.json")) as f:
        isaac = json.load(f)
    return usd, isaac


class IsaacBodies:
    """PhysX mass / com / inertia and world rest pose, indexed by body name."""

    def __init__(self, isaac):
        self.names = [short(n) for n in isaac["body_names"]]
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.masses = np.asarray(isaac["masses"], float).reshape(-1)
        self.pos = np.asarray(isaac["body_pos_w"], float).reshape(len(self.names), 3)
        self.quat = np.asarray(isaac["body_quat_w"], float).reshape(len(self.names), 4)
        self._coms = np.asarray(isaac.get("coms", []), float).reshape(len(self.names), -1)
        self._inertias = np.asarray(isaac.get("inertias", []), float).reshape(len(self.names), -1)

    def inertia(self, name):
        i = self.idx[name]
        m = float(self.masses[i])
        com = self._coms[i][:3].copy() if self._coms.size else np.zeros(3)
        if self._inertias.size and self._inertias.shape[1] == 9:
            I = self._inertias[i].reshape(3, 3)
        elif self._inertias.size and self._inertias.shape[1] == 3:
            I = np.diag(self._inertias[i])
        else:
            I = np.eye(3) * 1e-4
        I = 0.5 * (I + I.T)
        return pin.Inertia(m, com, I)


ARM_TIP_JOINT = {"l_link4_01": "dof_l4", "r_link4_01": "dof_r4"}


def build_full_model(usd, isaac, add_frames=True):
    """Full Pinocchio model: free-flyer base + all 16 revolute joints (anchor-based).

    The revolute axis is -Z of the anchor frame: the rotation sign that matches
    Isaac/PhysX (and hence the gRITE controller) was confirmed by the FK
    cross-check against Isaac at a non-zero configuration. add_frames=False skips
    body frames so the model can be reduced without frames on the locked joints.
    """
    bodies = IsaacBodies(isaac)
    by_child = {short(j["body0"]): j for j in usd["joints"]}

    model = pin.Model()
    model.name = "daam"
    body_jid = {}
    body_anchor = {}  # world placement of each body's driving-joint anchor at q=0

    rid = model.addJoint(0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), "root_joint")
    model.appendBodyToJoint(rid, bodies.inertia(BASE_BODY), pin.SE3.Identity())
    if add_frames:
        model.addBodyFrame(BASE_BODY, rid, pin.SE3.Identity(), 0)
    body_jid[BASE_BODY] = rid
    body_anchor[BASE_BODY] = pin.SE3.Identity()

    pending = [short(j["body0"]) for j in usd["joints"]]
    guard = 0
    while pending and guard < 1000:
        guard += 1
        progressed = False
        for child in list(pending):
            j = by_child[child]
            parent = short(j["body1"])
            if parent not in body_jid:
                continue
            A1 = se3_from_local(j["local_pos1"], j["local_rot1"])  # anchor in parent frame
            A0 = se3_from_local(j["local_pos0"], j["local_rot0"])  # anchor in child frame
            joint_placement = body_anchor[parent].inverse() * A1
            jid = model.addJoint(body_jid[parent],
                                 pin.JointModelRevoluteUnaligned(np.array([0.0, 0.0, -1.0])),
                                 joint_placement, j["name"])
            model.appendBodyToJoint(jid, bodies.inertia(child), A0.inverse())
            if add_frames:
                model.addBodyFrame(child, jid, A0.inverse(), 0)
            body_jid[child] = jid
            body_anchor[child] = A1
            pending.remove(child)
            progressed = True
        if not progressed:
            raise RuntimeError(f"kinematic tree did not close; pending={pending}")

    if add_frames:
        for fname, body in EE_BODIES.items():
            A0 = se3_from_local(by_child[body]["local_pos0"], by_child[body]["local_rot0"])
            placement = A0.inverse() * pin.SE3(np.eye(3), EE_GRIPPER_OFFSET[fname])
            model.addFrame(pin.Frame(fname, body_jid[body], model.getFrameId(body),
                                     placement, pin.FrameType.OP_FRAME))
    return model, body_jid


def _add_ee_frames(model, usd):
    joints = {j["name"]: j for j in usd["joints"]}
    for fname, body in EE_BODIES.items():
        jn = ARM_TIP_JOINT[body]
        A0 = se3_from_local(joints[jn]["local_pos0"], joints[jn]["local_rot0"])
        placement = A0.inverse() * pin.SE3(np.eye(3), EE_GRIPPER_OFFSET[fname])
        model.addFrame(pin.Frame(fname, model.getJointId(jn), 0,
                                 placement, pin.FrameType.OP_FRAME))


TILT_LINKS = ["lb_link1_01", "lb_link2_01", "lf_link1_01", "lf_link2_01",
              "rb_link1_01", "rb_link2_01", "rf_link1_01", "rf_link2_01"]


def _combine_inertia(Ya, Yb):
    """Sum two pin.Inertia expressed in the same frame (rigidly attached at q=0)."""
    m = Ya.mass + Yb.mass
    c = (Ya.mass * Ya.lever + Yb.mass * Yb.lever) / m

    def about(Y):
        d = Y.lever - c
        return Y.inertia + Y.mass * ((d @ d) * np.eye(3) - np.outer(d, d))

    return pin.Inertia(m, c, about(Ya) + about(Yb))


def build_planning_model():
    """Free-flyer base + 8 arm joints. The tilt-chain links are folded into the
    base (rigidly fixed at tilt angle 0 for quadrotor mode; at q=0 all link frames
    coincide at the base origin). Built directly to avoid buildReducedModel, which
    aborts on this Pinocchio build with unaligned revolute joints."""
    usd, isaac = load_extracted()
    bodies = IsaacBodies(isaac)
    joints = {j["name"]: j for j in usd["joints"]}

    base_I = bodies.inertia(BASE_BODY)
    for ln in TILT_LINKS:
        base_I = _combine_inertia(base_I, bodies.inertia(ln))

    model = pin.Model()
    model.name = "daam_planning"
    body_anchor = {BASE_BODY: pin.SE3.Identity()}
    rid = model.addJoint(0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), "root_joint")
    model.appendBodyToJoint(rid, base_I, pin.SE3.Identity())
    model.addBodyFrame(BASE_BODY, rid, pin.SE3.Identity(), 0)
    body_jid = {BASE_BODY: rid}

    for jn in ARM_JOINTS:
        j = joints[jn]
        child, parent = short(j["body0"]), short(j["body1"])
        A1 = se3_from_local(j["local_pos1"], j["local_rot1"])
        A0 = se3_from_local(j["local_pos0"], j["local_rot0"])
        joint_placement = body_anchor[parent].inverse() * A1
        jid = model.addJoint(body_jid[parent],
                             pin.JointModelRevoluteUnaligned(np.array([0.0, 0.0, -1.0])),
                             joint_placement, jn)
        model.appendBodyToJoint(jid, bodies.inertia(child), A0.inverse())
        model.addBodyFrame(child, jid, A0.inverse(), 0)
        body_jid[child] = jid
        body_anchor[child] = A1

    _add_ee_frames(model, usd)
    return model


class WholeBody:
    """cpin wrapper: EE FK, frame Jacobians, M(q) and h(q, v) as CasADi functions."""

    def __init__(self, model=None):
        self.model = model if model is not None else build_planning_model()
        self.data = self.model.createData()
        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.ee_frame_ids = {n: self.model.getFrameId(n) for n in EE_BODIES}

        q = ca.SX.sym("q", self.nq)
        v = ca.SX.sym("v", self.nv)
        cpin.forwardKinematics(self.cmodel, self.cdata, q, v)
        cpin.updateFramePlacements(self.cmodel, self.cdata)
        ee = {n: self.cdata.oMf[fid].translation for n, fid in self.ee_frame_ids.items()}
        self.fk_ee = ca.Function("fk_ee", [q], [ee["ee_l"], ee["ee_r"]],
                                 ["q"], ["p_l", "p_r"])
        self.M = ca.Function("M", [q], [cpin.crba(self.cmodel, self.cdata, q)], ["q"], ["M"])
        self.h = ca.Function("h", [q, v],
                             [cpin.rnea(self.cmodel, self.cdata, q, v, ca.SX.zeros(self.nv))],
                             ["q", "v"], ["h"])
        J = {n: cpin.computeFrameJacobian(self.cmodel, self.cdata, q, fid,
                                          pin.LOCAL_WORLD_ALIGNED)
             for n, fid in self.ee_frame_ids.items()}
        self.J_ee = ca.Function("J_ee", [q], [J["ee_l"], J["ee_r"]], ["q"], ["J_l", "J_r"])

        # Lie-group integrator and difference, for multiple-shooting on the manifold.
        dv = ca.SX.sym("dv", self.nv)
        q2 = ca.SX.sym("q2", self.nq)
        self.integrate = ca.Function("integ", [q, dv],
                                     [cpin.integrate(self.cmodel, q, dv)], ["q", "dv"], ["qn"])
        self.difference = ca.Function("diff", [q, q2],
                                      [cpin.difference(self.cmodel, q, q2)], ["q", "q2"], ["d"])


if __name__ == "__main__":
    wb = WholeBody()
    print(f"planning model: nq={wb.nq} nv={wb.nv}  (expect nq=15 nv=14)")
    print(f"joints: {[wb.model.names[i] for i in range(wb.model.njoints)]}")
    print(f"total mass in model: {sum(I.mass for I in wb.model.inertias):.4f} kg")
    q0 = pin.neutral(wb.model)
    p_l, p_r = wb.fk_ee(q0)
    print(f"EE at neutral: ee_l={np.array(p_l).ravel().round(4).tolist()}  "
          f"ee_r={np.array(p_r).ravel().round(4).tolist()}")

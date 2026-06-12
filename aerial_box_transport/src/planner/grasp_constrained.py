"""Grasp-constrained transport planning (custom CBiRRT) for whole-body reconfiguration while holding
the box -- the foundation for "must change configuration to pass a narrow gap" research.

Instead of freezing the arm (which forbids reconfiguration), we plan the transport ON THE GRASP
CONSTRAINT MANIFOLD: the two pads stay at the box's two face contact points (pressed in by the
squeeze), exactly as the OCP pins them. The arm is FREE to reconfigure as long as the grip holds.

  Constraint  f(q) = 0   (co-dimension 3):
    f0 = (pr.x - pl.x) - sep     gripped width held (sep = box width - 2*penetration)
    f1 = pl.y - pr.y             pads directly across (aligned in y)
    f2 = pl.z - pr.z             pads aligned in z
  so the squeeze axis stays world-x and the box stays level; the box POSITION (pad midpoint) is free
  and the whole body reconfigures around it. (Box tilt = add box-orientation DOF + a relative-pose
  constraint; an extension.)

This set is a measure-zero manifold in the 14-DoF C-space, so ordinary RRT cannot sample it. We use
the standard remedy (Berenson's CBiRRT): sample in the ambient space and PROJECT each sample onto the
manifold by Newton iteration (q <- q - J^+ f), growing two trees that are projected at every step.
OMPL's own ProjectedStateSpace exists but its nanobind state objects are not settable from Python, so
we run the projection ourselves on top of the same FK / geometry used everywhere else.

Run (smoke test): conda run -n am_sampling python src/planner/grasp_constrained.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np  # noqa: E402

from planner.sampling_compare import Geometry, GEOM_TOL  # noqa: E402

_LO = np.array([-2.9, -1.3, -0.7, -0.25, -0.25, -0.25, -1.6, 0, 0, 0, -1.6, 0, 0, 0])
_HI = np.array([2.9, 1.3, 1.35, 0.25, 0.25, 0.25, 1.6, np.pi, np.pi, np.pi, 1.6, np.pi, np.pi, np.pi])


class GraspManifold:
    """The grasp constraint f(q)=0, its projection, and the (OCP) collision predicate."""

    def __init__(self, geom, sep):
        self.geom = geom
        self.sep = float(sep)

    def f(self, q):
        # grip contact (pads on the box faces) + parallel jaws (pads pressed FLAT, so the sim grip is
        # a clean face contact). dof2-dof3-dof4 = 0 per arm is the parallel-jaw condition. The arm can
        # still reconfigure via dof1 (reach) and the floating base, which is the carry freedom we want.
        pl, pr = self.geom.fk14(q)
        return np.array([(pr[0] - pl[0]) - self.sep, pl[1] - pr[1], pl[2] - pr[2],
                         q[7] - q[8] - q[9], q[11] - q[12] - q[13]])

    def jac(self, q, f0, eps=1e-6):
        J = np.empty((len(f0), 14))
        for i in range(14):
            dq = q.copy()
            dq[i] += eps
            J[:, i] = (self.f(dq) - f0) / eps
        return J

    def project(self, q, tol=1e-4, maxit=30):
        q = q.astype(float).copy()
        for _ in range(maxit):
            f0 = self.f(q)
            if np.linalg.norm(f0) < tol:
                return q
            q = q - np.linalg.pinv(self.jac(q, f0)) @ f0
        return q if np.linalg.norm(self.f(q)) < tol else None

    def valid(self, q):
        if np.any(q < _LO) or np.any(q > _HI):
            return False
        pl, pr = self.geom.fk14(q)
        base, box = q[0:3], 0.5 * (pl + pr)
        tol = -GEOM_TOL
        for c in self.geom.base_clear(base):
            if c < tol:
                return False
        for c in self.geom.ee_clear(pl) + self.geom.ee_clear(pr):
            if c < tol:
                return False
        for c in self.geom.keep_out(box):
            if c < tol:
                return False
        for c in self.geom.keep_out(base):
            if c < tol:
                return False
        return True


def _nearest(nodes, q):
    return int(np.argmin(np.sum((np.asarray(nodes) - q) ** 2, axis=1)))


def _branch(parent, nodes, i):
    out = []
    while i != -1:
        out.append(nodes[i])
        i = parent[i]
    return out


def plan_constrained(man, q_start, q_goal, timeout=40.0, delta=0.2, seed=7):
    """CBiRRT-Connect on the grasp manifold. Returns (path Nx14, info dict) or (None, info)."""
    rng = np.random.default_rng(seed)
    qs, qg = man.project(q_start), man.project(q_goal)
    if qs is None or qg is None or not man.valid(qs) or not man.valid(qg):
        return None, {"reason": "start/goal not on a valid manifold point"}

    A = {"nodes": [qs], "parent": [-1]}      # start tree
    B = {"nodes": [qg], "parent": [-1]}      # goal tree
    a_is_start = True
    t0 = time.time()
    n_proj = 0

    def extend(T, q_to):
        """One constrained step of T's nearest node toward q_to. Returns (status, idx)."""
        nonlocal n_proj
        i_near = _nearest(T["nodes"], q_to)
        q_near = T["nodes"][i_near]
        d = np.linalg.norm(q_to - q_near)
        reach = d < delta
        q_step = q_to if reach else q_near + delta * (q_to - q_near) / d
        n_proj += 1
        q_new = man.project(q_step)
        if q_new is None or not man.valid(q_new):
            return "trapped", i_near
        if np.linalg.norm(q_new - q_near) > 2.5 * delta:    # projection jumped too far
            return "trapped", i_near
        T["nodes"].append(q_new)
        T["parent"].append(i_near)
        return ("reached" if reach else "advanced"), len(T["nodes"]) - 1

    def connect(T, q_to):
        s, idx = "advanced", None
        for _ in range(200):
            s, idx = extend(T, q_to)
            if s != "advanced":
                break
        return s, idx

    while time.time() - t0 < timeout:
        q_rand = rng.uniform(_LO, _HI)
        s, ia = extend(A, q_rand)
        if s != "trapped":
            q_new = A["nodes"][ia]
            sc, ib = connect(B, q_new)
            if sc == "reached":
                ba = _branch(A["parent"], A["nodes"], ia)[::-1]   # rootA .. q_new
                bb = _branch(B["parent"], B["nodes"], ib)         # q_new .. rootB
                full = ba + bb[1:]
                path = np.array(full if a_is_start else full[::-1])
                info = {"nodes": len(A["nodes"]) + len(B["nodes"]), "projections": n_proj,
                        "time": time.time() - t0}
                return path, info
        A, B = B, A
        a_is_start = not a_is_start

    return None, {"reason": "timeout", "nodes": len(A["nodes"]) + len(B["nodes"]),
                  "projections": n_proj, "time": time.time() - t0}


def alignment_cost(geom, q):
    """OCP's w_align term as a per-state cost: squared horizontal box<->base offset. The box (pad
    midpoint) should sit under the base so the carried load barely shifts the whole-body CoM."""
    pl, pr = geom.fk14(q)
    mid = 0.5 * (pl + pr)
    return (mid[0] - q[0]) ** 2 + (mid[1] - q[1]) ** 2


def _cost_grad(geom, q, eps=1e-5):
    c0 = alignment_cost(geom, q)
    g = np.zeros(14)
    for i in range(14):
        dq = q.copy()
        dq[i] += eps
        g[i] = (alignment_cost(geom, dq) - c0) / eps
    return g


def optimize_alignment(man, path, iters=60, alpha=0.6, beta=0.25):
    """Plan-then-optimize (method 5): pull the box under the base by gradient descent on the
    box<->base horizontal offset, RE-PROJECTING onto the grasp manifold after each step (so the grip
    is preserved) and rejecting steps that collide. A light smoothness term keeps the path from
    kinking. Endpoints are fixed (the box must stay at pick / place). This is CHOMP-on-a-manifold and
    mirrors adding the OCP's w_align cost, but as a post-process on the sampled path."""
    P = path.copy()
    n = len(P)
    for _ in range(iters):
        newP = P.copy()
        for k in range(1, n - 1):
            g = _cost_grad(man.geom, P[k]) + beta * (2 * P[k] - P[k - 1] - P[k + 1])
            # Riemannian step: project the gradient onto the manifold TANGENT (remove the component
            # in the constraint's row space), else the projection back onto the manifold undoes it.
            J = man.jac(P[k], man.f(P[k]))
            g_tan = g - np.linalg.pinv(J) @ (J @ g)
            q = man.project(P[k] - alpha * g_tan)
            if q is not None and man.valid(q):
                newP[k] = q
        P = newP
    return P


def _offsets(geom, path):
    o = np.array([np.sqrt(alignment_cost(geom, q)) for q in path])
    return float(o.mean()), float(o.max())


def main():
    geom = Geometry()
    _, _, q_grasp, q_place = geom.configs()
    pl, pr = geom.fk14(q_grasp)
    sep = float(pr[0] - pl[0])
    man = GraspManifold(geom, sep)
    print(f"gripped width sep = {sep:.4f} m  (pads pressed onto the box faces; box held, not frozen)")

    print("planning grasp-constrained transport (grasp -> place) on the manifold ...")
    path, info = plan_constrained(man, q_grasp, q_place, timeout=40.0)
    print("info:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in info.items()})
    if path is None:
        print("FAILED")
        return
    m0, x0 = _offsets(geom, path)
    print(f"path: {len(path)} states;  box<->base offset BEFORE: mean {m0*100:.1f} cm, max {x0*100:.1f} cm")
    print("optimizing alignment (method 5: Riemannian gradient descent on offset) ...")
    opt = optimize_alignment(man, path)
    m1, x1 = _offsets(geom, opt)
    res = np.array([np.abs(man.f(q)) for q in opt])
    print(f"  AFTER: mean {m1*100:.1f} cm ({100*(1-m1/m0):.0f}% lower), max {x1*100:.1f} cm "
          f"(the max is the rack-threading point, where the box must stay offset)")
    print(f"  grip still held: residual max = {res.max(0).round(5).tolist()} m")
    print(f"  arm dof1 still reconfigures: range = {np.ptp(opt[:,6], axis=0):.2f} rad")


if __name__ == "__main__":
    main()

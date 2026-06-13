"""Turn a sampling-based GEOMETRIC path into an executable reference the IsaacSim tracker can run,
so the OCP and the sampling planner can be compared in the SAME simulator (the fair comparison).

The OMPL planner returns only a collision-free path in the 14-DoF configuration space (no timing,
velocities, or dynamics). To execute it we do the standard two-stage post-processing:
  1. plan the approach (home -> pregrasp) and the transport (grasp -> place) with RRTConnect,
  2. shortcut/simplify each path, stitch them with the grasp/release mode switches,
  3. time-parameterize onto the SAME phase schedule the tracker hardcodes (approach 3.0 s, grasp to
     4.6 s, transport to 9.6 s, release to 10.8 s) and lightly smooth the base so it is trackable,
  4. write results/sampling_reference.npz in the tracker's format (times, base, arm, grite_ref, box).

Then: conda run -n am_isaac python src/sim/track_reference.py --ref results/sampling_reference.npz --loop

Run: conda run -n am_sampling python src/planner/sampling_to_reference.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np  # noqa: E402
from ompl import base as ob  # noqa: E402
from ompl import geometric as og  # noqa: E402
from ompl import util as ou  # noqa: E402

from planner.sampling_compare import Geometry, _Checker, make_space, PLANNERS, GEOM_TOL  # noqa: E402
from planner.grasp_constrained import (GraspManifold, plan_constrained,  # noqa: E402
                                        optimize_alignment, _offsets)

try:
    ou.setLogLevel(ou.LogLevel.LOG_WARN)
except Exception:
    pass


def plan_path(geom, start, goal, carrying, planner="RRTConnect", timeout=20.0, dense=300):
    space = make_space(geom)
    ss = og.SimpleSetup(space)
    si = ss.getSpaceInformation()
    ss.setStateValidityChecker(_Checker(si, geom, carrying))
    si.setStateValidityCheckingResolution(0.002)
    st, gl = si.allocState(), si.allocState()
    for i in range(14):
        st[i], gl[i] = float(start[i]), float(goal[i])
    ss.setStartAndGoalStates(st, gl)
    ss.setPlanner(PLANNERS[planner](si))
    ss.solve(timeout)
    if not ss.haveExactSolutionPath():
        raise RuntimeError(f"{planner} failed to solve (carrying={carrying})")
    ss.simplifySolution(5.0)
    p = ss.getSolutionPath()
    p.interpolate(dense)
    return np.array([[p.getState(i)[j] for j in range(14)] for i in range(p.getStateCount())])


def plan_transport_base(geom, grasp_arm, start_base, goal_base, timeout=20.0, dense=300):
    """Plan the TRANSPORT as a BASE-only motion with the gripper FROZEN at the grasp: the object is
    rigidly attached and rides at a fixed offset below the base. This is the standard manipulation
    transfer model and it guarantees the grip is held -- a free 14-DoF carry lets the sampler open
    the jaws (the pad midpoint still clears obstacles) and drop the box. The box and both pads are
    rigid with the base, so we collision-check the whole cluster against the same OCP geometry."""
    space = ob.RealVectorStateSpace(3)
    b = ob.RealVectorBounds(3)
    lo, hi = [-2.9, -1.3, -0.7], [2.9, 1.3, 1.3]
    for i in range(3):
        b.setLow(i, lo[i])
        b.setHigh(i, hi[i])
    space.setBounds(b)
    ss = og.SimpleSetup(space)
    si = ss.getSpaceInformation()

    class _BaseVC(ob.StateValidityChecker):
        def __init__(self, si):
            super().__init__(si)

        def isValid(self, s):
            q = np.array([s[0], s[1], s[2], 0.0, 0.0, 0.0, *grasp_arm])
            pl, pr = geom.fk14(q)
            base, box = q[0:3], 0.5 * (pl + pr)
            tol = -GEOM_TOL
            for c in geom.base_clear(base):
                if c < tol:
                    return False
            for c in geom.ee_clear(pl) + geom.ee_clear(pr):
                if c < tol:
                    return False
            for c in geom.keep_out(box):
                if c < tol:
                    return False
            for c in geom.keep_out(base):
                if c < tol:
                    return False
            return True

    ss.setStateValidityChecker(_BaseVC(si))
    si.setStateValidityCheckingResolution(0.002)
    st, gl = si.allocState(), si.allocState()
    for i in range(3):
        st[i], gl[i] = float(start_base[i]), float(goal_base[i])
    ss.setStartAndGoalStates(st, gl)
    ss.setPlanner(og.RRTConnect(si))
    ss.solve(timeout)
    if not ss.haveExactSolutionPath():
        raise RuntimeError("transport (base-only, grip frozen) failed to solve")
    ss.simplifySolution(5.0)
    p = ss.getSolutionPath()
    p.interpolate(dense)
    bp = np.array([[p.getState(i)[j] for j in range(3)] for i in range(p.getStateCount())])
    return np.array([np.concatenate([b3, [0.0, 0.0, 0.0], grasp_arm]) for b3 in bp])


def resample(path, n):
    """Arc-length resample a config path to n knots (constant-speed in config space)."""
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    d = np.concatenate([[0.0], np.cumsum(seg)])
    if d[-1] < 1e-9:
        return np.repeat(path[:1], n, axis=0)
    s = np.linspace(0.0, d[-1], n)
    return np.array([np.interp(s, d, path[:, j]) for j in range(path.shape[1])]).T


def smooth(x, sigma=1.5):
    try:
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(x, sigma, axis=0, mode="nearest")
    except Exception:
        return x


def rotvec_to_R(th):
    """Rotation vector -> rotation matrix (Rodrigues)."""
    a = float(np.linalg.norm(th))
    if a < 1e-9:
        return np.eye(3)
    k = np.asarray(th) / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


def build_trajectory(geom, seed=7):
    """Plan the full sampler trajectory and return Q (109 x 14) configs [base3, theta3=0, arm8]:
    over-then-down approach + grasp-constrained transport (grip held, arm reconfigures) + alignment
    post-opt, stitched onto the 109-knot phase schedule. Reused by the tracker converter and the
    sampler->OCP hybrid (as the IPOPT warm-start)."""
    ou.RNG.setSeed(seed)
    q_home, q_pre, q_grasp, q_place = geom.configs()

    # APPROACH = over-then-down, so the open jaws descend VERTICALLY onto the box instead of sweeping
    # into it (a straight C-space line moves base + arm together and knocks the box). Decompose into:
    #   (1) a sampled free-space FLIGHT to a standoff directly ABOVE the grasp, with the arm already
    #       in the reach-down OPEN pose but high enough that the pads clear the box top, then
    #   (2) a prescribed VERTICAL descent (only base z changes, arm frozen) so the open pads drop
    #       straight down and straddle the box. This mirrors the OCP's descent constraint.
    q_stand = q_pre.copy()
    q_stand[2] = 0.0          # standoff: pregrasp x,y + arm, raised to the home flight height
    print("planning approach flight (home -> standoff directly above the box) ...")
    flight = plan_path(geom, q_home, q_stand, carrying=False)
    print(f"  flight path: {len(flight)} dense states; descent is a prescribed vertical drop")
    descent = resample(np.vstack([q_stand, q_pre]), 60)
    bad = [k for k in range(len(descent)) if not geom.valid(descent[k], carrying=False)]
    print(f"  vertical descent valid at {len(descent) - len(bad)}/{len(descent)} samples "
          f"(open jaws straddle the box)")

    # TRANSPORT on the GRASP CONSTRAINT MANIFOLD: the two pads stay on the box faces (flat, gripped),
    # but the ARM is free to reconfigure (dof1 reach + the floating base). The box is HELD, not frozen.
    # This is the model for whole-body reconfiguration while carrying, and the basis for narrow-gap work.
    pl_g, pr_g = geom.fk14(q_grasp)
    sep = float(pr_g[0] - pl_g[0])
    man = GraspManifold(geom, sep)
    print("planning transport (grasp-constrained; grip held, arm reconfigures) ...")
    pathB, info = plan_constrained(man, q_grasp, q_place, timeout=40.0)
    if pathB is None:
        raise RuntimeError(f"constrained transport failed: {info}")
    # method 5: pull the box under the base (OCP's w_align) by Riemannian gradient descent on the
    # box<->base offset, so the carried load shifts the CoM less (-> less base tilt in the sim).
    m0, _ = _offsets(geom, pathB)
    pathB = optimize_alignment(man, pathB)
    m1, _ = _offsets(geom, pathB)
    arm_range = np.ptp(pathB[:, 6:14], axis=0)
    box_final = 0.5 * (np.add(*geom.fk14(pathB[-1])))
    print(f"  transport path: {len(pathB)} states; arm reconfigures (dof1 range {arm_range[0]:.2f} rad)")
    print(f"  box<->base offset: mean {m0*100:.1f} -> {m1*100:.1f} cm ({100*(1-m1/m0):.0f}% lower); "
          f"box ends at {box_final.round(3).tolist()}")
    q_hold = pathB[-1].copy()                                # final held pose; tracker opens jaws

    # stitch onto the tracker's phase schedule (dt 0.1): approach 30 (20 flight + 10 vertical descent),
    # grasp 16, transport 50, release 13 knots -> 109 knots over [0, 10.8] s, matching the tracker.
    A = np.vstack([resample(flight, 20), resample(descent, 10)])   # flight, then vertical descent
    G = resample(np.vstack([q_pre, q_grasp]), 16)            # close the gripper (pregrasp .. grasp)
    T = resample(pathB, 50)                                  # base carries the rigid box (narrow passage)
    Rl = np.repeat(q_hold[None], 13, axis=0)                 # hold at place (tracker opens the gripper)
    Q = np.vstack([A, G, T, Rl])
    assert Q.shape[0] == 109, Q.shape
    return Q


def main():
    geom = Geometry()
    Q = build_trajectory(geom)
    dt = 0.1
    M = Q.shape[0]
    times = np.arange(M) * dt
    base = Q[:, 0:3].copy()
    base[:32] = smooth(base[:32], sigma=1.5)                 # smooth ONLY the approach (over-then-down);
    #   grasp/transport/release base stays ON the grasp manifold so the grip residual stays ~0
    arm = Q[:, 6:14].copy()                                  # arm joints (drive the gripper too)

    theta = Q[:, 3:6].copy()                                 # base attitude (rotvec) from the sampler
    omega = np.gradient(theta, dt, axis=0)                   # ~ angular velocity feedforward
    vel = np.gradient(base, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    jerk = np.gradient(acc, dt, axis=0)
    gr = np.zeros((M - 1, 30))
    gr[:, 0:3], gr[:, 3:6], gr[:, 6:9], gr[:, 9:12] = base[:-1], vel[:-1], acc[:-1], jerk[:-1]
    for k in range(M - 1):                                   # store the PLANNED base attitude (R, omega)
        gr[k, 12:21] = rotvec_to_R(theta[k]).flatten(order="F")
        gr[k, 21:24] = omega[k]

    # box centre per knot (FK pad midpoint); unused by the dynamic-box tracker but kept for format.
    box = np.array([0.5 * (np.add(*geom.fk14(Q[k]))) for k in range(M)])

    out = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir,
                                       "results", "sampling_reference.npz"))
    np.savez(out, times=times, base=base, arm=arm, grite_ref=gr, box=box)
    bz = base[:, 2]
    print(f"\nwrote {out}")
    print(f"  base z: {bz.min():+.2f} .. {bz.max():+.2f} m (home frame)  "
          f"max |base vel| = {np.abs(vel).max():.2f} m/s")
    print(f"  knots={M}, duration={times[-1]:.1f}s  "
          f"(approach<=3.0, grasp<=4.6, transport<=9.6, release<=10.8)")
    if geom.win:
        bad = sum(1 for k in range(M) if not (
            geom.window_clear(box[k], 0.0) and geom.window_clear(Q[k, 0:3], 0.0)
            and geom.window_clear(geom.fk14(Q[k])[0], 0.0)
            and geom.window_clear(geom.fk14(Q[k])[1], 0.0)))
        wb = box[(box[:, 0] > -1.1) & (box[:, 0] < -0.3)]
        print(f"  WINDOW self-check: {M - bad}/{M} knots clear (box+base+pads), "
              f"max |base tilt| = {np.degrees(np.abs(Q[:,3:6]).max()):.1f} deg; "
              f"box thru window z {wb[:,2].min():.2f}..{wb[:,2].max():.2f}" if len(wb) else "")


if __name__ == "__main__":
    main()

"""Sampling-based whole-body planner baseline, to compare against the OCP (paper).

The OCP (src/planner/ocp.py) finds a DYNAMICALLY feasible, contact/slip-aware whole-body
TRAJECTORY in one nonlinear solve. A sampling-based planner instead finds a collision-free
GEOMETRIC path in configuration space. We make the comparison apples-to-apples by giving the
sampler the SAME geometry the OCP sees: the bounding-box keep-out / base-clearance / EE-clearance
constraints (NOT mesh collision). What is compared is time-to-feasible on the identical
narrow-passage geometry; the sampler returns a path, the OCP a feasible trajectory through it.

Setup (per the user's choice): plan the FULL sequence as two queries in the FULL 14-DoF whole-body
configuration space q = [base_xyz(3), base_rotvec(3), arm(8)]:
  A. approach : q_home    -> q_pregrasp   (box on the desk, NOT carried)
  B. transport: q_grasp   -> q_place      (box CARRIED at the EE midpoint -- the narrow passage:
                                           up off the desk, over, down into the rack slot)
The grasp / release are mode switches (gripper close / open in place), not planned motions.

We sweep the OMPL planner zoo and report success rate + planning time per planner, so we can see
which planners clear the narrow passage and which fail (the paper's planner-type search).

Run (in the am_sampling env, which has OMPL + Pinocchio):
  conda run -n am_sampling python src/planner/sampling_compare.py --check
  conda run -n am_sampling python src/planner/sampling_compare.py --sweep --timeout 30 --trials 5
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np  # noqa: E402
import pinocchio as pin  # noqa: E402
from ompl import base as ob  # noqa: E402
from ompl import geometric as og  # noqa: E402
from ompl import util as ou  # noqa: E402

from config_io import load_config  # noqa: E402
from model.whole_body import WholeBody  # noqa: E402

try:
    ou.setLogLevel(ou.LogLevel.LOG_WARN)
except Exception:
    pass

# Planner name -> OMPL geometric class. RRT is the single-tree baseline expected to struggle in the
# narrow passage; RRTConnect / BKPIECE1 / LBKPIECE1 are the bidirectional narrow-passage workhorses.
PLANNERS = {
    "RRT": og.RRT, "RRTConnect": og.RRTConnect, "KPIECE1": og.KPIECE1,
    "BKPIECE1": og.BKPIECE1, "LBKPIECE1": og.LBKPIECE1, "PRM": og.PRM,
    "PRMstar": og.PRMstar, "BITstar": og.BITstar, "FMT": og.FMT, "BFMT": og.BFMT,
}

GEOM_TOL = 1e-3   # 1 mm slack so configs resting ON a surface (box on desk / shelf) count valid


def rotvec_to_quat_xyzw(theta):
    """Rotation vector -> quaternion (xyzw), matching ocp.theta_to_quat / Pinocchio order."""
    ang = float(np.linalg.norm(theta))
    if ang < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    ax = np.asarray(theta) / ang
    s = np.sin(0.5 * ang)
    return np.array([ax[0] * s, ax[1] * s, ax[2] * s, np.cos(0.5 * ang)])


class Geometry:
    """Numpy port of the OCP's geometric constraints (ocp.py): keep_out, base_clear, ee_clear,
    plus FK for the two gripper pads. A residual >= -GEOM_TOL means satisfied (collision-free)."""

    def __init__(self):
        robot, task = load_config()
        o = task["ocp"]
        self.o = o
        pg, gp = o["arm_pregrasp"], o["arm_grasp"]
        self.arm_pre = np.array([pg[0], pg[1], pg[2], pg[3], -pg[0], pg[1], pg[2], pg[3]])
        self.arm_grasp = np.array([gp[0], gp[1], gp[2], gp[3], -gp[0], gp[1], gp[2], gp[3]])
        self.arm_start = self.arm_pre.copy()
        self.arm_start[0] = self.arm_start[4] = 0.0   # dof1 = 0 (arm level) at home

        # Pinocchio model + a fast numpy FK for the two pad frames (reuse the OCP's WholeBody).
        self.wb = WholeBody()
        self.model = self.wb.model
        self.data = self.model.createData()
        self.fid_l = self.model.getFrameId("ee_l")
        self.fid_r = self.model.getFrameId("ee_r")

        self.pick = np.asarray(o["pick"], float)
        self.place = np.asarray(o["place"], float)

        # obstacle geometry (home frame = world - spawn z 1.5), identical to ocp.py
        box_half = 0.5 * float(task["box"]["size"][2])
        self.bm = 0.35
        self.sw = 0.04
        desk, rack = o["obstacles"]["desk"], o["obstacles"]["rack"]
        self.desk_fp = (desk["x"][0] - self.bm, desk["x"][1] + self.bm,
                        desk["y"][0] - self.bm, desk["y"][1] + self.bm)
        self.rack_fp = (rack["x"][0] - self.bm, rack["x"][1] + self.bm,
                        rack["y"][0] - self.bm, rack["y"][1] + self.bm)
        self.desk_top = desk["top"] - 1.5 + box_half
        self.rack_shelf = rack["top"] - 1.5 + box_half
        self.rack_clear = 2.0 - 1.5 + box_half + 0.1
        self.slot = (self.place[0] - 0.15, self.place[0] + 0.15, -0.35, 0.75)
        self.clr = o.get("base_clearance", 0.30)
        self.desk_surf = desk["top"] - 1.5
        self.rack_surf = rack["top"] - 1.5
        self.r_ee = o.get("ee_radius", 0.06)

    # --- FK: q14 (base xyz, rotvec, arm8) -> pad positions (numpy) ---
    def fk(self, base_theta6, arm8):
        q = np.empty(self.model.nq)
        q[0:3] = base_theta6[0:3]
        q[3:7] = rotvec_to_quat_xyzw(base_theta6[3:6])
        q[7:15] = arm8
        pin.framesForwardKinematics(self.model, self.data, q)
        return (np.array(self.data.oMf[self.fid_l].translation),
                np.array(self.data.oMf[self.fid_r].translation))

    def fk14(self, q14):
        return self.fk(q14[0:6], q14[6:14])

    @staticmethod
    def _sind(v, lo, hi, sw):
        return 0.5 * (np.tanh((v - lo) / sw) - np.tanh((v - hi) / sw))

    def keep_out(self, p):
        s = self.sw
        od = self._sind(p[0], self.desk_fp[0], self.desk_fp[1], s) * \
            self._sind(p[1], self.desk_fp[2], self.desk_fp[3], s)
        orr = self._sind(p[0], self.rack_fp[0], self.rack_fp[1], s) * \
            self._sind(p[1], self.rack_fp[2], self.rack_fp[3], s)
        ins = self._sind(p[0], self.slot[0], self.slot[1], s) * \
            self._sind(p[1], self.slot[2], self.slot[3], s)
        return (p[2] + 1.5 * (1 - od) - self.desk_top,
                p[2] + 1.5 * (1 - orr) - self.rack_shelf,
                p[2] + 1.5 * (1 - orr * (1 - ins)) - self.rack_clear)

    def base_clear(self, p):
        s = self.sw
        od = self._sind(p[0], self.desk_fp[0], self.desk_fp[1], s) * \
            self._sind(p[1], self.desk_fp[2], self.desk_fp[3], s)
        orr = self._sind(p[0], self.rack_fp[0], self.rack_fp[1], s) * \
            self._sind(p[1], self.rack_fp[2], self.rack_fp[3], s)
        return (p[2] + 1.5 * (1 - od) - (self.desk_surf + self.clr),
                p[2] + 1.5 * (1 - orr) - (self.rack_surf + self.clr))

    def ee_clear(self, p):
        s = self.sw
        od = self._sind(p[0], self.desk_fp[0], self.desk_fp[1], s) * \
            self._sind(p[1], self.desk_fp[2], self.desk_fp[3], s)
        orr = self._sind(p[0], self.rack_fp[0], self.rack_fp[1], s) * \
            self._sind(p[1], self.rack_fp[2], self.rack_fp[3], s)
        return (p[2] + 1.5 * (1 - od) - (self.desk_surf + self.r_ee),
                p[2] + 1.5 * (1 - orr) - (self.rack_surf + self.r_ee))

    def valid(self, q14, carrying):
        """True iff config q14 satisfies the same geometry the OCP enforces."""
        base = q14[0:3]
        pl, pr = self.fk14(q14)
        tol = -GEOM_TOL
        for c in self.base_clear(base):
            if c < tol:
                return False
        for c in self.ee_clear(pl) + self.ee_clear(pr):
            if c < tol:
                return False
        if carrying:
            box = 0.5 * (pl + pr)              # box rides at the EE midpoint (no-slip carry)
            for c in self.keep_out(box):
                if c < tol:
                    return False
            for c in self.keep_out(base):
                if c < tol:
                    return False
        return True

    # --- the four key configs (14-dim), taken from the OCP-discovered reference ---
    # The grasp/place poses are the SHARED task endpoints both planners connect. We take them from
    # the OCP reference (results/ocp_reference.npz) because they are valid, geometrically-correct
    # grasp/place configs (pads on the box faces, box at y=0). The literal arm_pregrasp/arm_grasp in
    # task.yaml are only OCP SEEDS (e.g. their dof1 differs from the discovered dof1~pi/2), so they
    # do NOT place the pads on the box; using them as static configs is wrong. theta (base attitude)
    # is not stored but stays <1 deg, so we set it to 0. Phase knots for N=108: approach 0..29,
    # grasp 30..45, transport 46..95, release 96..107, terminal 108.
    def configs(self):
        ref = os.path.abspath(os.path.join(
            os.path.dirname(__file__), os.pardir, os.pardir, "results", "ocp_reference.npz"))
        if not os.path.exists(ref):
            raise FileNotFoundError(
                f"{ref} not found. Run the OCP first:\n"
                f"  conda run -n am_dualarm python src/planner/ocp.py")
        d = np.load(ref)
        b, a = d["base"], d["arm"]

        def cfg(k):
            return np.concatenate([b[k], [0, 0, 0], a[k]])

        return cfg(0), cfg(29), cfg(45), cfg(108)   # home, pregrasp, grasp, place


# ---- OMPL plumbing ----
class _Checker(ob.StateValidityChecker):
    def __init__(self, si, geom, carrying):
        super().__init__(si)
        self.geom = geom
        self.carrying = carrying
        self._q = np.empty(14)

    def isValid(self, s):
        for i in range(14):
            self._q[i] = s[i]
        return self.geom.valid(self._q, self.carrying)


def make_space(geom):
    space = ob.RealVectorStateSpace(14)
    b = ob.RealVectorBounds(14)
    lo = [-2.9, -1.3, -0.7,  -0.25, -0.25, -0.25,  -1.6, 0, 0, 0,  -1.6, 0, 0, 0]
    hi = [2.9,  1.3,  1.35,   0.25,  0.25,  0.25,   1.6, np.pi, np.pi, np.pi,
          1.6, np.pi, np.pi, np.pi]
    for i in range(14):
        b.setLow(i, lo[i])
        b.setHigh(i, hi[i])
    space.setBounds(b)
    return space


def _path_is_valid(geom, path, carrying, n_dense=400):
    """Densely re-check the returned path against the geometry (honesty gate: OMPL only guarantees
    validity at its checking resolution; we confirm at finer granularity)."""
    p = og.PathGeometric(path)
    p.interpolate(n_dense)
    q = np.empty(14)
    for i in range(p.getStateCount()):
        s = p.getState(i)
        for j in range(14):
            q[j] = s[j]
        if not geom.valid(q, carrying):
            return False
    return True


def plan_once(geom, start14, goal14, carrying, planner_name, timeout):
    space = make_space(geom)
    ss = og.SimpleSetup(space)
    si = ss.getSpaceInformation()
    ss.setStateValidityChecker(_Checker(si, geom, carrying))
    si.setStateValidityCheckingResolution(0.002)
    st = si.allocState()
    gl = si.allocState()
    for i in range(14):
        st[i] = float(start14[i])
        gl[i] = float(goal14[i])
    ss.setStartAndGoalStates(st, gl)
    # Measure TIME-TO-FIRST feasible solution uniformly: asymptotically-optimal planners (PRMstar,
    # BITstar) otherwise run to the full timeout improving the path. A huge cost threshold makes them
    # stop at the first solution, like the feasibility planners (RRT, RRTConnect, ...) already do.
    obj = ob.PathLengthOptimizationObjective(si)
    obj.setCostThreshold(ob.Cost(1e9))
    ss.setOptimizationObjective(obj)
    ss.setPlanner(PLANNERS[planner_name](si))
    t0 = time.time()
    solved = ss.solve(float(timeout))
    dt = time.time() - t0
    exact = bool(ss.haveExactSolutionPath())
    if not exact:
        return {"ok": False, "t": dt, "states": 0, "len": float("nan"), "valid": False}
    ss.simplifySolution(2.0)
    path = ss.getSolutionPath()
    valid = _path_is_valid(geom, path, carrying)
    return {"ok": True, "t": dt, "states": path.getStateCount(),
            "len": path.length(), "valid": valid}


QUERIES = [
    ("A_approach", "q_home", "q_pregrasp", False),
    ("B_transport", "q_grasp", "q_place", True),
]


def check(geom):
    q_home, q_pre, q_grasp, q_place = geom.configs()
    named = {"q_home": q_home, "q_pregrasp": q_pre, "q_grasp": q_grasp, "q_place": q_place}
    print("=== config validity (same geometry as the OCP) ===")
    print("start/goal endpoints taken from results/ocp_reference.npz (home, pregrasp, grasp, place)")
    for nm, q in named.items():
        pl, pr = geom.fk14(q)
        box = 0.5 * (pl + pr)
        carry = nm in ("q_grasp", "q_place")
        print(f"  {nm:11s} valid(carry={carry!s:5s})={geom.valid(q, carry)!s:5s}  "
              f"base_z={q[2]:+.3f}  padL={pl.round(3).tolist()}  box={box.round(3).tolist()}")

    # straight-line baseline: is the naive linear interpolation collision-free? If not, the task
    # genuinely needs planning (and B has a narrow passage the straight line cannot pass).
    print("\n=== straight-line interpolation collision-free? (fraction of 400 samples valid) ===")
    for qname, s_nm, g_nm, carrying in QUERIES:
        a, b = named[s_nm], named[g_nm]
        ok = sum(geom.valid(a + (b - a) * t, carrying) for t in np.linspace(0, 1, 400)) / 400.0
        print(f"  {qname:12s} {s_nm} -> {g_nm}:  {ok*100:5.1f}% valid  "
              f"({'needs planning' if ok < 0.999 else 'trivially straight'})")

    print("\n=== quick narrow-passage test: RRTConnect on B (transport) ===")
    ou.RNG.setSeed(1)
    r = plan_once(geom, q_grasp, q_place, True, "RRTConnect", timeout=20.0)
    print(f"  RRTConnect: ok={r['ok']} t={r['t']:.3f}s states={r['states']} "
          f"len={r['len']:.3f} path_valid={r['valid']}")


def sweep(geom, planners, timeout, trials):
    q_home, q_pre, q_grasp, q_place = geom.configs()
    named = {"q_home": q_home, "q_pregrasp": q_pre, "q_grasp": q_grasp, "q_place": q_place}
    print(f"=== OMPL planner sweep  (timeout={timeout}s, trials={trials}) ===")
    print("query        planner       succ   t_med(s)  t_mean(s)  len_med  pathOK")
    results = {}
    for qname, s_nm, g_nm, carrying in QUERIES:
        for pname in planners:
            ts, oks, ls, good = [], 0, [], 0
            for tr in range(trials):
                r = plan_once(geom, named[s_nm], named[g_nm], carrying, pname, timeout)
                if r["ok"]:
                    oks += 1
                    ts.append(r["t"])
                    ls.append(r["len"])
                    good += int(r["valid"])
            succ = oks / trials
            tmed = np.median(ts) if ts else float("nan")
            tmean = np.mean(ts) if ts else float("nan")
            lmed = np.median(ls) if ls else float("nan")
            results[(qname, pname)] = (succ, tmed, tmean, lmed, good, oks)
            print(f"{qname:12s} {pname:12s} {succ*100:4.0f}%  {tmed:8.3f}  {tmean:8.3f}  "
                  f"{lmed:7.2f}  {good}/{oks}")
    return results


def save_results(results, timeout, trials, path):
    import json
    recs = []
    for (qname, pname), (succ, tmed, tmean, lmed, good, oks) in results.items():
        recs.append({"query": qname, "planner": pname, "success": succ,
                     "t_med": None if np.isnan(tmed) else tmed,
                     "t_mean": None if np.isnan(tmean) else tmean,
                     "len_med": None if np.isnan(lmed) else lmed,
                     "paths_valid": good, "n_success": oks})
    with open(path, "w") as f:
        json.dump({"timeout": timeout, "trials": trials, "records": recs}, f, indent=2)
    print(f"\nsaved -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="validate configs + one quick plan")
    ap.add_argument("--sweep", action="store_true", help="full planner sweep")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--planners", type=str, default="",
                    help="comma list; default = curated set")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="", help="save sweep results to this JSON path")
    args = ap.parse_args()

    # OMPL allows the global RNG seed to be set only ONCE, before any sampling. Set it here so the
    # whole sweep is reproducible; trials then differ because the global RNG keeps advancing.
    try:
        ou.RNG.setSeed(args.seed)
    except Exception:
        pass

    geom = Geometry()
    if args.check or not args.sweep:
        check(geom)
    if args.sweep:
        # default curated set: single-tree (RRT, KPIECE1), bidirectional (RRTConnect, BKPIECE1,
        # LBKPIECE1), roadmap (PRM, PRMstar), optimal (BITstar). FMT/BFMT are batch planners that
        # need separate sample-count tuning (they fail even on the free-space approach untuned), so
        # they are excluded from the default sweep but remain available via --planners.
        default = ["RRT", "RRTConnect", "KPIECE1", "BKPIECE1", "LBKPIECE1",
                   "PRM", "PRMstar", "BITstar"]
        planners = args.planners.split(",") if args.planners else default
        results = sweep(geom, planners, args.timeout, args.trials)
        if args.out:
            save_results(results, args.timeout, args.trials, args.out)


if __name__ == "__main__":
    main()

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

import coal  # noqa: E402   (hpp-fcl successor: GJK/EPA collision for capsule/box body model)
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

        # awning window (teammate's scene): a wall SOLID except the opening, plus a tilted sash slab.
        # Config values are WORLD; convert z to the home frame (-1.5). The box + drone must thread the
        # opening AND avoid the sash -> forces the diagonal-up reconfiguration. Sash modeled as one
        # tilted slab: hinge (top, wall side) -> bottom (out toward +x, down) at tilt_deg from vertical.
        w = o["obstacles"].get("window")
        self.win = w is not None
        if self.win:
            self.win_wx = (float(w["wall_x"][0]), float(w["wall_x"][1]))
            self.win_oy = (float(w["opening_y"][0]), float(w["opening_y"][1]))
            self.win_oz = (float(w["opening_z"][0]) - 1.5, float(w["opening_z"][1]) - 1.5)
            s = w["sash"]
            th = np.radians(s["tilt_deg"])
            hinge = np.array([s["hinge_x"], 0.0, s["hinge_z"] - 1.5])
            u = np.array([np.sin(th), 0.0, -np.cos(th)])      # hinge -> bottom (out +x, down)
            self.sash_c = hinge + 0.5 * s["slant"] * u        # slab centre
            self.sash_u, self.sash_v = u, np.array([0.0, 1.0, 0.0])
            self.sash_n = np.cross(u, self.sash_v)            # slab face normal (cos th, 0, sin th)
            self.sash_h = np.array([0.5 * s["slant"], 0.5 * s["width"], 0.5 * s["thickness"]])

        # COLLISION-BODY model (coal / hpp-fcl successor, GJK/EPA): the thin carbon-fibre arm links
        # are CAPSULES (link axis segment + true pipe radius = Minkowski sum of the segment with a
        # ball), NOT spheres -- a sphere set badly over-bounds a long thin pipe (radial slack + axial
        # over-cover) and falsely blocks the gap. The drone base is a flat BOX and the payload a BOX.
        # Collision vs the window = exact GJK distance from each body shape to the wall-border / sash
        # boxes; capsules are tight for cylinders, so a feasible diagonal-up squeeze is not blocked.
        # Chains use link1..link4..pad (all consecutive PHYSICAL links, so each pair is one capsule).
        self._chains = {"l": ["l_link1_01", "l_link2_01", "l_link3_01", "l_link4_01", "ee_l"],
                        "r": ["r_link1_01", "r_link2_01", "r_link3_01", "r_link4_01", "ee_r"]}
        self._coll_fids = {n: self.model.getFrameId(n)
                           for n in set(sum(self._chains.values(), []) + ["base_link_01"])}
        self.r_link = 0.03                                  # arm-pipe capsule radius (pipe + margin)
        self.base_box = (0.50, 0.50, 0.16)                  # drone platform full extents (flat slab)
        self.box_size = np.asarray(task["box"]["size"], float)   # payload box full extents
        self.win_margin = 0.0                               # extra gap (radii already carry a margin)
        if self.win:
            self._build_window_coal()

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

    def window_clear(self, p, r):
        """True if point p (clearance r, HOME frame) clears the window: passes through the wall
        opening (not into the solid wall) AND is outside the tilted sash slab."""
        if not self.win:
            return True
        if self.win_wx[0] - r <= p[0] <= self.win_wx[1] + r:          # inside the wall x-slab
            if not (self.win_oy[0] + r <= p[1] <= self.win_oy[1] - r and
                    self.win_oz[0] + r <= p[2] <= self.win_oz[1] - r):
                return False                                          # hits the solid wall
        d = p - self.sash_c                                          # tilted sash slab
        if (abs(d @ self.sash_u) <= self.sash_h[0] + r and
                abs(d @ self.sash_v) <= self.sash_h[1] + r and
                abs(d @ self.sash_n) <= self.sash_h[2] + r):
            return False                                              # hits the sash
        return True

    @staticmethod
    def _R_from_axis(d):
        """Orthonormal rotation whose 3rd column is unit(d) -- coal Capsule axis is its local z, so
        this orients a capsule along the link segment direction d."""
        z = d / np.linalg.norm(d)
        tmp = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = np.cross(tmp, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        return np.column_stack([x, y, z])

    def _build_window_coal(self):
        """Build the coal collision objects ONCE (constructing coal geoms is ~25x slower than mutating
        them, and this is the planner's inner loop). Obstacles (static): the wall is solid EXCEPT the
        opening (a hole, so non-convex) -> tile its frame with 4 axis-aligned border boxes (below /
        above / left / right) + the tilted sash box. Body (mutated per query in _update_body): the
        drone base box, one capsule per arm-link segment, and the payload box."""
        wx0, wx1 = self.win_wx
        oy0, oy1 = self.win_oy
        oz0, oz1 = self.win_oz
        WY, Z0, Z1 = 2.0, -1.5, 2.5          # wall outer extent (~4 m wall): y in [-2,2], z_home [-1.5,2.5]
        thick, cx = wx1 - wx0, 0.5 * (wx0 + wx1)

        def box(full, center, R=np.eye(3)):
            g = coal.Box(float(full[0]), float(full[1]), float(full[2]))
            return (g, coal.Transform3s(np.asarray(R, float), np.asarray(center, float)))

        self._win_obs = [
            box((thick, 2 * WY, oz0 - Z0), (cx, 0.0, 0.5 * (Z0 + oz0))),                       # below
            box((thick, 2 * WY, Z1 - oz1), (cx, 0.0, 0.5 * (oz1 + Z1))),                       # above
            box((thick, oy0 + WY, oz1 - oz0), (cx, 0.5 * (oy0 - WY), 0.5 * (oz0 + oz1))),      # left
            box((thick, WY - oy1, oz1 - oz0), (cx, 0.5 * (oy1 + WY), 0.5 * (oz0 + oz1))),      # right
            box(2 * self.sash_h, self.sash_c,                                                  # sash
                np.column_stack([self.sash_u, self.sash_v, self.sash_n])),
        ]
        self._dreq = coal.DistanceRequest()
        self._dres = coal.DistanceResult()

        # body objects, created once and mutated in place each query. Ordered fid chains for capsules.
        self._chain_fids = [[self._coll_fids[n] for n in ch] for ch in self._chains.values()]
        self._base_fid = self._coll_fids["base_link_01"]
        self._fid_l, self._fid_r = self._coll_fids["ee_l"], self._coll_fids["ee_r"]
        self._g_base, self._T_base = coal.Box(*self.base_box), coal.Transform3s()
        self._g_box, self._T_box = coal.Box(*self.box_size), coal.Transform3s()
        n_caps = sum(len(ch) - 1 for ch in self._chain_fids)
        self._caps = [coal.Capsule(self.r_link, 1.0) for _ in range(n_caps)]
        self._T_caps = [coal.Transform3s() for _ in range(n_caps)]
        self._body = ([(self._g_base, self._T_base)] + list(zip(self._caps, self._T_caps))
                      + [(self._g_box, self._T_box)])

    def _update_body(self):
        """Pose the precreated body coal objects from the current FK (call fk14 first). Base / payload
        boxes inherit the base attitude; each capsule spans one arm-link segment (length recomputed --
        the link frames are not all on the joint axes, so segment lengths vary with the arm pose)."""
        oMf = self.data.oMf
        bR = oMf[self._base_fid].rotation
        self._T_base.setRotation(bR)
        self._T_base.setTranslation(oMf[self._base_fid].translation)
        ci = 0
        for chain in self._chain_fids:
            prev = oMf[chain[0]].translation
            for f in chain[1:]:
                cur = oMf[f].translation
                d = cur - prev
                L = float(np.linalg.norm(d))
                self._caps[ci].halfLength = 0.5 * L
                self._T_caps[ci].setRotation(self._R_from_axis(d) if L > 1e-9 else np.eye(3))
                self._T_caps[ci].setTranslation(0.5 * (prev + cur))
                prev, ci = cur, ci + 1
        self._T_box.setRotation(bR)
        self._T_box.setTranslation(0.5 * (oMf[self._fid_l].translation + oMf[self._fid_r].translation))

    def window_clear_body(self, q14):
        """True if the WHOLE robot body (base box + arm-link capsules + payload box) clears the window
        (wall-border boxes + tilted sash), by coal GJK distance. Tight for the thin pipes, so a
        feasible diagonal-up squeeze is not falsely blocked the way an enclosing sphere set would be."""
        if not self.win:
            return True
        self.fk14(q14)                                               # FK populates self.data.oMf
        self._update_body()
        for bg, bT in self._body:
            for og_, oT in self._win_obs:
                self._dres.clear()        # GJK result is stateful: a stale penetration result poisons
                d = coal.distance(bg, bT, og_, oT, self._dreq, self._dres)   # the next query if reused
                if d < self.win_margin:
                    return False
        return True

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
        # window: the WHOLE body (base + arm links + pads + box, sphere set) must clear it
        if not self.window_clear_body(q14):
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
    lo = [-5.0, -1.3, -0.7,  -0.25, -0.25, -0.25,  -1.6, 0, 0, 0,  -1.6, 0, 0, 0]  # base x -> -5 (rack at -4)
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

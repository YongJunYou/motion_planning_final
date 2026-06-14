"""M4: multi-phase whole-body trajectory optimization (OCP), SE(3) base.

Acceleration-level control over the full floating base (SE(3): position + attitude,
via a rotation-vector / exponential-coordinates parameterization, all Euclidean
integration) and the 8 arm joints. The box is grasped by friction and CARRIED
(no-slip): during transport the box position equals the end-effector midpoint, so
lifting the box requires lifting the base/arms. The slip-aware condition is a hard
constraint: the squeeze (a smooth function of penetration, D2/D3) must exceed the
force needed so friction can hold the box, lambda >= F_n_required(a_z) (M5). This
is the no-slip limit of the free body in D6 and embeds the headline slip-aware law
directly into the planner. No complementarity. IPOPT via CasADi Opti.

Phases (D4, time-indexed cost FSM; dynamics + contact identical and smooth across
all phases): approach (EEs to pre-grasp), grasp (build the squeeze on the box at
rest on a support), transport (carry the box up to the lift height), release (box
placed on a shelf, squeeze relaxed). Squeeze axis is world x.

Run: conda run -n am_dualarm python src/planner/ocp.py
Outputs: results/ocp_reference.npz, results/ocp_phases.png
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import casadi as ca  # noqa: E402
import numpy as np  # noqa: E402

from config_io import load_config  # noqa: E402
from model.box import required_normal_force  # noqa: E402
from model.contact import smooth_normal_force  # noqa: E402
from model.whole_body import WholeBody  # noqa: E402
from planner.transcription import phase_schedule, smoothstep  # noqa: E402


def theta_to_quat(theta):
    """Rotation vector -> quaternion (xyzw), CasADi, smooth at theta = 0."""
    ang = ca.sqrt(ca.sumsqr(theta) + 1e-12)
    s = ca.sin(0.5 * ang) / ang
    return ca.vertcat(theta[0] * s, theta[1] * s, theta[2] * s, ca.cos(0.5 * ang))


def theta_to_R_ca(theta):
    """Rotation vector -> rotation matrix (CasADi, Rodrigues), smooth at theta = 0. Used to map the
    BASE-frame grip geometry into the world (the box reorients rigidly with the base)."""
    ang = ca.sqrt(ca.sumsqr(theta) + 1e-12)
    k = theta / ang
    K = ca.vertcat(ca.horzcat(0, -k[2], k[1]),
                   ca.horzcat(k[2], 0, -k[0]),
                   ca.horzcat(-k[1], k[0], 0))
    return ca.DM.eye(3) + ca.sin(ang) * K + (1 - ca.cos(ang)) * (K @ K)


def theta_to_R_np(theta):
    ang = float(np.linalg.norm(theta))
    if ang < 1e-9:
        return np.eye(3)
    ax = np.asarray(theta) / ang
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1.0 - np.cos(ang)) * (K @ K)


def _arc_resample(path, n):
    """Constant-arc-length resample of an (M,d) path to n knots."""
    path = np.asarray(path, float)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    d = np.concatenate([[0.0], np.cumsum(seg)])
    if d[-1] < 1e-9:
        return np.repeat(path[:1], n, axis=0)
    s = np.linspace(0.0, d[-1], n)
    return np.array([np.interp(s, d, path[:, j]) for j in range(path.shape[1])]).T


def _symmetrize_arm(A):
    """Project arm angles (n,8)=[l1..l4,r1..r4] onto the OCP's L/R mirror subspace (dof1_l=-dof1_r,
    dof_k_l=dof_k_r for k=2,3,4), so a (possibly asymmetric) sampler seed does not start IPOPT in
    violation of the symmetry equalities."""
    A = np.asarray(A, float)
    S = A.copy()
    d1 = 0.5 * (A[:, 0] - A[:, 4])
    S[:, 0], S[:, 4] = d1, -d1
    for i in (1, 2, 3):
        m = 0.5 * (A[:, i] + A[:, 4 + i])
        S[:, i], S[:, 4 + i] = m, m
    return S


def solve_ocp(verbose=False, seed=None, use_cylinders=True, window=False, transport_dur=None,
              window_mode="soft", keyframe=None, w_kf=300.0, w_place=0.0, w_level=0.0,
              w_padlevel=0.0, w_rise=0.0, warm=None):
    # window_mode: "soft" = point-sample body + soft penalty (the working baseline);
    #              "soft_box" = base+box as ANALYTIC oriented boxes (exact, no sampling), soft penalty;
    #              "hard_box" = same analytic body, but opening-containment is a HARD constraint (convex
    #                           corridor) while the non-convex sash stays a soft penalty.
    # keyframe: KEYFRAME-GUIDED mode (method 2). dict {"q14": (14,) config, "box_x": float}. When set,
    #           the sampler-route TRACKING is OFF and a soft waypoint cost (weight w_kf) pulls the window
    #           knot toward q14. The warm-start still comes from the (interpolated) seed -- no sampler.
    robot, task = load_config()
    c, sq, o = robot["contact"], robot["squeeze"], task["ocp"]
    g, m_o = task["gravity"], task["box"]["m_o"]
    k_c, eps, mu = o.get("k_planner", c["k"]), c["eps"], c["mu"]
    half, dt = o["half_extent"], o["dt"]
    durs = dict(o["durations"])
    if transport_dur is not None:            # slow the transport so the controller can track it better
        durs["transport"] = float(transport_dur)
    phase_of, N, times = phase_schedule(durs, dt)
    phase_bounds = np.array([durs["approach"], durs["approach"] + durs["grasp"],
                             durs["approach"] + durs["grasp"] + durs["transport"]])  # [appEnd,trStart,relStart]

    pg, gp = o["arm_pregrasp"], o["arm_grasp"]    # [dof1, dof2, dof3, dof4]; dof1 OPPOSITE on L/R
    arm_pre = np.array([pg[0], pg[1], pg[2], pg[3], -pg[0], pg[1], pg[2], pg[3]])
    arm_grasp = np.array([gp[0], gp[1], gp[2], gp[3], -gp[0], gp[1], gp[2], gp[3]])

    wb = WholeBody()

    def q_full(qr):
        return ca.vertcat(qr[0:3], theta_to_quat(qr[3:6]), qr[6:14])

    # The task is given by two box points: pick and place (in the home frame, where
    # home = the drone spawn). With the base level the EE midpoint sits a fixed
    # offset r_off in front of the base, so to grasp a box at `pick` the base must
    # hover at base_grasp = pick - r_off. The drone starts at home, flies to
    # base_grasp (approach), grasps, then carries the box to `place`.
    pl0, pr0 = wb.fk_ee(np.concatenate([[0, 0, 0], [0, 0, 0, 1], arm_grasp]))  # quat xyzw
    r_off = 0.5 * (np.array(pl0).ravel() + np.array(pr0).ravel())
    # The box-under-base alignment cost holds the base directly OVER the box, so the base
    # guide/seed should use ZERO forward (y) offset. Keeping arm_grasp's forward reach here
    # made base_ref swing out in y (0 -> 0.33 -> 0); the controller then lagged ~5 cm at the
    # grasp and the gripper closed just OFF the (small) box. Zero it so the base flies in
    # straight (the actual arm reach-down is still discovered, pads pinned to the box).
    r_off[1] = 0.0
    pick = np.asarray(o["pick"], float)
    place = np.asarray(o["place"], float)
    home = np.zeros(3)
    base_grasp = pick - r_off
    base_place = place - r_off

    ap = [k for k in range(N) if phase_of[k] == "approach"]
    tr = [k for k in range(N) if phase_of[k] == "transport"]

    # TASK objective: just get the box from pick to place. box_ref is a STRAIGHT line
    # (softly tracked). Obstacle avoidance is a HARD constraint (keep_out below), so the
    # OCP DISCOVERS the up-and-over path rather than it being prescribed.
    box_ref = np.tile(pick, (N + 1, 1))
    for j, k in enumerate(tr):
        box_ref[k + 1] = pick + (place - pick) * smoothstep((j + 1) / len(tr))
    for k in range(N + 1):
        if phase_of[min(k, N - 1)] == "release":
            box_ref[k] = place

    # feasible UP-AND-OVER initial guess (clears the rack) to seed IPOPT
    high_z = place[2] + o.get("place_rise", 0.6)
    lift_pt = np.array([pick[0], pick[1], high_z])
    over_pt = np.array([place[0], place[1], high_z])
    box_guess = np.tile(pick, (N + 1, 1))
    n_tr = len(tr)
    n1 = max(1, int(round(0.25 * n_tr)))
    n2 = max(1, int(round(0.50 * n_tr)))
    for j, k in enumerate(tr):
        if j < n1:
            box_guess[k + 1] = pick + (lift_pt - pick) * smoothstep((j + 1) / n1)
        elif j < n1 + n2:
            box_guess[k + 1] = lift_pt + (over_pt - lift_pt) * smoothstep((j + 1 - n1) / n2)
        else:
            box_guess[k + 1] = over_pt + (place - over_pt) * smoothstep(
                (j + 1 - n1 - n2) / (n_tr - n1 - n2))
    for k in range(N + 1):
        if phase_of[min(k, N - 1)] == "release":
            box_guess[k] = place
    # HYBRID (gentle handoff): replace the hand-crafted up-and-over box_guess with the SAMPLER's
    # discovered box route (the narrow-passage homotopy). base_ref / arm seed / velocities are then
    # built consistently around it below, so IPOPT starts dynamically sane (unlike a raw config seed).
    if seed is not None and "box" in seed:
        box_guess = np.asarray(seed["box"], float).copy()
    # WINDOW hybrid: the sampler hands off its discovered transport ROUTE (base SE(3) + arm + box).
    # Resample it onto the OCP transport knots; the box route drives box_guess (so box_ref_z / fn_set
    # / base_ref below are all consistent), and the base/attitude/arm seed is applied near set_initial.
    win_route = None
    if seed is not None and "q_route" in seed and "box_route" in seed:
        win_route = (_arc_resample(seed["q_route"], len(tr)),
                     _arc_resample(seed["box_route"], len(tr)))
        for j, k in enumerate(tr):
            box_guess[k + 1] = win_route[1][j]
    box_ref_z = box_guess[:, 2]   # use the realistic (up-and-over) z for the slip-aware a_z

    # OBSTACLE keep-out (home frame = world - spawn z 1.5). Footprints inflated by body
    # margin `bm` so the WHOLE drone+box (not just the box centre) stays clear. The rack
    # has a central landing SLOT where the box descends onto the top shelf; over the rest of
    # the rack footprint the box must clear the rack FRAME (-> climb up, descend in the slot),
    # and must never sink below the top shelf anywhere over the rack. keep_out is applied to
    # BOTH the box (PB) and the drone base.
    box_half = 0.5 * float(task["box"]["size"][2])
    bm = 0.35
    desk, rack = o["obstacles"]["desk"], o["obstacles"]["rack"]
    desk_fp = (desk["x"][0] - bm, desk["x"][1] + bm, desk["y"][0] - bm, desk["y"][1] + bm)
    desk_top = desk["top"] - 1.5 + box_half
    rack_fp = (rack["x"][0] - bm, rack["x"][1] + bm, rack["y"][0] - bm, rack["y"][1] + bm)
    rack_shelf = rack["top"] - 1.5 + box_half          # land on the top shelf
    rack_clear = 2.0 - 1.5 + box_half + 0.1            # above the rack frame (top = 2.0 world)
    slot = (place[0] - 0.15, place[0] + 0.15, -0.35, 0.75)   # narrow central landing column
    sw = 0.04                                          # sharper indicators -> less slot leakage

    def sind(v, lo, hi):                               # smooth in-[lo,hi] indicator in [0,1]
        return 0.5 * (ca.tanh((v - lo) / sw) - ca.tanh((v - hi) / sw))

    def keep_out(p):
        od = sind(p[0], desk_fp[0], desk_fp[1]) * sind(p[1], desk_fp[2], desk_fp[3])
        orr = sind(p[0], rack_fp[0], rack_fp[1]) * sind(p[1], rack_fp[2], rack_fp[3])
        ins = sind(p[0], slot[0], slot[1]) * sind(p[1], slot[2], slot[3])
        return [p[2] + 1.5 * (1 - od) - desk_top,                  # above the desk
                p[2] + 1.5 * (1 - orr) - rack_shelf,               # above top shelf over rack
                p[2] + 1.5 * (1 - orr * (1 - ins)) - rack_clear]   # above rack frame, not slot

    # base CLEARANCE: the drone BASE (not the arm) must hover at least `clr` above the
    # desk top and the rack top shelf whenever it is over their footprints, so it never
    # sinks toward the surface (large ground effect + collision risk from small errors).
    # The arm reaches DOWN to the box; only the base is held high. Same smooth-indicator
    # + big-M form as keep_out, but measured to the bare SURFACE (no box half-height).
    clr = o.get("base_clearance", 0.30)
    desk_surf = desk["top"] - 1.5                      # desk top surface (home frame)
    rack_surf = rack["top"] - 1.5                      # rack top shelf surface (home frame)

    def base_clear(p):
        od = sind(p[0], desk_fp[0], desk_fp[1]) * sind(p[1], desk_fp[2], desk_fp[3])
        orr = sind(p[0], rack_fp[0], rack_fp[1]) * sind(p[1], rack_fp[2], rack_fp[3])
        return [p[2] + 1.5 * (1 - od) - (desk_surf + clr),    # base >= desk top + clr
                p[2] + 1.5 * (1 - orr) - (rack_surf + clr)]   # base >= rack shelf + clr

    # EE pads must not dive INTO the desk/rack surface on the (near-vertical) descent to the
    # box. Model each pad as a small sphere (radius r_ee) and keep it above the surface over
    # the footprints. r_ee MUST be < the box half-height above the surface (0.057 m) so the
    # pad can still reach the box face at grip height. Same smooth-indicator + big-M form.
    r_ee = o.get("ee_radius", 0.04)

    def ee_clear(p):
        od = sind(p[0], desk_fp[0], desk_fp[1]) * sind(p[1], desk_fp[2], desk_fp[3])
        orr = sind(p[0], rack_fp[0], rack_fp[1]) * sind(p[1], rack_fp[2], rack_fp[3])
        return [p[2] + 1.5 * (1 - od) - (desk_surf + r_ee),    # pad sphere >= desk top
                p[2] + 1.5 * (1 - orr) - (rack_surf + r_ee)]   # pad sphere >= rack shelf

    # DISCOVERED up-over-down (NO path guide): the box CENTRE is confined to a narrow vertical
    # CYLINDER above the pick, and above the place, for the first `cyl_h` of height -> it must
    # rise / descend VERTICALLY there. Above the cylinder it is free, and keep_out makes it clear
    # the rack, so the optimizer DISCOVERS up-over-down. Only the box is constrained (the arm /
    # drone may be anywhere). Gated by the desk/rack FOOTPRINT and the HEIGHT (so the place
    # cylinder is inactive at the pick). Smooth indicator + big-M, like keep_out.
    cyl_r, cyl_M, cyl_h = 0.10, 25.0, 0.50
    x_mid = 0.5 * (pick[0] + place[0])    # midpoint x; pick side is x > x_mid, place side x < x_mid
    z_top_pick = desk_surf + cyl_h
    z_top_place = rack_surf + cyl_h

    def cylinder(p, cx, cy, z_top, side):
        # Gate by which SIDE of the midpoint the box is on (NOT a radius): the gate stays ~1 across
        # the whole narrow column, so the box cannot drift sideways out of it -- it must climb to
        # z_top to be released. side=+1 -> pick side (x>x_mid); side=-1 -> place side.
        gx = 0.5 * (1.0 + ca.tanh(side * (p[0] - x_mid) / 0.30))   # ~1 on this side of the midpoint
        sz = 0.5 * (1.0 + ca.tanh((p[2] - z_top) / 0.04))         # ~0 below z_top, ~1 above
        act = gx * (1.0 - sz)                                      # active: this side AND low
        d2 = (p[0] - cx) ** 2 + (p[1] - cy) ** 2
        return cyl_r ** 2 + cyl_M * (1.0 - act) - d2              # confine d2 <= cyl_r^2 when active

    # ---- AWNING WINDOW keep-out: a DIFFERENTIABLE surrogate of the sampler's coal capsule/box model
    # (IPOPT cannot use GJK). A set of body points (base, arm-link samples, pads, box) is kept out of
    # (a) the SOLID WALL -- "a point inside the wall's x-slab must lie inside the opening" (smooth
    # implication, big-M), and (b) the TILTED SASH slab -- "outside the box" via a smooth max of the
    # three (inflated) face distances. Each point carries a radius r so the finite samples cover the
    # real link/base/box thickness (Minkowski). Same role as keep_out, but for the window geometry.
    win = o["obstacles"].get("window") if window else None
    if win:
        from pinocchio import casadi as _cpin               # noqa: F401  (cpin already a dep)
        wwx = (float(win["wall_x"][0]), float(win["wall_x"][1]))
        woy = (float(win["opening_y"][0]), float(win["opening_y"][1]))
        woz = (float(win["opening_z"][0]) - 1.5, float(win["opening_z"][1]) - 1.5)
        s = win["sash"]
        s_th = np.radians(float(s["tilt_deg"]))
        s_hinge = np.array([float(s["hinge_x"]), 0.0, float(s["hinge_z"]) - 1.5])
        s_u = np.array([np.sin(s_th), 0.0, -np.cos(s_th)])     # hinge -> bottom (out +x, down)
        s_v = np.array([0.0, 1.0, 0.0])
        s_n = np.cross(s_u, s_v)                               # slab face normal
        s_c = s_hinge + 0.5 * float(s["slant"]) * s_u          # slab centre
        s_h = np.array([0.5 * float(s["slant"]), 0.5 * float(s["width"]), 0.5 * float(s["thickness"])])

        # CasADi FK for the collision frames (positions), built once from the cpin model.
        qsym = ca.SX.sym("qb", wb.nq)
        _cpin.forwardKinematics(wb.cmodel, wb.cdata, qsym)
        _cpin.updateFramePlacements(wb.cmodel, wb.cdata)
        _coll = ["base_link_01", "l_link1_01", "l_link2_01", "l_link3_01", "l_link4_01",
                 "r_link1_01", "r_link2_01", "r_link3_01", "r_link4_01", "ee_l", "ee_r"]
        _fexpr = [wb.cdata.oMf[wb.model.getFrameId(n)].translation for n in _coll]
        fk_body = ca.Function("fk_body", [qsym], _fexpr, ["q"], _coll)
        # pad (end-effector) ROTATIONS in WORLD, for leveling the grip at the place: the carried box's
        # orientation = the EE pad orientation (not the base), so leveling the base is not enough -- the
        # arm reaching down pitches the pads. At the level grip the EE-frame y-axis is vertical, so we
        # penalize its horizontal components to keep the pads (hence the box) flat for set-down.
        _Rexpr = [wb.cdata.oMf[wb.model.getFrameId(n)].rotation for n in ("ee_l", "ee_r")]
        fk_padR = ca.Function("fk_padR", [qsym], _Rexpr, ["q"], ["RL", "RR"])
        _idx = {n: i for i, n in enumerate(_coll)}
        _chains = [["l_link1_01", "l_link2_01", "l_link3_01", "l_link4_01", "ee_l"],
                   ["r_link1_01", "r_link2_01", "r_link3_01", "r_link4_01", "ee_r"]]
        r_link, r_base = 0.06, 0.18                   # arm-capsule / base inflation (Minkowski radius)
        box_h = 0.5 * np.asarray(task["box"]["size"], float)   # box half-extents (for oriented-box corners)
        r_corner = 0.02                               # small inflation on each box corner
        # (frame a, frame b, t, radius): points DENSELY along each a->b link segment, so the sphere set
        # covers the (thin) capsule continuously -- sparse samples leave gaps the true coal model catches.
        body_samples = [(_idx["base_link_01"], _idx["base_link_01"], 0.0, r_base)]
        for ch in _chains:
            for a, b in zip(ch[:-1], ch[1:]):
                for t in (0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0):
                    body_samples.append((_idx[a], _idx[b], float(t), r_link))
        # only constrain knots whose BOX is near the window in x: elsewhere the body is far and the
        # sash LSE arguments are metres (exp overflow) while the constraint is trivially satisfied.
        near_win = set(k for k in range(N) if -2.05 <= box_guess[min(k + 1, N)][0] <= 0.6)
        _BETA = 12.0                                   # smooth-max/min sharpness (softer = better conditioned)

        def _sind_w(v, lo, hi, sw=0.10):               # soft in-[lo,hi] indicator (window x-slab)
            return 0.5 * (ca.tanh((v - lo) / sw) - ca.tanh((v - hi) / sw))

        def _smin(a, b, beta=_BETA):
            return -1.0 / beta * ca.log(ca.exp(-beta * a) + ca.exp(-beta * b))

        def _sabs(x):
            return ca.sqrt(x * x + 1e-9)

        def _viol(c, d=1e-4):                          # smooth hinge ~ max(0, -c)^2 (penetration^2)
            return (0.5 * (ca.sqrt(c * c + d) - c)) ** 2

        def win_wall(p, r):
            # PENALTY (>=0) for a point INSIDE the wall x-slab but OUTSIDE the opening. The old form
            # 'smin(my,mz) + 8*(1-sx)' (a big-M "in-slab => in-opening" implication) was BLIND for this
            # thin slab: the slab is 0.10 m and the indicator smoothing was just as wide, so sx maxed at
            # ~0.6 and the 8*(1-sx) ~ 3 m term swamped the sub-metre violation -- a box sitting ENTIRELY
            # below the opening read as +3 m clear. Multiply instead: penalize the out-of-opening amount
            # scaled by how much the point is in the slab. A clear (in-opening) point gives _viol=0 either
            # way, so the working high-tilt passage is untouched; only true clips now register.
            sx = _sind_w(p[0], wwx[0] - r, wwx[1] + r, sw=0.04)     # sharp: ~1 inside the thin x-slab
            my = _smin(p[1] - (woy[0] + r), (woy[1] - r) - p[1])    # >=0 inside the opening (y)
            mz = _smin(p[2] - (woz[0] + r), (woz[1] - r) - p[2])    # >=0 inside the opening (z)
            return sx * _viol(_smin(my, mz))                       # in slab AND outside opening => penalty

        def win_sash(p, r, beta=_BETA):
            d = p - s_c
            du = _sabs(ca.dot(d, s_u)) - (s_h[0] + r)
            dv = _sabs(ca.dot(d, s_v)) - (s_h[1] + r)
            dn = _sabs(ca.dot(d, s_n)) - (s_h[2] + r)
            return 1.0 / beta * ca.log(ca.exp(beta * du) + ca.exp(beta * dv) + ca.exp(beta * dn))

        def window_points(qr, pb):
            P = fk_body(q_full(qr))                                # 11 frame positions
            pts = []
            for (ia, ib, t, r) in body_samples:
                pa = P[ia]
                pts.append((pa if ia == ib else pa + t * (P[ib] - pa), r))
            # the carried box as an ORIENTED box. A single centre sphere (r=0.08 < box half-diagonal
            # 0.109) under-bounded it; but sampling only the 8 CORNERS has a blind spot too -- the box
            # x-extent (0.106 m) ~ the wall x-thickness (0.10 m), so when the box straddles the wall plane
            # all 8 corners poke OUT of the wall's x-slab and the in-slab penalty never fires while the
            # body sits in the solid wall (coal saw a -10 cm clip the corner surrogate missed). Sample a
            # 3x3x3 grid instead (corners + edge mids + face centres + CENTRE): the centre / mid-x points
            # land inside the wall slab on a straddle, so the penalty sees the body, not just the corners.
            Rb = theta_to_R_ca(qr[3:6])
            for sx in (-1.0, 0.0, 1.0):
                for sy in (-1.0, 0.0, 1.0):
                    for sz in (-1.0, 0.0, 1.0):
                        c = pb + Rb @ ca.vertcat(sx * box_h[0], sy * box_h[1], sz * box_h[2])
                        pts.append((c, r_corner))
            return pts

        base_h = np.array([0.25, 0.25, 0.08])        # drone base box half-extents (0.5 x 0.5 x 0.16 m)

        def arm_only_points(qr):                      # arm-link capsule samples ONLY (base is analytic below)
            P = fk_body(q_full(qr))
            out = []
            for (ia, ib, t, r) in body_samples[1:]:
                out.append((P[ia] if ia == ib else P[ia] + t * (P[ib] - P[ia]), r))
            return out

        def _box_radius(Rb, h, n):
            # support half-width of an oriented box (rotation Rb, half-extents h) along world dir n:
            # sum_i h_i |(Rb^T n)_i|. Exact + differentiable -> the body needs no point sampling.
            bn = Rb.T @ n
            return h[0] * _sabs(bn[0]) + h[1] * _sabs(bn[1]) + h[2] * _sabs(bn[2])

        _EX = ca.DM([1.0, 0.0, 0.0]); _EY = ca.DM([0.0, 1.0, 0.0]); _EZ = ca.DM([0.0, 0.0, 1.0])
        _Smat = ca.horzcat(ca.DM(s_u.reshape(3, 1)), ca.DM(s_v.reshape(3, 1)), ca.DM(s_n.reshape(3, 1)))

        def win_obox(cc, Rb, h):
            # ANALYTIC oriented-box vs window. Returns opening-containment margins oy/oz (>=0 = box fully
            # inside the opening y/z), sx (box x-extent overlaps the wall slab -> gates the opening term),
            # and a sash separation margin (>0 = clear) via a 6-axis box-box SAT smooth-max (3 sash + 3
            # box face normals; less conservative than slab-axes-only).
            rx = _box_radius(Rb, h, _EX); ry = _box_radius(Rb, h, _EY); rz = _box_radius(Rb, h, _EZ)
            oy_lo = (cc[1] - ry) - woy[0]; oy_hi = woy[1] - (cc[1] + ry)
            oz_lo = (cc[2] - rz) - woz[0]; oz_hi = woz[1] - (cc[2] + rz)
            sx = _sind_w(cc[0], wwx[0] - rx, wwx[1] + rx, sw=0.04)
            d = cc - ca.DM(s_c.reshape(3, 1))
            seps = []
            for j in range(3):                                       # 3 sash face normals
                a = _Smat[:, j]
                seps.append(_sabs(ca.dot(d, a)) - (float(s_h[j]) + _box_radius(Rb, h, a)))
            for j in range(3):                                       # 3 box face normals
                a = Rb[:, j]
                sa = (float(s_h[0]) * _sabs(ca.dot(_Smat[:, 0], a))
                      + float(s_h[1]) * _sabs(ca.dot(_Smat[:, 1], a))
                      + float(s_h[2]) * _sabs(ca.dot(_Smat[:, 2], a)))
                seps.append(_sabs(ca.dot(d, a)) - (h[j] + sa))
            sash = 1.0 / _BETA * ca.log(sum(ca.exp(_BETA * sp) for sp in seps))
            return oy_lo, oy_hi, oz_lo, oz_hi, sx, sash

        _SN = ca.DM(s_n.reshape(3, 1))                       # sash normal (points room-ward + up)
        _SC = ca.DM(s_c.reshape(3, 1))

        def win_under(p, r):
            # KEEP the whole body UNDER the sash (below its lower face) within the sash's x-footprint, so
            # the drone threads the LOWER opening and climbs diagonally up UNDER the sash instead of
            # skimming the space above it. s_n points room-ward + up, so 'above the sash' is the +s_n side.
            gx = _sind_w(p[0], -1.10, -0.30, sw=0.12)        # ~1 only under the sash's x-footprint
            below = -(float(s_h[2]) + r) - ca.dot(p - _SC, _SN)   # >=0 when the point is below the lower face
            return gx * _viol(below)

        def win_under_box(cc, Rb, h):                        # analytic oriented box (base / payload)
            gx = _sind_w(cc[0], -1.10, -0.30, sw=0.12)
            rn = _box_radius(Rb, h, _SN)                      # box half-extent toward the sash normal
            below = -float(s_h[2]) - (ca.dot(cc - _SC, _SN) + rn)
            return gx * _viol(below)

    # base reference path (guide + soft target). APPROACH is over-then-DOWN: home (the
    # spawn) is already ABOVE the box, so no climb is needed; just traverse at the start
    # height to directly OVER the grasp hover, then descend VERTICALLY onto the box. The
    # open grippers come straight down around it instead of sweeping in horizontally
    # (which knocked the box off the desk). Then follow the carried box (base = box -
    # r_off) through the transport, hold at place.
    above_grasp = np.array([base_grasp[0], base_grasp[1], home[2]])
    n_ap = max(len(ap), 1)
    n_ap_down = max(1, int(round(0.45 * n_ap)))   # last ~45% of approach: vertical descent
    n_ap_over = n_ap - n_ap_down                   # first ~55%: traverse up to above the box
    ap_descend = set(ap[n_ap_over:])               # steps held over the box, descending in z
    base_ref = np.tile(base_grasp, (N + 1, 1))
    for j, k in enumerate(ap):
        if j < n_ap_over:
            base_ref[k] = home + (above_grasp - home) * smoothstep((j + 1) / n_ap_over)
        else:
            base_ref[k] = above_grasp + (base_grasp - above_grasp) * smoothstep(
                (j + 1 - n_ap_over) / n_ap_down)
    base_ref[0] = home
    for j, k in enumerate(tr):
        base_ref[k + 1] = box_guess[k + 1] - r_off
    for k in range(N + 1):
        if phase_of[min(k, N - 1)] == "release":
            base_ref[k] = base_place

    # the arm starts NOT lowered (dof1 = 0, level open gripper) at home and lowers to the
    # reach-down pre-grasp during the traverse (up high, before the vertical descent), so
    # the descent itself just brings the already-down open pads straight onto the box.
    arm_start = arm_pre.copy()
    arm_start[0] = arm_start[4] = 0.0                  # dof1 -> 0 on both arms (arm level)
    arm_ref_ap = np.tile(arm_pre, (N + 1, 1))
    for j, k in enumerate(ap):
        frac = smoothstep(min((j + 1) / n_ap_over, 1.0))   # 0 at home -> 1 by end of traverse
        arm_ref_ap[k] = arm_start + (arm_pre - arm_start) * frac
    arm_ref_ap[0] = arm_start

    # attitude + arm SEED arrays (default: level base, approach-then-grasp arm). The WINDOW hybrid
    # overrides the transport knots with the sampler's discovered base tilt + (symmetrized) arm
    # reconfiguration, and points base_ref at the sampler BASE (NOT box - r_off: the box rides off to
    # the side to thread the gap, so base != box - r_off there). This seeds the non-level homotopy.
    theta_seed = np.zeros((N + 1, 3))
    arm_seed_arr = np.array([arm_ref_ap[k] if phase_of[min(k, N - 1)] == "approach" else arm_grasp
                             for k in range(N + 1)], float)
    if win_route is not None:
        Qr = win_route[0]
        for j, k in enumerate(tr):
            base_ref[k + 1] = Qr[j, 0:3]
            theta_seed[k + 1] = Qr[j, 3:6]
            arm_seed_arr[k + 1] = Qr[j, 6:14]      # RAW asymmetric arm (window transport allows L!=R)

    fn_set = np.full(N + 1, float(sq["floor"]))
    for k in tr:
        a_z = (box_ref_z[k + 1] - 2 * box_ref_z[k] + box_ref_z[k - 1]) / dt ** 2
        fn_set[k] = max(required_normal_force(m_o, max(a_z, 0.0), mu, g, 2) + sq["margin"],
                        sq["floor"])

    _q_ident = ca.DM([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])    # base at identity (pos 0, quat xyzw)

    def fk_arm(arm):
        return wb.fk_ee(ca.vertcat(_q_ident, arm))           # pads in the BASE frame

    def contact(qr):
        """BASE-FRAME grip: the squeeze axis is the base x (NOT world x), so the grip is invariant to
        base attitude and the box reorients rigidly with the (omnidirectional) base. Pads via arm-only
        FK (base at identity); the world pads / box centre are recovered as base + R(theta) @ (.)."""
        pl_a, pr_a = fk_arm(qr[6:14])
        mid_a = 0.5 * (pl_a + pr_a)
        lam_l = smooth_normal_force((mid_a[0] - pl_a[0]) - half, k_c, eps, backend=ca)
        lam_r = smooth_normal_force((pr_a[0] - mid_a[0]) - half, k_c, eps, backend=ca)
        return pl_a, pr_a, mid_a, lam_l, lam_r

    opti = ca.Opti()
    QR = [opti.variable(14) for _ in range(N + 1)]   # [pos3, theta3, arm8]
    VR = [opti.variable(14) for _ in range(N + 1)]   # [linvel3, omega3, armvel8]
    PB = [opti.variable(3) for _ in range(N + 1)]    # box position (constrained per phase)
    AR = [opti.variable(14) for _ in range(N)]

    arm_lo = np.array([-1e3, 0, 0, 0, -1e3, 0, 0, 0.0])
    arm_hi = np.array([1e3, np.pi, np.pi, np.pi, 1e3, np.pi, np.pi, np.pi])
    w_base, w_f, w_box, w_hold = 50.0, 1.0, 200.0, 2.0
    w_reg_a, w_reg_v, w_sym, w_tilt = 1e-3, 1e-2, 2.0, 40.0
    w_yaw = 0.0       # DISABLED: a yaw penalty (tried 8, 15) drags the OCP into a low-tilt local min that
    #                   CLIPS the box -- the clearing solution uses a combined tilt+yaw whole-body
    #                   reconfiguration (seed carries yaw; penalizing the rotvec's yaw pulls tilt down too).
    #                   Box-clear must win. To reduce yaw cleanly would need a low-yaw seed, not a cost.
    w_yawrate = 10.0  # penalize the RATE of yaw (rotvec-z velocity VR[5]), NOT its value, to remove
    #                   gratuitous yaw (the trajectory wandered ~440 deg for a ~70 deg net turn). Applied
    #                   GLOBALLY: the window keep-out (win_wall) now correctly forces the box through the
    #                   opening, so minimizing yaw motion can no longer escape into a low-tilt clip. The
    #                   necessary threading yaw is held by the keep-out; only the gratuitous swing goes.
    #                   (Earlier, with win_wall blind to the wall clip, a global cost flipped to a clip
    #                   and this had to be gated off the passage; fixing win_wall made the gate moot.)
    w_align = 50.0    # box-under-base: penalize the horizontal box<->base offset (CoM keep)
    w_win = 60000.0   # window keep-out SOFT penalty (raised so box-clear dominates any local min)
    # WINDOW tracking: anchor the transport to the sampler's collision-free route (base path + attitude
    # + arm reconfiguration). Without it IPOPT wanders out of the sampler's good basin into ugly local
    # minima (fly over the wall, or tilt ~90 deg). With it the OCP REFINES the sampler motion into a
    # dynamically-feasible, slip-aware, smooth trajectory in the same homotopy.
    # track the sampler's BASE PATH + ARM (the collision-free homotopy) but NOT its attitude: the raw
    # CBiRRT seed has excess, run-dependent tilt (e.g. 84 deg early in transport), so let w_tilt
    # minimize the tilt freely while the window penalty still forces the tilt needed at the gap.
    w_trk, w_trk_th, w_trk_a = 80.0, 0.0, 20.0
    import os as _osw
    level_x0 = float(_osw.environ.get("W_LEVEL_X0", "-1.35"))  # box_x below which w_level levels the base
    #   (tilt+yaw -> 0). Default -1.35 (just past the wall). Push it MORE negative (e.g. -2.0) to DECOUPLE
    #   the window passage from the placement leveling, so each method keeps its NATIVE window homotopy
    #   (e.g. the sampler's yaw-sideways squeeze) instead of being pulled to a low-yaw pitch passage.
    kf_q = ca.DM(keyframe["q14"]) if keyframe is not None else None   # keyframe-guided
    kf_bx = float(keyframe["box_x"]) if keyframe is not None else None
    if keyframe is not None:
        import os as _os
        w_trk_th = float(_os.environ.get("W_TRK_TH", "80.0"))   # keyframe attitude tracking (env-tunable;
        #                   0 = soft_box-style base+arm-only tracking, attitude free; >0 follows the
        #                   keyframe's 60 deg tilt. Ramp this up (continuation) if a cold high value fails.
    v_max, a_max = (5.0, 40.0) if window else (3.0, 25.0)   # window threading is a faster maneuver

    opti.subject_to(QR[0] == np.concatenate([home, [0, 0, 0], arm_start]))  # arm level, not down
    opti.subject_to(VR[0] == np.zeros(14))

    cost = 0
    for k in range(N):
        ph = phase_of[k]
        pl_a, pr_a, mid_a, lam_l, lam_r = contact(QR[k])
        Rk = theta_to_R_ca(QR[k][3:6])
        base_k = QR[k][0:3]
        pl = base_k + Rk @ pl_a                                  # world-frame pads (clearance/descent)
        pr = base_k + Rk @ pr_a
        box_world = base_k + Rk @ mid_a                          # box centre in world (rides the grip)

        # robot integration (Euclidean; attitude is a rotation vector)
        opti.subject_to(VR[k + 1] == VR[k] + AR[k] * dt)
        opti.subject_to(QR[k + 1] == QR[k] + VR[k + 1] * dt)

        # box position by phase (no-slip carry). The box RIDES the base-frame grip: PB == the world
        # grip midpoint whenever gripped, pinned to pick/place on the supports. Pads sit ACROSS the
        # box faces (aligned in the base y,z); the squeeze (lambda) presses them along the base x. This
        # is attitude-invariant, so the base may tilt to thread the window while the grip stays held.
        if ph in ("approach", "grasp"):
            opti.subject_to(PB[k] == pick)                        # rests on the pick-up support
        elif ph == "release":
            opti.subject_to(PB[k] == place)                       # placed on the shelf
        if ph in ("grasp", "transport", "release"):
            opti.subject_to(PB[k] == box_world)                   # box held at the world grip midpoint
            opti.subject_to(pl_a[1] == pr_a[1])                   # pads across the box in base y
            opti.subject_to(pl_a[2] == pr_a[2])                   # pads across the box in base z
        if ph == "transport":
            opti.subject_to(lam_l >= fn_set[k])                   # slip-aware: enough squeeze to hold
            opti.subject_to(lam_r >= fn_set[k])
            for c in keep_out(PB[k]):                             # the box avoids desk/rack
                opti.subject_to(c >= 0)
            for c in keep_out(QR[k][0:3]):                        # the drone body avoids them too
                opti.subject_to(c >= 0)

        opti.subject_to(opti.bounded(arm_lo, QR[k][6:14], arm_hi))
        # keep the gripper jaws PARALLEL (pad faces along world x) at every step, so
        # closing from pre-grasp to grasp is a pure squeeze (only the gap shrinks, the
        # pads never rotate into the box). The pad facing angle is dof2 - dof3 - dof4
        # per arm, which is 0 at BOTH arm_pre and arm_grasp; hold it at 0 throughout.
        # k = 0 is already pinned by the initial condition, so skip it (no redundant row).
        if k >= 1:
            # L/R arm SYMMETRY (the two arms are mirror images): without it the optimizer can pick an
            # asymmetric pose during the open approach that still satisfies the midpoint-over-box pin.
            # BUT threading a tilted gap genuinely needs ASYMMETRIC arm reach (the sampler used it), so
            # for the window we DROP the mirror during transport/release and instead enforce parallel
            # jaws on BOTH arms independently (the grip stays flat while the arms reconfigure freely).
            if (not win) or ph in ("approach", "grasp"):
                opti.subject_to(QR[k][6] + QR[k][10] == 0.0)   # dof1_l = -dof1_r
                opti.subject_to(QR[k][7] == QR[k][11])         # dof2_l = dof2_r
                opti.subject_to(QR[k][8] == QR[k][12])         # dof3_l = dof3_r
                opti.subject_to(QR[k][9] == QR[k][13])         # dof4_l = dof4_r
                opti.subject_to(QR[k][7] - QR[k][8] - QR[k][9] == 0.0)  # parallel L (R by mirror)
            else:
                opti.subject_to(QR[k][7] - QR[k][8] - QR[k][9] == 0.0)     # parallel jaws, L
                opti.subject_to(QR[k][11] - QR[k][12] - QR[k][13] == 0.0)  # parallel jaws, R
        opti.subject_to(opti.bounded(-v_max, VR[k], v_max))
        opti.subject_to(opti.bounded(-a_max, AR[k], a_max))
        if window:
            # cap the base height so it THREADS the opening rather than flying OVER the (finite) wall:
            # the soft window penalty alone lets the base escape above the wall top, since nothing
            # models wall above z=2.5; this matches the sampler's base-z bound that forced threading.
            opti.subject_to(QR[k][2] <= 1.6)
        for c in base_clear(QR[k][0:3]):                      # base stays clr above surfaces
            opti.subject_to(c >= 0)
        for c in ee_clear(pl) + ee_clear(pr):                 # gripper pads stay above surfaces
            opti.subject_to(c >= 0)
        # AWNING WINDOW: keep the whole body (base + arm-link samples + pads) and the carried box out
        # of the solid wall (must pass through the opening) and the tilted sash, while threading
        # (transport) and settling behind the wall (release). Inactive elsewhere (points far in x).
        if win and k in near_win:
            if window_mode == "soft":
                for (pp, rr) in window_points(QR[k], PB[k]):
                    cost += w_win * (win_wall(pp, rr) + _viol(win_sash(pp, rr)) + win_under(pp, rr))
            else:                                       # soft_box / hard_box: analytic base + box
                for (pp, rr) in arm_only_points(QR[k]):                  # arm links: keep point-sampled
                    cost += w_win * (win_wall(pp, rr) + _viol(win_sash(pp, rr)) + win_under(pp, rr))
                Rk = theta_to_R_ca(QR[k][3:6])
                gx = float(box_guess[min(k + 1, N)][0])
                crossing = (wwx[0] - 0.12) <= gx <= (wwx[1] + 0.12)      # box at the wall plane (seed)
                for (cc, hh) in ((QR[k][0:3], base_h), (PB[k], box_h)):
                    oy_lo, oy_hi, oz_lo, oz_hi, sx, sash = win_obox(cc, Rk, hh)
                    cost += w_win * (_viol(sash) + win_under_box(cc, Rk, hh))   # outside slab AND below it
                    if window_mode == "hard_box" and crossing:          # HARD opening containment (convex)
                        opti.subject_to(oy_lo >= 0); opti.subject_to(oy_hi >= 0)
                        opti.subject_to(oz_lo >= 0); opti.subject_to(oz_hi >= 0)
                    else:                                                # soft opening (sx-gated to the slab)
                        cost += w_win * sx * (_viol(oy_lo) + _viol(oy_hi)
                                              + _viol(oz_lo) + _viol(oz_hi))
        # box rises / descends VERTICALLY in the pick / place cylinders (discovered up-over-down).
        # In the HYBRID (use_cylinders=False) we DROP these: the sampler's box route already supplies
        # the up-over-down homotopy, so the OCP needs only keep_out (real collision) + the costs. The
        # cylinders are this OCP's own hand-crafted path-shaping device, incompatible with the sampler
        # route, so seeding the cylinder-constrained OCP with that route makes IPOPT infeasible.
        if use_cylinders:
            opti.subject_to(cylinder(PB[k], pick[0], pick[1], z_top_pick, +1.0) >= 0)
            opti.subject_to(cylinder(PB[k], place[0], place[1], z_top_place, -1.0) >= 0)
        # WINDOW placement fix (SOFT penalties, so they never break feasibility; the hard corridor
        # destabilized IPOPT). Both are gated to the rack x-FOOTPRINT (|x - place_x| small), NOT to
        # the side of x_mid = -1.0 which coincides with the window -- so neither touches the tilted
        # window passage. Without them the box swings in from the side, tilted and yawed, and tips
        # when the terminal pin (PB[N]==place) snaps it to the shelf.
        if (w_place > 0.0 or w_level > 0.0 or w_padlevel > 0.0 or w_rise > 0.0) and ph in ("transport", "release"):
            g_rack = 0.5 * (ca.tanh((PB[k][0] - (place[0] - 0.5)) / 0.12)
                            - ca.tanh((PB[k][0] - (place[0] + 0.5)) / 0.12))   # ~1 over the rack x
            if w_rise > 0.0:                        # UP-AND-OVER: while approaching/over the rack but NOT
                # yet in the narrow descent column, keep the box ABOVE z_top_place so it rises over the
                # rack and drops straight down the column (vs the sampler's low diagonal that grazes the
                # shelf). keyframe already does this via its seed; this forces the sampler to as well.
                g_rise = 0.5 * (ca.tanh((PB[k][0] - (place[0] - 0.5)) / 0.12)
                                - ca.tanh((PB[k][0] - (place[0] + 1.0)) / 0.12))   # box_x in ~[-4.5,-3.0]
                d2c = (PB[k][0] - place[0]) ** 2 + (PB[k][1] - place[1]) ** 2
                in_col = 0.5 * (1.0 - ca.tanh((d2c - 0.04) / 0.02))   # ~1 within ~0.2 m of place (column)
                cost += w_rise * g_rise * (1.0 - in_col) * ca.fmax(0.0, z_top_place - PB[k][2]) ** 2
            if w_padlevel > 0.0 and win:           # LEVEL THE PADS (hence the rigidly-gripped box) at the
                # rack: box orientation = EE orientation, and the arm reaching down pitches the pads ~65
                # deg even with the base level. Penalize the EE-frame y-axis horizontal components (it is
                # vertical at the level grip); the wrist + base reconfigure to lower the box FLAT.
                _Rp = fk_padR(q_full(QR[k]))
                _RL, _RR = _Rp[0], _Rp[1]
                cost += w_padlevel * g_rack * (_RL[0, 1] ** 2 + _RL[1, 1] ** 2
                                               + _RR[0, 1] ** 2 + _RR[1, 1] ** 2)
            if w_place > 0.0:                       # pull the box DOWN a vertical column over `place`:
                g_low = 0.5 * (1.0 - ca.tanh((PB[k][2] - z_top_place) / 0.06))  # ~1 below z_top_place
                cost += w_place * g_rack * g_low * ((PB[k][0] - place[0]) ** 2
                                                    + (PB[k][1] - place[1]) ** 2)
            if w_level > 0.0:                       # LEVEL the base (tilt AND yaw -> 0) for the WHOLE
                # post-window flight, not just at the rack. Once the box clears the wall (x < ~-1.35)
                # the window pitch is no longer needed, so recover to level and HOLD it through the
                # flight + descent, so the rigidly-gripped box stays FLAT and lands on its bottom face.
                # Why this is needed: the box-under-base cost (w_align=50) ROLLS the base ~30 deg to
                # tuck the forward-reaching grip (mid_arm ~0.43 m forward) under the CoM, which plain
                # w_tilt (40) does not outvote -- so without this the base coasts tilted to the rack
                # and the box comes down on an edge. (yaw is otherwise unpenalized: w_yaw = 0.)
                g_post = 0.5 * (1.0 - ca.tanh((PB[k][0] - level_x0) / 0.15))   # ~1 once box_x < level_x0
                cost += w_level * g_post * ca.sumsqr(QR[k][3:6])

        if ph == "approach":
            # Hold the gripper OPEN (dof2-4) but leave dof1 FREE (arm-lowering discovered).
            cost += w_hold * (ca.sumsqr(QR[k][7:10] - arm_pre[1:4])     # left jaw stays open
                              + ca.sumsqr(QR[k][11:14] - arm_pre[5:8]))  # right jaw stays open
            # NO base_ref tracking: the approach FLIGHT is DISCOVERED. The alignment cost (below,
            # now all phases) pulls the base OVER the box and the grasp/EE constraints pull it
            # down; base_ref survives only as the initial-guess SEED.
            if k in ap_descend:
                # keep the OPEN gripper centred OVER the box during the descent so the wide pads
                # straddle it and the close is a pure x-squeeze (no forward sweep that pushes it).
                opti.subject_to(0.5 * (pl[0] + pr[0]) == PB[k][0])    # pad midpoint over box x
                opti.subject_to(0.5 * (pl[1] + pr[1]) == PB[k][1])    # pad midpoint over box y
        elif ph == "grasp":
            # base + arm are DISCOVERED to reach the pinned pad targets (above); no
            # base_grasp / arm_grasp tracking. Squeeze presses the pads into the +-x faces.
            cost += w_f * ((lam_l - o["squeeze_grasp"]) ** 2 + (lam_r - o["squeeze_grasp"]) ** 2)
            cost += w_sym * (lam_l - lam_r) ** 2
        elif ph == "transport":
            # NO box-path GUIDE: the up-over-down SHAPE is DISCOVERED from the pick/place
            # cylinders + keep-out (above) + effort minimization -- not tracked to box_guess.
            # hold a FIRM squeeze while carrying (so the sim friction grip holds); the
            # slip-aware value stays enforced as the lower bound (lam >= fn_set above).
            cost += w_f * ((lam_l - o["squeeze_grasp"]) ** 2 + (lam_r - o["squeeze_grasp"]) ** 2)
            cost += w_sym * (lam_l - lam_r) ** 2
        elif ph == "release":
            # base + arm DISCOVERED to hold the pads at the placed box; squeeze relaxed.
            cost += w_f * ((lam_l - sq["floor"]) ** 2 + (lam_r - sq["floor"]) ** 2)

        # BOX-UNDER-BASE alignment (soft): keep the carried object's centre on the vertical
        # line (global z) through the base centre, so the load barely shifts the whole-body
        # CoM and applies little disturbance torque on the (omnidirectional) base. NOT given
        # as a guide/waypoint -- the optimizer DISCOVERS the straight-down pick / carry from
        # this penalty on the horizontal box<->base offset, traded off against base_clear.
        # ALL phases: during the approach this also pulls the base toward over-the-box,
        # replacing the removed base_ref flight guide (so the approach flight is discovered).
        if win:
            # near the wall x the box MUST ride off to the side to thread the gap, so fade the
            # box-under-base alignment out there (else it fights the passage); full weight elsewhere.
            ag = 1.0 - sind(PB[k][0], wwx[0] - 0.7, wwx[1] + 0.7)
            cost += w_align * ag * ((PB[k][0] - QR[k][0]) ** 2 + (PB[k][1] - QR[k][1]) ** 2)
        else:
            cost += w_align * ((PB[k][0] - QR[k][0]) ** 2 + (PB[k][1] - QR[k][1]) ** 2)

        # WINDOW: TRACK the transport route (base + arm, and attitude in keyframe mode) so the OCP refines
        # that collision-free homotopy instead of wandering to an ugly / infeasible min. The route is the
        # sampler path (sampler-guided) OR the keyframe interpolation (keyframe-guided) -- same mechanism,
        # different source. This tracking is load-bearing: without it the window OCP wanders.
        if win and win_route is not None and ph == "transport":
            cost += (w_trk * ca.sumsqr(QR[k][0:3] - base_ref[k])
                     + w_trk_th * ca.sumsqr(QR[k][3:6] - theta_seed[k])
                     + w_trk_a * ca.sumsqr(QR[k][6:14] - arm_seed_arr[k]))
        # KEYFRAME-GUIDED: a light extra waypoint at the window moment sharpens the anchor on the keyframe
        # knot, on top of the full-route tracking above (the route already passes through the keyframe).
        if win and keyframe is not None and ph == "transport":
            if abs(float(box_guess[min(k + 1, N)][0]) - kf_bx) <= 0.10:
                cost += w_kf * (ca.sumsqr(QR[k][0:3] - kf_q[0:3])
                                + ca.sumsqr(QR[k][3:6] - kf_q[3:6])
                                + ca.sumsqr(QR[k][6:14] - kf_q[6:14]))

        # penalize only base TILT (pitch/roll = base z-axis off vertical), NOT yaw. With q = exp(theta),
        # qx^2+qy^2 = sin^2(tilt/2) is exactly the off-level angle and is yaw-invariant. So yaw / position
        # / arm stay cheap (only the effort reg below) and the optimizer reorients via yaw + arm, using
        # pitch/roll only when a constraint (e.g. a tilted gap) forces it.
        qd = theta_to_quat(QR[k][3:6])
        cost += w_tilt * (qd[0] ** 2 + qd[1] ** 2) + w_yaw * qd[2] ** 2
        cost += w_reg_a * ca.sumsqr(AR[k]) + w_reg_v * ca.sumsqr(VR[k])
        cost += w_yawrate * VR[k][5] ** 2          # minimize yaw motion (see w_yawrate note): the window
        #                                            keep-out holds the threading yaw, this drops the rest

    opti.subject_to(PB[N] == place)
    qdN = theta_to_quat(QR[N][3:6])
    cost += w_tilt * (qdN[0] ** 2 + qdN[1] ** 2) + w_yaw * qdN[2] ** 2
    opti.minimize(cost)

    # seed the full kinematic guess (pos + vel + acc consistent with base_ref) so
    # IPOPT starts near the solution; with metre-scale flights a zero-velocity guess
    # is too far and the solver fails restoration. HYBRID: the sampler handoff (seed["box"]) is
    # GENTLE -- it overrides only box_guess (the box route) earlier, so base_ref / arm seed /
    # velocities below are built consistently around the sampler route. A raw full-config geometric
    # seed instead fails (it violates the dynamics and the L/R symmetry; IPOPT stalls far from feasible).
    base_v = np.zeros((N + 1, 3))
    base_v[:-1] = (base_ref[1:] - base_ref[:-1]) / dt
    theta_v = np.zeros((N + 1, 3))
    theta_v[:-1] = (theta_seed[1:] - theta_seed[:-1]) / dt        # seed omega from the attitude route
    arm_v = np.zeros((N + 1, 8))
    arm_v[:-1] = (arm_seed_arr[1:] - arm_seed_arr[:-1]) / dt
    Vseed = [np.clip(np.concatenate([base_v[k], theta_v[k], arm_v[k]]), -v_max, v_max)
             for k in range(N + 1)]
    if warm is not None:
        # FULL warm-start (CONTINUATION): seed EVERY decision variable from a previously-converged
        # trajectory (base/theta/arm/box + finite-difference velocities), so IPOPT starts at inf_pr~0
        # and newly-added soft costs (w_place / w_level) only nudge the solution LOCALLY instead of
        # perturbing the cold path to feasibility out of the good basin. base is seeded from the ACTUAL
        # converged base (not box - r_off), which the seed route cannot supply.
        def _rs(a):                                            # resample to N+1 knots (linear in index)
            a = np.asarray(a, float)
            if len(a) == N + 1:
                return a
            xs = np.linspace(0.0, 1.0, len(a))
            return np.stack([np.interp(np.linspace(0, 1, N + 1), xs, a[:, j]) for j in range(a.shape[1])], 1)
        # NOTE: do NOT name these `wb` etc. -- `wb` is the WholeBody object used by fk_arm below.
        ws_b, ws_th, ws_a, ws_bx = (_rs(warm["base"]), _rs(warm["theta"]),
                                    _rs(warm["arm"]), _rs(warm["box"]))
        ws_q = np.concatenate([ws_b, ws_th, ws_a], axis=1)
        ws_v = np.zeros((N + 1, 14))
        ws_v[:-1, 0:3] = (ws_b[1:] - ws_b[:-1]) / dt
        ws_v[:-1, 3:6] = (ws_th[1:] - ws_th[:-1]) / dt
        ws_v[:-1, 6:14] = (ws_a[1:] - ws_a[:-1]) / dt
        ws_v = np.clip(ws_v, -v_max, v_max)
        for k in range(N + 1):
            opti.set_initial(QR[k], ws_q[k])
            opti.set_initial(PB[k], ws_bx[k])
            opti.set_initial(VR[k], ws_v[k])
        for k in range(N):
            opti.set_initial(AR[k], np.clip((ws_v[k + 1] - ws_v[k]) / dt, -a_max, a_max))
    else:
        for k in range(N + 1):
            opti.set_initial(QR[k], np.concatenate([base_ref[k], theta_seed[k], arm_seed_arr[k]]))
            opti.set_initial(PB[k], box_guess[k])
            opti.set_initial(VR[k], Vseed[k])
        for k in range(N):
            opti.set_initial(AR[k], np.clip((Vseed[k + 1] - Vseed[k]) / dt, -a_max, a_max))

    # warm-started CONTINUATION runs start feasible and only locally adjust, but the soft place/level
    # costs make IPOPT dither near the optimum (the dual never fully settles), so it can churn for
    # thousands of iters past a perfectly good feasible point. Cap it and loosen the ACCEPTABLE test so
    # it exits cleanly once feasible + near-optimal. Cold runs keep the strict defaults.
    # adaptive mu (default) DITHERS near the optimum on the warm-started continuation: its second-order
    # corrections overshoot (objective spikes) and never let the acceptable test trigger, so it churns
    # to max_iter past a good feasible point. monotone mu from a warm start near the optimum descends
    # smoothly and converges; cold runs keep adaptive (which needs it to reach feasibility from afar).
    _warm = warm is not None
    import os as _os2
    _warm_maxit = int(_os2.environ.get("WARM_MAXIT", "400"))   # monotone mu converges steadily (no
    #   dithering), so raise this when a warm run is still descending at the cap (e.g. a different
    #   homotopy that the warm-start is far from); 400 suffices for a local nudge.
    opti.solver("ipopt", {"print_time": False},
                {"max_iter": _warm_maxit if _warm else 5000, "tol": 1e-4,
                 "acceptable_tol": 1e-2 if _warm else 1e-3, "acceptable_iter": 8 if _warm else 15,
                 "print_level": 5 if verbose else 0,
                 "mu_strategy": "monotone" if _warm else "adaptive"})
    try:
        sol = opti.solve()
        status = "solved"
    except RuntimeError as exc:
        sol = opti.debug
        status = f"failed ({str(exc).splitlines()[-1][:60]})"

    val = sol.value
    base = np.array([val(QR[k])[0:3] for k in range(N + 1)])
    theta = np.array([val(QR[k])[3:6] for k in range(N + 1)])
    base_vel = np.array([val(VR[k])[0:3] for k in range(N + 1)])
    base_om = np.array([val(VR[k])[3:6] for k in range(N + 1)])
    base_acc = np.array([val(AR[k])[0:3] for k in range(N)])
    base_omdot = np.array([val(AR[k])[3:6] for k in range(N)])
    arm = np.array([val(QR[k])[6:14] for k in range(N + 1)])
    box = np.array([val(PB[k]) for k in range(N + 1)])
    lam = np.array([[float(val(contact(QR[k])[3])),
                     float(val(contact(QR[k])[4]))] for k in range(N)])
    # grippers are open and away from the box during approach: the 1D (x-only)
    # contact model reports a spurious value there, so zero it (no contact yet).
    for k in range(N):
        if phase_of[k] == "approach":
            lam[k] = 0.0

    jerk = np.zeros((N, 3))
    jerk[1:] = (base_acc[1:] - base_acc[:-1]) / dt
    omddot = np.zeros((N, 3))
    omddot[1:] = (base_omdot[1:] - base_omdot[:-1]) / dt
    Rcols = np.array([theta_to_R_np(theta[k]).flatten(order="F") for k in range(N)])
    grite_ref = np.concatenate([base[:N], base_vel[:N], base_acc, jerk, Rcols,
                                base_om[:N], base_omdot, omddot], axis=1)

    return {
        "status": status, "times": times, "base": base, "arm": arm, "box": box, "theta": theta,
        "phase_bounds": phase_bounds,
        "lam": lam, "fn_set": fn_set, "box_ref_z": box_ref_z, "box_ref": box_ref,
        "phase_of": phase_of, "pick": pick, "place": place, "base_ref": base_ref,
        "home": home, "grite_ref": grite_ref, "place_delta": place - pick,
        "lift": float(place[2] - pick[2]),
        "max_tilt_deg": float(np.degrees(np.max(np.linalg.norm(theta, axis=1)))),
    }


def plot_phases(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = res["times"]
    tc = 0.5 * (t[:-1] + t[1:])
    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    bounds, k0 = {}, 0
    for ph in ["approach", "grasp", "transport", "release"]:
        n = sum(1 for p in res["phase_of"] if p == ph)
        bounds[ph] = (t[k0], t[min(k0 + n, len(t) - 1)])
        k0 += n
    colors = {"approach": "0.92", "grasp": "#ffe8cc", "transport": "#d8f0d8", "release": "#f0d8d8"}
    for axi in ax:
        for ph, (a, b) in bounds.items():
            axi.axvspan(a, b, color=colors[ph], alpha=0.6)
    ax[0].plot(tc, res["lam"][:, 0], "C0-", lw=2, label="lambda_n left")
    ax[0].plot(tc, res["lam"][:, 1], "C1--", lw=2, label="lambda_n right")
    ax[0].plot(t, res["fn_set"], "k:", lw=1.5, label="slip-aware setpoint (min)")
    ax[0].set_ylabel("normal force [N]")
    ax[0].set_title("Multi-phase OCP (SE(3) base, carried box): force, box path, base path")
    ax[0].legend(loc="upper left", fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].plot(t, res["box"][:, 0], "C0-", lw=2, label="box x (lateral)")
    ax[1].plot(t, res["box"][:, 2], "C2-", lw=2, label="box z (height)")
    ax[1].plot(t, res["box_ref"][:, 0], "C0:", lw=1.2)
    ax[1].plot(t, res["box_ref"][:, 2], "k:", lw=1.2, label="reference")
    ax[1].set_ylabel("box position [m]")
    ax[1].legend(loc="upper left", fontsize=8)
    ax[1].grid(alpha=0.3)
    ax[2].plot(t, res["base"][:, 0], "C0-", lw=2, label="base x (lateral)")
    ax[2].plot(t, res["base"][:, 2], "C4-", lw=2, label="base z (height)")
    ax[2].set_ylabel("base position [m]")
    ax[2].set_xlabel("time [s]")
    ax[2].legend(loc="upper left", fontsize=8)
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    res = solve_ocp(verbose=True)
    rdir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "results"))
    os.makedirs(rdir, exist_ok=True)
    np.savez(os.path.join(rdir, "ocp_reference.npz"),
             times=res["times"], base=res["base"], arm=res["arm"], box=res["box"],
             lam=res["lam"], fn_set=res["fn_set"], box_ref_z=res["box_ref_z"],
             box_ref=res["box_ref"], grite_ref=res["grite_ref"])
    plot_phases(res, os.path.join(rdir, "ocp_phases.png"))
    lam, ph = res["lam"], res["phase_of"]
    gm = lam[[k for k, p in enumerate(ph) if p == "grasp"]].mean()
    tm = lam[[k for k, p in enumerate(ph) if p == "transport"]].mean()
    print(f"\n[OCP] status: {res['status']}")
    print(f"[OCP] peak lambda: {lam.max():.2f} N, mean grasp={gm:.2f} transport={tm:.2f} N")
    pick, place, home = res["pick"], res["place"], res["home"]
    print(f"[OCP] home {np.round(home, 2).tolist()} -> box pick "
          f"{np.round(pick, 2).tolist()} -> place {np.round(place, 2).tolist()} m")
    print(f"[OCP] base flies: x {res['base'][:, 0].min():.2f}..{res['base'][:, 0].max():.2f}, "
          f"y {res['base'][:, 1].min():.2f}..{res['base'][:, 1].max():.2f}, "
          f"z {res['base'][:, 2].min():.2f}..{res['base'][:, 2].max():.2f} m")
    print(f"[OCP] max base tilt: {res['max_tilt_deg']:.2f} deg")
    print(f"[OCP] figure -> {rdir}/ocp_phases.png")


if __name__ == "__main__":
    main()

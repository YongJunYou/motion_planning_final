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


def theta_to_R_np(theta):
    ang = float(np.linalg.norm(theta))
    if ang < 1e-9:
        return np.eye(3)
    ax = np.asarray(theta) / ang
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1.0 - np.cos(ang)) * (K @ K)


def solve_ocp(verbose=False, seed=None, use_cylinders=True, keyframe=None, warm=None, window=None):
    robot, task = load_config()
    c, sq, o = robot["contact"], robot["squeeze"], task["ocp"]
    g, m_o = task["gravity"], task["box"]["m_o"]
    k_c, eps, mu = o.get("k_planner", c["k"]), c["eps"], c["mu"]
    half, dt = o["half_extent"], o["dt"]
    phase_of, N, times = phase_schedule(o["durations"], dt)

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

    # AWNING WINDOW (the box must thread the opening). The wall normal is world x (teammate places
    # awing_window.usd at world (-1,0,0)); the opening is lateral world y in [-y_half, y_half] and
    # vertical from a sill up to z_top (world 2.06), with a slanted awning above. We DON'T model the
    # slant analytically yet -- we keep the box BELOW z_top (so it passes under the slant) and within
    # the lateral opening WHENEVER it is inside the wall slab (smooth gate gx ~1 near x_w). Same
    # big-M + smooth-indicator idiom as keep_out: each returned expr must be >= 0. All in HOME frame.
    # opening spans world z in [z_bot=2.06, z_top=3.07] (the slanted awning gap) and lateral
    # y in [-y_half, y_half]; the box must be INSIDE that rectangle when crossing the wall slab.
    WIN = {"x": -1.0, "half_thick": 0.025, "y_half": 0.865,
           "z_bot": 2.060 - 1.5, "z_top": 3.068 - 1.5, "clear": 0.04, "M": 6.0, "approach": 0.18}
    if isinstance(window, dict):
        WIN.update(window)

    def window_clear(p, obj_half):
        # gate ~1 while p.x is inside the wall slab (+ an approach margin so the squeeze into the
        # opening is gradual, not a step at the wall). Outside the slab the big-M slackens every row.
        gx = sind(p[0], WIN["x"] - WIN["half_thick"] - WIN["approach"],
                  WIN["x"] + WIN["half_thick"] + WIN["approach"])
        yh = WIN["y_half"] - obj_half - WIN["clear"]
        zt = WIN["z_top"] - obj_half - WIN["clear"]
        zb = WIN["z_bot"] + obj_half + WIN["clear"]
        slack = WIN["M"] * (1.0 - gx)
        return [(yh - p[1]) + slack,        # y <= +yh near the wall (within the opening, laterally)
                (p[1] + yh) + slack,        # y >= -yh
                (zt - p[2]) + slack,         # z <= opening top (below the slant top line)
                (p[2] - zb) + slack]         # z >= opening bottom (above z=2.06)

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

    fn_set = np.full(N + 1, float(sq["floor"]))
    for k in tr:
        a_z = (box_ref_z[k + 1] - 2 * box_ref_z[k] + box_ref_z[k - 1]) / dt ** 2
        fn_set[k] = max(required_normal_force(m_o, max(a_z, 0.0), mu, g, 2) + sq["margin"],
                        sq["floor"])

    # ---- KEYFRAME (optional through-config near the transport apex) -----------------------------
    # The caller hands ONE full config (base pos in the home frame, a base pitch, a 4-DoF arm) the
    # carry must PASS THROUGH (e.g. to thread the box through the awning window). Parsed UP-FRONT so
    # the cost loop can (a) RELAX the level-attitude and box-under-base penalties around the apex --
    # they actively fight a tilted, bent pose and were the root cause of the earlier divergence (the
    # seed got "flattened" back and IPOPT ran away to base z=6 m) -- and (b) add a STRONG waypoint
    # penalty pulling the config to the keyframe at the apex knot. A soft (not hard-equality) waypoint
    # is used on purpose: a hard QR[k]==config equality overlaps the L/R-symmetry + parallel-jaw
    # equalities already imposed at that knot (rank-deficient Jacobian -> IPOPT restoration failure).
    kf = None
    att_scale = np.ones(N + 1)
    align_scale = np.ones(N + 1)
    if keyframe is not None and len(tr) > 0:
        d = np.asarray(keyframe["arm"], float)                       # [dof1, dof2, dof3, dof4] per arm
        kf = {
            "base": np.asarray(keyframe["base"], float),             # [x, y, z], home frame
            "pitch": float(keyframe.get("pitch", 0.0)),              # rotation about world y [rad]
            "arm": np.array([d[0], d[1], d[2], d[3], -d[0], d[1], d[2], d[3]]),  # mirrored L/R
            "d": d,
            "w_wp": float(keyframe.get("w_wp", 600.0)),              # waypoint penalty weight
        }
        # PITCH is about the BODY X axis: the drone's forward is body -y, and pitch tilts that
        # forward direction up/down -> rotation about the lateral (x) axis, NOT world y. Pitching
        # about x keeps the two grippers (separated in x) at EQUAL z, so the level grip stays valid
        # (pitching about y split them in z and made the solve infeasible). Sign: nose-up lifts the
        # forward-carried box toward the elevated window.
        kf["theta"] = np.array([kf["pitch"], 0.0, 0.0])
        # apex = the transport knot whose base-x GUESS is nearest the keyframe x (the carry naturally
        # passes that lateral position then). base_ref during transport tracks the carried box.
        tr_arr = np.array(tr)
        kf["k_star"] = int(tr_arr[np.argmin(np.abs(base_ref[tr_arr + 1, 0] - kf["base"][0]))]) + 1
        kf["sigma"] = max(2.0, 0.18 * len(tr))                       # apex bump width in knots
        kf["bump"] = np.exp(-0.5 * ((np.arange(N + 1) - kf["k_star"]) / kf["sigma"]) ** 2)
        att_scale = 1.0 - 0.98 * kf["bump"]      # ~0 attitude penalty at the apex (allow the tilt)
        align_scale = 1.0 - 0.95 * kf["bump"]    # ~0 box-under-base penalty (allow the bent reach)

    def contact(qr, pb):
        pl, pr = wb.fk_ee(q_full(qr))
        lam_l = smooth_normal_force((pb[0] - pl[0]) - half, k_c, eps, backend=ca)
        lam_r = smooth_normal_force((pr[0] - pb[0]) - half, k_c, eps, backend=ca)
        return pl, pr, lam_l, lam_r

    opti = ca.Opti()
    QR = [opti.variable(14) for _ in range(N + 1)]   # [pos3, theta3, arm8]
    VR = [opti.variable(14) for _ in range(N + 1)]   # [linvel3, omega3, armvel8]
    PB = [opti.variable(3) for _ in range(N + 1)]    # box position (constrained per phase)
    AR = [opti.variable(14) for _ in range(N)]

    arm_lo = np.array([-1e3, 0, 0, 0, -1e3, 0, 0, 0.0])
    arm_hi = np.array([1e3, np.pi, np.pi, np.pi, 1e3, np.pi, np.pi, np.pi])
    w_base, w_f, w_box, w_hold = 50.0, 1.0, 200.0, 2.0
    w_reg_a, w_reg_v, w_sym, w_att = 1e-3, 1e-2, 2.0, 30.0
    w_align = 50.0    # box-under-base: penalize the horizontal box<->base offset (CoM keep)

    opti.subject_to(QR[0] == np.concatenate([home, [0, 0, 0], arm_start]))  # arm level, not down
    opti.subject_to(VR[0] == np.zeros(14))

    # COST is accumulated into NAMED groups (att / align / reg_acc / reg_vel / force / sym / hold)
    # so an IPOPT callback can record each term per iteration and we can graph which term drives the
    # objective (and blows it up when it does). `cost` is the real total handed to minimize().
    terms = {}

    def add(name, expr):
        terms[name] = terms.get(name, 0) + expr
        return expr

    cost = 0
    for k in range(N):
        ph = phase_of[k]
        pl, pr, lam_l, lam_r = contact(QR[k], PB[k])

        # robot integration (Euclidean; attitude is a rotation vector)
        opti.subject_to(VR[k + 1] == VR[k] + AR[k] * dt)
        opti.subject_to(QR[k + 1] == QR[k] + VR[k + 1] * dt)

        # box position by phase (carried with no slip during transport)
        if ph in ("approach", "grasp"):
            opti.subject_to(PB[k] == pick)                        # rests on the pick-up support
        elif ph == "transport":
            if kf is not None:
                # TILTED-grip transport: the box is just the EE midpoint in FULL 3D. The default
                # level-grip pinning (pl.z == pr.z == box.z, below) demands both pads share a height,
                # which a pitched base CANNOT satisfy (the two pads, separated in x by the grip, split
                # in z when the base tilts) -- that is exactly what made the keyframe solve infeasible
                # (inf_pr frozen at 7.51). Tying the box to the midpoint instead lets the grip tilt
                # with the base (the box has no modeled orientation, so the midpoint is the proxy).
                opti.subject_to(PB[k] == 0.5 * (pl + pr))
            else:
                opti.subject_to(PB[k][0] == 0.5 * (pl[0] + pr[0]))    # box x at the EE midpoint
            opti.subject_to(lam_l >= fn_set[k])                   # slip-aware: enough squeeze to hold
            opti.subject_to(lam_r >= fn_set[k])
            for c in keep_out(PB[k]):                             # the box avoids desk/rack
                opti.subject_to(c >= 0)
            for c in keep_out(QR[k][0:3]):                        # the drone body avoids them too
                opti.subject_to(c >= 0)
            if window is not None:
                for c in window_clear(PB[k], box_half):           # box threads the window opening
                    opti.subject_to(c >= 0)
                for c in window_clear(QR[k][0:3], 0.20):          # drone body clears the wall too
                    opti.subject_to(c >= 0)
        elif ph == "release":
            opti.subject_to(PB[k] == place)                       # placed on the shelf

        # GRASP DEFINITION (whole-body): the task gives WHERE each pad must be -- the centre
        # of the box's left / right face. Pin each pad's y,z to the box face centre (PB.y,
        # PB.z) in every gripping phase; the squeeze (lambda) presses them into the +-x faces.
        # So the OCP DISCOVERS the base + arm that reach these two pad targets, instead of
        # tracking a precomputed base_grasp / arm_grasp pose. (Box y,z then follow the pads.)
        # (transport under a keyframe uses the 3D midpoint tie above instead, to allow the tilt.)
        _level_grip = ph in ("grasp", "transport", "release") and not (ph == "transport" and kf is not None)
        if _level_grip:
            opti.subject_to(pl[1] == PB[k][1])                    # left pad at box face y
            opti.subject_to(pl[2] == PB[k][2])                    # left pad at box face z
            opti.subject_to(pr[1] == PB[k][1])                    # right pad at box face y
            opti.subject_to(pr[2] == PB[k][2])                    # right pad at box face z

        opti.subject_to(opti.bounded(arm_lo, QR[k][6:14], arm_hi))
        # keep the gripper jaws PARALLEL (pad faces along world x) at every step, so
        # closing from pre-grasp to grasp is a pure squeeze (only the gap shrinks, the
        # pads never rotate into the box). The pad facing angle is dof2 - dof3 - dof4
        # per arm, which is 0 at BOTH arm_pre and arm_grasp; hold it at 0 throughout.
        # k = 0 is already pinned by the initial condition, so skip it (no redundant row).
        if k >= 1:
            # L/R arm SYMMETRY (the two arms are mirror images): without it the optimizer can
            # pick an asymmetric pose during the open approach (left pad in front of the box,
            # right pad behind) that still satisfies the midpoint-over-box pin. Force the mirror.
            opti.subject_to(QR[k][6] + QR[k][10] == 0.0)   # dof1_l = -dof1_r
            opti.subject_to(QR[k][7] == QR[k][11])         # dof2_l = dof2_r
            opti.subject_to(QR[k][8] == QR[k][12])         # dof3_l = dof3_r
            opti.subject_to(QR[k][9] == QR[k][13])         # dof4_l = dof4_r
            # parallel jaws (pad faces along world x); the right arm follows by symmetry.
            opti.subject_to(QR[k][7] - QR[k][8] - QR[k][9] == 0.0)
        opti.subject_to(opti.bounded(-3.0, VR[k], 3.0))
        opti.subject_to(opti.bounded(-25.0, AR[k], 25.0))
        for c in base_clear(QR[k][0:3]):                      # base stays clr above surfaces
            opti.subject_to(c >= 0)
        for c in ee_clear(pl) + ee_clear(pr):                 # gripper pads stay above surfaces
            opti.subject_to(c >= 0)
        # box rises / descends VERTICALLY in the pick / place cylinders (discovered up-over-down).
        # In the HYBRID (use_cylinders=False) we DROP these: the sampler's box route already supplies
        # the up-over-down homotopy, so the OCP needs only keep_out (real collision) + the costs. The
        # cylinders are this OCP's own hand-crafted path-shaping device, incompatible with the sampler
        # route, so seeding the cylinder-constrained OCP with that route makes IPOPT infeasible.
        if use_cylinders:
            opti.subject_to(cylinder(PB[k], pick[0], pick[1], z_top_pick, +1.0) >= 0)
            opti.subject_to(cylinder(PB[k], place[0], place[1], z_top_place, -1.0) >= 0)

        if ph == "approach":
            # Hold the gripper OPEN (dof2-4) but leave dof1 FREE (arm-lowering discovered).
            cost += add("hold", w_hold * (ca.sumsqr(QR[k][7:10] - arm_pre[1:4])     # left jaw open
                              + ca.sumsqr(QR[k][11:14] - arm_pre[5:8])))  # right jaw stays open
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
            cost += add("force", w_f * ((lam_l - o["squeeze_grasp"]) ** 2 + (lam_r - o["squeeze_grasp"]) ** 2))
            cost += add("sym", w_sym * (lam_l - lam_r) ** 2)
        elif ph == "transport":
            # NO box-path GUIDE: the up-over-down SHAPE is DISCOVERED from the pick/place
            # cylinders + keep-out (above) + effort minimization -- not tracked to box_guess.
            # hold a FIRM squeeze while carrying (so the sim friction grip holds); the
            # slip-aware value stays enforced as the lower bound (lam >= fn_set above).
            cost += add("force", w_f * ((lam_l - o["squeeze_grasp"]) ** 2 + (lam_r - o["squeeze_grasp"]) ** 2))
            cost += add("sym", w_sym * (lam_l - lam_r) ** 2)
        elif ph == "release":
            # base + arm DISCOVERED to hold the pads at the placed box; squeeze relaxed.
            cost += add("force", w_f * ((lam_l - sq["floor"]) ** 2 + (lam_r - sq["floor"]) ** 2))

        # BOX-UNDER-BASE alignment (soft): keep the carried object's centre on the vertical
        # line (global z) through the base centre, so the load barely shifts the whole-body
        # CoM and applies little disturbance torque on the (omnidirectional) base. NOT given
        # as a guide/waypoint -- the optimizer DISCOVERS the straight-down pick / carry from
        # this penalty on the horizontal box<->base offset, traded off against base_clear.
        # ALL phases: during the approach this also pulls the base toward over-the-box,
        # replacing the removed base_ref flight guide (so the approach flight is discovered).
        cost += add("align", w_align * align_scale[k]
                    * ((PB[k][0] - QR[k][0]) ** 2 + (PB[k][1] - QR[k][1]) ** 2))

        cost += add("att", w_att * att_scale[k] * ca.sumsqr(QR[k][3:6]))   # attitude -> level
        cost += add("reg_acc", w_reg_a * ca.sumsqr(AR[k]))
        cost += add("reg_vel", w_reg_v * ca.sumsqr(VR[k]))

        # KEYFRAME waypoint: strong penalty pulling the apex config to the keyframe (base pos +
        # pitch + arm). att/align are relaxed here (above) so this does not fight them.
        if kf is not None and k == kf["k_star"]:
            # base POSITION + ATTITUDE only. The ARM is NOT referenced: forcing the arm joints to a
            # fixed config fights the grasp pinning and makes the sim drop the box. The arm is left
            # free to whatever holds the grip while the base reaches the through-window pose.
            cost += add("waypoint", kf["w_wp"] * (
                ca.sumsqr(QR[k][0:3] - kf["base"])
                + ca.sumsqr(QR[k][3:6] - kf["theta"])))

    opti.subject_to(PB[N] == place)
    cost += add("att", w_att * att_scale[N] * ca.sumsqr(QR[N][3:6]))
    opti.minimize(cost)

    # record each named cost term per IPOPT iteration (diagnostic: which term drives / blows up the
    # objective). opti.debug.value(expr) evaluates at the current iterate inside the callback.
    cost_hist = {name: [] for name in terms}
    cost_hist["total"] = []
    _term_exprs = dict(terms)

    def _record_costs(i):
        try:
            cost_hist["total"].append(float(opti.debug.value(cost)))
            for nm, ex in _term_exprs.items():
                cost_hist[nm].append(float(opti.debug.value(ex)))
        except Exception:
            pass

    opti.callback(_record_costs)

    # seed the full kinematic guess (pos + vel + acc consistent with base_ref) so
    # IPOPT starts near the solution; with metre-scale flights a zero-velocity guess
    # is too far and the solver fails restoration. HYBRID: the sampler handoff (seed["box"]) is
    # GENTLE -- it overrides only box_guess (the box route) earlier, so base_ref / arm seed /
    # velocities below are built consistently around the sampler route. A raw full-config geometric
    # seed instead fails (it violates the dynamics and the L/R symmetry; IPOPT stalls far from feasible).
    # build the per-knot arm + attitude SEED arrays (theta is the rotation-vector attitude,
    # normally seeded level = 0). Splitting them out (vs the old inline [0,0,0]/arm_grasp) lets
    # an optional KEYFRAME blend a tilted, specific config into the transport apex below.
    theta_seed = np.zeros((N + 1, 3))
    arm_seed = np.array([arm_ref_ap[k] if phase_of[min(k, N - 1)] == "approach" else arm_grasp
                         for k in range(N + 1)])

    # KEYFRAME warm-start: blend the keyframe config (parsed up-front into `kf`) into
    # base_ref / theta_seed / arm_seed with the same smooth Gaussian apex bump, so the velocity/accel
    # guess (rebuilt from base_ref below) stays dynamically consistent. The waypoint PENALTY + the
    # apex att/align relaxation (both set up before the cost loop) do the actual pulling; this just
    # gives IPOPT a warm start already near the tilted, bent apex pose so it converges there.
    if kf is not None:
        bump = kf["bump"]
        for k in range(N + 1):
            w = float(bump[k])
            if w < 1e-3:
                continue
            base_ref[k] = (1.0 - w) * base_ref[k] + w * kf["base"]
            theta_seed[k] = (1.0 - w) * theta_seed[k] + w * kf["theta"]
            # NOTE: the arm is NOT blended toward the keyframe arm -- it is left at the natural grasp
            # seed so it is free to hold the grip (a forced arm config drops the box).
        if verbose:
            print(f"[OCP] keyframe apex at transport knot {kf['k_star']}/{N} "
                  f"(t={times[kf['k_star']]:.2f}s): base={np.round(kf['base'], 2).tolist()} "
                  f"pitch={np.degrees(kf['pitch']):.0f}deg "
                  f"arm(deg)={np.round(np.degrees(kf['d']), 0).tolist()}  w_wp={kf['w_wp']:.0f}")

    # FULLY-consistent kinematic seed: derive VR and AR from the WHOLE config seed (base + theta +
    # arm), not just the base. The integration constraints are QR[k+1]=QR[k]+VR[k+1]*dt and
    # VR[k+1]=VR[k]+AR[k]*dt, so VR[k]=(QR[k]-QR[k-1])/dt (VR[0]=0) and AR[k]=(VR[k+1]-VR[k])/dt.
    # Previously only the 3 base-linear velocities were seeded and theta/arm velocities were left 0;
    # with the keyframe apex bump the seeded theta/arm POSITIONS change but their velocities were 0,
    # so the integration rows started violated by ~10 (inf_pr) and IPOPT fell into restoration and
    # stalled. Seeding the matching velocities makes the start dynamically feasible.
    q_seed = np.stack([np.concatenate([base_ref[k], theta_seed[k], arm_seed[k]])
                       for k in range(N + 1)])
    v_seed = np.zeros((N + 1, 14))
    v_seed[1:] = (q_seed[1:] - q_seed[:-1]) / dt          # VR[k] = (QR[k]-QR[k-1])/dt, VR[0]=0
    v_seed = np.clip(v_seed, -2.8, 2.8)                   # stay inside the |VR|<=3 box (else inf_pr)
    a_seed = np.zeros((N, 14))
    a_seed[:] = np.clip((v_seed[1:] - v_seed[:-1]) / dt, -20.0, 20.0)   # inside |AR|<=25
    for k in range(N + 1):
        opti.set_initial(QR[k], q_seed[k])
        opti.set_initial(PB[k], box_guess[k])
        opti.set_initial(VR[k], v_seed[k])
    for k in range(N):
        opti.set_initial(AR[k], a_seed[k])

    # HOMOTOPY warm start: override the constructed seed with a full prior solution (e.g. the
    # baseline no-keyframe solve). That prior is already FEASIBLE for this (relaxed) problem, so
    # IPOPT starts from feasibility and only has to slide toward the soft keyframe waypoint, instead
    # of clawing out of the bad cold seed (which stalled in restoration at inf_pr 11.8 for ~900 iters).
    if warm is not None:
        for k in range(N + 1):
            opti.set_initial(QR[k], warm["QR"][k])
            opti.set_initial(VR[k], warm["VR"][k])
            opti.set_initial(PB[k], warm["PB"][k])
        for k in range(N):
            opti.set_initial(AR[k], warm["AR"][k])

    # IPOPT iteration log -> a FILE too (conda run buffers stdout, so the live progress of a long
    # solve is otherwise invisible; tail results/ipopt_log.txt to watch it converge / diverge).
    _ipopt_log = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir,
                                              "results", "ipopt_log.txt"))
    os.makedirs(os.path.dirname(_ipopt_log), exist_ok=True)
    opti.solver("ipopt", {"print_time": False},
                {"max_iter": 3000, "tol": 1e-4, "acceptable_tol": 1e-3,
                 "print_level": 5, "mu_strategy": "adaptive",
                 "output_file": _ipopt_log, "file_print_level": 5, "print_frequency_iter": 10})
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
    # full raw decision-variable solution, for reuse as a HOMOTOPY warm start (solve_ocp(warm=...)).
    sol_full = {"QR": np.array([val(QR[k]) for k in range(N + 1)]),
                "VR": np.array([val(VR[k]) for k in range(N + 1)]),
                "AR": np.array([val(AR[k]) for k in range(N)]),
                "PB": box}
    lam = np.array([[float(val(contact(QR[k], PB[k])[2])),
                     float(val(contact(QR[k], PB[k])[3]))] for k in range(N)])
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
        "status": status, "times": times, "base": base, "arm": arm, "box": box,
        "lam": lam, "fn_set": fn_set, "box_ref_z": box_ref_z, "box_ref": box_ref,
        "phase_of": phase_of, "pick": pick, "place": place, "base_ref": base_ref,
        "home": home, "grite_ref": grite_ref, "place_delta": place - pick,
        "lift": float(place[2] - pick[2]),
        "max_tilt_deg": float(np.degrees(np.max(np.linalg.norm(theta, axis=1)))),
        "cost_hist": cost_hist,
        "sol": sol_full,
    }


def plot_cost_breakdown(res, path):
    """Graph each named cost term vs IPOPT iteration (log y), so the term that drives -- or blows
    up -- the objective is obvious. Prints the final-iterate ranking too."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = res.get("cost_hist", {})
    total = h.get("total", [])
    if not total:
        print("[cost] no cost history recorded (solver took 0 iterations?)")
        return
    it = np.arange(len(total))
    terms = [k for k in h if k != "total" and len(h[k]) == len(total)]
    terms.sort(key=lambda k: -(h[k][-1] if h[k] else 0.0))   # largest final term first

    fig, ax = plt.subplots(2, 1, figsize=(9, 9))
    ax[0].plot(it, total, "k-", lw=2.5, label="TOTAL")
    for nm in terms:
        ax[0].plot(it, h[nm], lw=1.6, label=nm)
    ax[0].set_yscale("symlog")
    ax[0].set_xlabel("IPOPT iteration")
    ax[0].set_ylabel("cost contribution")
    ax[0].set_title(f"OCP cost breakdown ({res.get('status','?')})  -- final total = {total[-1]:.3e}")
    ax[0].legend(loc="upper right", fontsize=8, ncol=2)
    ax[0].grid(alpha=0.3)

    finals = [(nm, (h[nm][-1] if h[nm] else 0.0)) for nm in terms]
    names = [f"{nm}\n{v:.2e}" for nm, v in finals]
    ax[1].bar(range(len(finals)), [max(v, 1e-12) for _, v in finals], color="C1")
    ax[1].set_yscale("log")
    ax[1].set_xticks(range(len(finals)))
    ax[1].set_xticklabels(names, fontsize=8)
    ax[1].set_ylabel("final-iterate cost")
    ax[1].set_title("Final cost per term (dominant term = the one to relax/constrain)")
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)

    print(f"[cost] final term ranking ({res.get('status','?')}):")
    for nm, v in finals:
        print(f"[cost]   {nm:10s} = {v:.4e}")
    print(f"[cost]   {'TOTAL':10s} = {total[-1]:.4e}")
    print(f"[cost] figure -> {path}")


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
    plot_cost_breakdown(res, os.path.join(rdir, "ocp_cost_breakdown.png"))
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

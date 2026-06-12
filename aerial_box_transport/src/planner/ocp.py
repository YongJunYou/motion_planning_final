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


def solve_ocp(verbose=False):
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
            opti.subject_to(PB[k][0] == 0.5 * (pl[0] + pr[0]))    # box x at the EE midpoint
            opti.subject_to(lam_l >= fn_set[k])                   # slip-aware: enough squeeze to hold
            opti.subject_to(lam_r >= fn_set[k])
            for c in keep_out(PB[k]):                             # the box avoids desk/rack
                opti.subject_to(c >= 0)
            for c in keep_out(QR[k][0:3]):                        # the drone body avoids them too
                opti.subject_to(c >= 0)
        elif ph == "release":
            opti.subject_to(PB[k] == place)                       # placed on the shelf

        # GRASP DEFINITION (whole-body): the task gives WHERE each pad must be -- the centre
        # of the box's left / right face. Pin each pad's y,z to the box face centre (PB.y,
        # PB.z) in every gripping phase; the squeeze (lambda) presses them into the +-x faces.
        # So the OCP DISCOVERS the base + arm that reach these two pad targets, instead of
        # tracking a precomputed base_grasp / arm_grasp pose. (Box y,z then follow the pads.)
        if ph in ("grasp", "transport", "release"):
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
        # box rises / descends VERTICALLY in the pick / place cylinders (discovered up-over-down)
        opti.subject_to(cylinder(PB[k], pick[0], pick[1], z_top_pick, +1.0) >= 0)
        opti.subject_to(cylinder(PB[k], place[0], place[1], z_top_place, -1.0) >= 0)

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
        cost += w_align * ((PB[k][0] - QR[k][0]) ** 2 + (PB[k][1] - QR[k][1]) ** 2)

        cost += w_att * ca.sumsqr(QR[k][3:6])                     # attitude -> level
        cost += w_reg_a * ca.sumsqr(AR[k]) + w_reg_v * ca.sumsqr(VR[k])

    opti.subject_to(PB[N] == place)
    cost += w_att * ca.sumsqr(QR[N][3:6])
    opti.minimize(cost)

    # seed the full kinematic guess (pos + vel + acc consistent with base_ref) so
    # IPOPT starts near the solution; with metre-scale flights a zero-velocity guess
    # is too far and the solver fails restoration.
    base_v = np.zeros((N + 1, 3))
    base_v[:-1] = (base_ref[1:] - base_ref[:-1]) / dt
    for k in range(N + 1):
        arm_guess = arm_ref_ap[k] if phase_of[min(k, N - 1)] == "approach" else arm_grasp
        opti.set_initial(QR[k], np.concatenate([base_ref[k], [0, 0, 0], arm_guess]))
        opti.set_initial(PB[k], box_guess[k])
        opti.set_initial(VR[k], np.concatenate([base_v[k], np.zeros(11)]))
    for k in range(N):
        opti.set_initial(AR[k], np.concatenate([(base_v[k + 1] - base_v[k]) / dt, np.zeros(11)]))

    opti.solver("ipopt", {"print_time": False},
                {"max_iter": 5000, "tol": 1e-4, "acceptable_tol": 1e-3,
                 "print_level": 5 if verbose else 0, "mu_strategy": "adaptive"})
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

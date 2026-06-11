"""Closed-loop tracking validation: OCP reference -> gRITE wrench applied directly
to the base (allocation skipped) + arm joint commands -> Isaac Sim.

Validates that the planned base trajectory and arm motion are trackable by the
gRITE controller, with the base wrench applied directly to the root body (D5
abstraction). No box yet (M6 scene is gated on the teammate); the disturbance
here is the arm holding the extended grasp configuration, which shifts the
whole-body CoM and is rejected by gRITE's gravity-moment compensation.

Runs in GUI mode (headless hangs on this machine); writes metrics before close().
Run: conda run -n am_isaac python src/sim/track_reference.py --max_time 5.0
Outputs: results/track_log.npz (+ printed RMS tracking error)
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

_THIS = os.path.dirname(os.path.abspath(__file__))
USD = "/home/jaewoo/Research/motion_planning_final/dual_arm_final.usd"

parser = argparse.ArgumentParser(description="gRITE closed-loop reference tracking.")
parser.add_argument("--max_time", type=float, default=5.0)
parser.add_argument("--loop", action="store_true",
                    help="replay the reference continuously (ping-pong) for GUI viewing")
parser.add_argument("--ref", default=os.path.abspath(os.path.join(_THIS, os.pardir, os.pardir,
                                                                  "results", "ocp_reference.npz")))
parser.add_argument("--out", default=os.path.abspath(os.path.join(_THIS, os.pardir, os.pardir,
                                                                  "results", "track_log.npz")))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.assets.articulation import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import matrix_from_quat  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(_THIS, os.pardir)))
from sim.grite_controller import GRITEController  # noqa: E402

SPAWN = (0.0, 0.0, 1.5)
J_BASE_DIAG = [0.0819, 0.1563, 0.2341]
ARM_HOME = {f"dof_{s}{i}": 0.0 for s in ("l", "r") for i in range(1, 5)}
TILT_HOME = {f"dof_{s}{p}{i}": 0.0 for s in ("l", "r") for p in ("b", "f") for i in (1, 2)}
ARM_ORDER = [f"dof_{s}{i}" for s in ("l", "r") for i in range(1, 5)]

# Teammate's scene assets (same world poses as their main.py). The desk and rack
# are loaded VISUAL-ONLY (no colliders): the planar arm forces the drone to sit at
# the box/shelf height, so a physics collider would block it from reaching the box.
# The box is KINEMATIC and driven along the OCP box path each step.
_REPO = "/home/jaewoo/Research/motion_planning_final"
DESK_USD = f"{_REPO}/surroundings/desk_01/desk_01_inst_base.usd"
RACK_USD = f"{_REPO}/surroundings/rack_l01/rack_l01_inst_base.usd"
BOX_USD = f"{_REPO}/box/cubebox_a01/cubebox_a01.usd"
DESK_POS = (2.0, 0.0, 0.0)
RACK_POS = (-2.0, 0.0, 0.0)
BOX_BASE_TO_CENTER = 0.079    # box prim origin at its base; center this far up (taller box 0.158/2)
# pad inner-face contact point in the link4 body frame (matches the OCP EE), used to
# read the ACTUAL gripper midpoint from the sim for debugging / closed-loop box attach.
EE_OFFSET = {"l": np.array([-0.3969, -0.067, 0.0]), "r": np.array([0.397, -0.067, 0.0])}
# Phase boundaries [s], must match task.yaml durations (approach, grasp, transport, release).
APPROACH_END = 3.0
TRANSPORT_START = 4.6   # approach + grasp; the box is lifted off the desk from here, so its
#                         mass is added to the gRITE controller (else it sags/lags under-actuated)
RELEASE_START = 9.6     # approach + grasp + transport

ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            articulation_enabled=True, fix_root_link=False, enabled_self_collisions=False,
            solver_position_iteration_count=8, solver_velocity_iteration_count=0)),
    init_state=ArticulationCfg.InitialStateCfg(pos=SPAWN, joint_pos={**TILT_HOME, **ARM_HOME}),
    actuators={
        "tilt": ImplicitActuatorCfg(joint_names_expr=["dof_[lr][bf][12]"],
                                    stiffness=100.0, damping=10.0, effort_limit_sim=20.0),
        "arms": ImplicitActuatorCfg(joint_names_expr=["dof_[lr][1-4]"],
                                    stiffness=6000.0, damping=400.0, effort_limit_sim=300.0),
    },
)


@configclass
class TrackSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500.0))
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # desk and rack are static colliders (AssetBaseCfg = not simulated as rigid bodies,
    # so they stay put) so the box rests on the desk and the shelf.
    desk = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Desk",
                        spawn=sim_utils.UsdFileCfg(usd_path=DESK_USD,
                            variants={"PhysicsVariant": "RigidBody"},
                            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True)),
                        init_state=AssetBaseCfg.InitialStateCfg(pos=DESK_POS))
    rack = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Rack",
                        spawn=sim_utils.UsdFileCfg(usd_path=RACK_USD,
                            variants={"PhysicsVariant": "RigidBody"},
                            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True)),
                        init_state=AssetBaseCfg.InitialStateCfg(pos=RACK_POS))
    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim_utils.UsdFileCfg(
            usd_path=BOX_USD,
            scale=(1.0, 1.0, 1.5),                              # taller box: centre ~8 cm above desk
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),   # override so the bigger box stays light
            variants={"PhysicsVariant": "RigidBody"},
            # DYNAMIC box: rests on the desk, gripped by the pads via friction, carried,
            # placed on the shelf. Colliders ON so the grippers/desk/rack can touch it.
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False, disable_gravity=False,
                solver_position_iteration_count=16, solver_velocity_iteration_count=1),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, 0.785)))


class Reference:
    def __init__(self, path):
        d = np.load(path)
        self.t = d["times"]
        self.base = d["base"]               # (M,3) base position (OCP frame, starts at 0)
        self.arm = d["arm"]                 # (M,8) arm joints [l1..l4, r1..r4]
        self.gr = d["grite_ref"]            # (M-1,30)
        self.box = d["box"]                 # (M,3) box CENTER in the OCP/home frame
        self.tg = self.t[:len(self.gr)]
        self.spawn = np.asarray(SPAWN)

    def at(self, t):
        t = float(np.clip(t, self.t[0], self.t[-1]))
        p_d = self.spawn + np.array([np.interp(t, self.t, self.base[:, i]) for i in range(3)])
        v_d = np.array([np.interp(t, self.tg, self.gr[:, 3 + i]) for i in range(3)])
        a_d = np.array([np.interp(t, self.tg, self.gr[:, 6 + i]) for i in range(3)])
        q_arm = np.array([np.interp(t, self.t, self.arm[:, i]) for i in range(8)])
        return p_d, v_d, a_d, q_arm

    def box_prim_at(self, t):
        """World pose [pos3, quat_wxyz] of the box PRIM (origin at its base)."""
        t = float(np.clip(t, self.t[0], self.t[-1]))
        center = self.spawn + np.array([np.interp(t, self.t, self.box[:, i]) for i in range(3)])
        pos = center - np.array([0.0, 0.0, BOX_BASE_TO_CENTER])
        return np.concatenate([pos, [1.0, 0.0, 0.0, 0.0]])


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / 200.0))
    sim.set_camera_view(eye=[2.5, 2.5, 2.0], target=[0.0, 0.0, 1.6])
    scene = InteractiveScene(TrackSceneCfg(num_envs=1, env_spacing=4.0))
    sim.reset()
    robot: Articulation = scene["robot"]
    box: RigidObject = scene["box"]
    device = robot.device
    sim_dt = sim.get_physics_dt()

    ref = Reference(args_cli.ref)
    arm_ids = [int(robot.find_joints(n)[0][0]) for n in ARM_ORDER]
    l4 = int(robot.find_bodies("l_link4_01")[0][0])
    r4 = int(robot.find_bodies("r_link4_01")[0][0])

    def actual_ee_mid():
        bl = robot.data.body_pos_w[0, l4].cpu().numpy()
        br = robot.data.body_pos_w[0, r4].cpu().numpy()
        Rl = matrix_from_quat(robot.data.body_quat_w[0, l4].unsqueeze(0))[0].cpu().numpy()
        Rr = matrix_from_quat(robot.data.body_quat_w[0, r4].unsqueeze(0))[0].cpu().numpy()
        return 0.5 * ((bl + Rl @ EE_OFFSET["l"]) + (br + Rr @ EE_OFFSET["r"]))

    m_total = robot.root_physx_view.get_masses()[0].sum().item()
    box_mass = box.root_physx_view.get_masses()[0].sum().item()
    J_base = np.diag(J_BASE_DIAG)
    ctrl = GRITEController(m_total, J_base, g=9.81, dt=sim_dt)
    print(f"[INFO] box mass = {box_mass:.3f} kg (added to gRITE while carrying)")

    # start with arms at the reference initial config (so it begins matched)
    q0 = robot.data.default_joint_pos.clone()
    q0[0, arm_ids] = torch.tensor(ref.arm[0], device=device, dtype=q0.dtype)
    root = robot.data.default_root_state.clone()
    root[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_root_velocity_to_sim(root[:, 7:])
    robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
    scene.reset()
    print(f"[INFO] mass={m_total:.3f} kg, arm ids={arm_ids}, ref horizon={ref.t[-1]:.1f}s")

    R_d = np.eye(3)
    omega_d = np.zeros(3)
    log = {"t": [], "p": [], "p_d": [], "tilt_deg": [], "arm_err": [],
           "ee_mid": [], "box_set": []}
    t, count = 0.0, 0
    T = args_cli.max_time

    def reset_scene():                          # restart the pick-and-place from the top
        robot.write_root_pose_to_sim(root[:, :7])
        robot.write_root_velocity_to_sim(root[:, 7:])
        robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
        box_reset = torch.zeros(1, 7, device=device)
        box_reset[0, :3] = torch.tensor([2.0, 0.0, 0.785], device=device) + scene.env_origins[0]
        box_reset[0, 3] = 1.0
        box.write_root_pose_to_sim(box_reset)
        box.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))
        ctrl.reset()

    try:
        while simulation_app.is_running() and (args_cli.loop or t <= T):
            if args_cli.loop and t > T:         # forward replay (dynamic box -> no ping-pong)
                reset_scene()
                t = 0.0
            rt = t
            p_d, v_d, a_d, q_arm_d = ref.at(rt)

            p = robot.data.root_pos_w[0].cpu().numpy()
            quat = robot.data.root_quat_w[0]
            R = matrix_from_quat(quat.unsqueeze(0))[0].cpu().numpy()
            v = robot.data.root_lin_vel_w[0].cpu().numpy()
            omega_b = robot.data.root_ang_vel_b[0].cpu().numpy()

            masses = robot.root_physx_view.get_masses()[0].to(device)
            body_com = robot.data.body_com_pos_w[0]
            whole_com_w = (body_com * masses.unsqueeze(-1)).sum(0) / masses.sum()
            com_off_b = (R.T @ (whole_com_w.cpu().numpy() - p))

            # while the box is lifted (transport), add its mass so gRITE compensates the
            # extra weight/inertia and the drone does not sag/lag carrying it.
            ctrl.m = m_total + (box_mass if TRANSPORT_START <= rt < RELEASE_START else 0.0)
            f_body, tau_body = ctrl.compute(p, R, v, omega_b, com_off_b,
                                            p_d, v_d, a_d, R_d, omega_d)

            forces = torch.zeros(robot.num_instances, 1, 3, device=device)
            torques = torch.zeros_like(forces)
            forces[:, 0, :] = torch.tensor(f_body, device=device, dtype=forces.dtype)
            torques[:, 0, :] = torch.tensor(tau_body, device=device, dtype=torques.dtype)
            robot.set_external_force_and_torque(forces, torques, body_ids=[0])

            # follow the planned arm; at release, OPEN the grippers (command the pregrasp
            # config) so the box is set down on the shelf instead of carried back.
            arm_cmd = ref.arm[0] if rt >= RELEASE_START else q_arm_d
            q_target = robot.data.default_joint_pos.clone()
            q_target[0, arm_ids] = torch.tensor(arm_cmd, device=device, dtype=q_target.dtype)
            robot.set_joint_position_target(q_target)

            # box is DYNAMIC now: no kinematic posing. The pads grip it via friction.

            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
            t += sim_dt
            count += 1

            tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
            arm_now = robot.data.joint_pos[0, arm_ids].cpu().numpy()
            log["t"].append(t)
            log["p"].append(p.tolist())
            log["p_d"].append(p_d.tolist())
            log["tilt_deg"].append(tilt)
            log["arm_err"].append(float(np.linalg.norm(arm_now - q_arm_d)))
            log["ee_mid"].append(actual_ee_mid().tolist())              # actual gripper midpoint
            log["box_set"].append((box.data.root_pos_w[0].cpu().numpy()
                                   + np.array([0.0, 0.0, BOX_BASE_TO_CENTER])).tolist())  # actual box
            if count % 200 == 0:
                pe = p - p_d
                print(f"[t={t:5.2f}] pos_err=({pe[0]:+.3f},{pe[1]:+.3f},{pe[2]:+.3f}) m "
                      f"tilt={tilt:5.2f} deg arm_err={log['arm_err'][-1]:.3f}")
    finally:
        p = np.array(log["p"])
        p_d = np.array(log["p_d"])
        np.savez(args_cli.out, t=np.array(log["t"]), p=p, p_d=p_d,
                 tilt_deg=np.array(log["tilt_deg"]), arm_err=np.array(log["arm_err"]),
                 ee_mid=np.array(log["ee_mid"]), box_set=np.array(log["box_set"]))
        if len(p):
            settle = np.array(log["t"]) > 1.0   # ignore the initial settle
            rmse = np.sqrt(np.mean(np.sum((p[settle] - p_d[settle]) ** 2, axis=1)))
            print(f"[RESULT] position RMSE (after settle) = {rmse * 1e3:.1f} mm")
            print(f"[RESULT] max tilt = {np.max(log['tilt_deg']):.2f} deg, "
                  f"final arm err = {log['arm_err'][-1]:.3f} rad")
            print(f"[RESULT] final base z = {p[-1, 2]:.3f} m (ref {p_d[-1, 2]:.3f} m)")
            print(f"[RESULT] wrote {args_cli.out}")


if __name__ == "__main__":
    main()
    simulation_app.close()

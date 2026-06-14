"""Side-by-side CLOSED-LOOP PHYSICS comparison of two OCP references (am_isaac).

Two cloned envs, each a FULL dynamic scene (solid window/desk/rack colliders + a dynamic, friction-
gripped box) with its own gRITE controller: env 0 tracks --refA, env 1 tracks --refB, looping. Unlike
play_compare.py (kinematic, collisions OFF), here collisions and grip dynamics are REAL -- the body
can actually hit the wall colliders and the box is held only by pad friction. Use it to physically
compare e.g. the sampler-seeded vs keyframe-guided window passage.

Run: conda run -n am_isaac python src/sim/track_compare.py \
        --refA results/window_reference_sampler_g2.npz --refB results/window_reference_keyframe_g2.npz --loop
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = "/home/jaewoo/Research/motion_planning_final"
USD = f"{_REPO}/dual_arm_final.usd"

parser = argparse.ArgumentParser(description="Side-by-side gRITE physics comparison of two references.")
parser.add_argument("--refA", required=True, help="left env reference npz (OCP output)")
parser.add_argument("--refB", required=True, help="right env reference npz")
parser.add_argument("--max_time", type=float, default=15.0)
parser.add_argument("--loop", action="store_true", help="replay continuously (reset each cycle)")
parser.add_argument("--spacing", type=float, default=7.0, help="env spacing (m)")
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
DESK_USD = f"{_REPO}/surroundings/desk_01/desk_01_inst_base.usd"
RACK_USD = f"{_REPO}/surroundings/rack_l01/rack_l01_inst_base.usd"
BOX_USD = f"{_REPO}/box/cubebox_a01/cubebox_a01.usd"
WINDOW_USD = f"{_REPO}/surroundings/awing_window.usd"
DESK_POS, RACK_POS, WINDOW_POS = (2.0, 0.0, 0.0), (-4.0, 0.0, 0.0), (-1.0, 0.0, 0.0)
BOX_BASE_TO_CENTER = 0.079

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
class CmpSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500.0))
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
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
    window = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Window",
                          spawn=sim_utils.UsdFileCfg(usd_path=WINDOW_USD,
                              collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True)),
                          init_state=AssetBaseCfg.InitialStateCfg(pos=WINDOW_POS))
    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim_utils.UsdFileCfg(
            usd_path=BOX_USD, scale=(1.0, 1.0, 1.5),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            variants={"PhysicsVariant": "RigidBody"},
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False, disable_gravity=False,
                solver_position_iteration_count=16, solver_velocity_iteration_count=1),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, 0.785)))


class Reference:
    def __init__(self, path):
        d = np.load(path)
        self.t = d["times"]
        self.base = d["base"]
        self.arm = d["arm"]
        self.gr = d["grite_ref"]
        self.box = d["box"]
        self.tg = self.t[:len(self.gr)]
        self.spawn = np.asarray(SPAWN)
        pb = d["phase_bounds"] if "phase_bounds" in d.files else None
        self.transport_start = float(pb[1]) if pb is not None else 4.6
        self.release_start = float(pb[2]) if pb is not None else 13.6

    def at(self, t):
        t = float(np.clip(t, self.t[0], self.t[-1]))
        p_d = self.spawn + np.array([np.interp(t, self.t, self.base[:, i]) for i in range(3)])
        v_d = np.array([np.interp(t, self.tg, self.gr[:, 3 + i]) for i in range(3)])
        a_d = np.array([np.interp(t, self.tg, self.gr[:, 6 + i]) for i in range(3)])
        q_arm = np.array([np.interp(t, self.t, self.arm[:, i]) for i in range(8)])
        Rcol = np.array([np.interp(t, self.tg, self.gr[:, 12 + i]) for i in range(9)])
        U, _, Vt = np.linalg.svd(Rcol.reshape(3, 3, order="F"))
        R_d = U @ Vt
        if np.linalg.det(R_d) < 0:
            U[:, -1] *= -1.0
            R_d = U @ Vt
        omega_d = np.array([np.interp(t, self.tg, self.gr[:, 21 + i]) for i in range(3)])
        omega_d_dot = np.array([np.interp(t, self.tg, self.gr[:, 24 + i]) for i in range(3)])
        return p_d, v_d, a_d, q_arm, R_d, omega_d, omega_d_dot


def spawn_window_proxy(origin, env_i):
    """Window keep-out volumes as SOLID colliders at this env's origin (the window USD has none)."""
    if not os.path.exists("/tmp/window_ocp_play.npz"):
        print("[WARN] no /tmp/window_ocp_play.npz -> no window colliders")
        return
    d = np.load("/tmp/window_ocp_play.npz")
    full, cen, quat = d["proxy_full"], d["proxy_cen"], d["proxy_quat"]
    o = origin.cpu().numpy()
    for i in range(len(full)):
        cfg = sim_utils.CuboidCfg(
            size=tuple(float(v) for v in full[i]),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.12, 0.12), opacity=0.30))
        cfg.func(f"/World/WinProxy/e{env_i}_box{i}", cfg,
                 translation=tuple(float(v) for v in (cen[i] + o)),
                 orientation=tuple(float(v) for v in quat[i]))


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / 200.0))
    scene = InteractiveScene(CmpSceneCfg(num_envs=2, env_spacing=args_cli.spacing))
    origins = scene.env_origins
    spawn_window_proxy(origins[0], 0)
    spawn_window_proxy(origins[1], 1)
    sim.reset()
    robot: Articulation = scene["robot"]
    box: RigidObject = scene["box"]
    device = robot.device
    sim_dt = sim.get_physics_dt()

    refs = [Reference(args_cli.refA), Reference(args_cli.refB)]
    arm_ids = [int(robot.find_joints(n)[0][0]) for n in ARM_ORDER]
    m_total = robot.root_physx_view.get_masses()[0].sum().item()
    box_mass = box.root_physx_view.get_masses()[0].sum().item()
    J_base = np.diag(J_BASE_DIAG)
    ctrls = [GRITEController(m_total, J_base, g=9.81, dt=sim_dt),
             GRITEController(m_total, J_base, g=9.81, dt=sim_dt)]

    cen = origins.mean(0).cpu().numpy()
    span = float((origins[1] - origins[0]).norm().cpu())
    sim.set_camera_view(eye=[cen[0] + 3.0, cen[1] - (span + 5.0), cen[2] + 3.5],
                        target=[cen[0] - 1.0, cen[1], 1.6])
    print(f"[INFO] env A (refA) = {os.path.basename(args_cli.refA)}, "
          f"env B (refB) = {os.path.basename(args_cli.refB)}; span {span:.1f} m, box {box_mass:.2f} kg")

    # initial state: arms matched to each ref's start, base at default + env origin
    q0 = robot.data.default_joint_pos.clone()
    for e in (0, 1):
        q0[e, arm_ids] = torch.tensor(refs[e].arm[0], device=device, dtype=q0.dtype)
    root = robot.data.default_root_state.clone()
    root[:, :3] += origins

    def reset_scene():
        robot.write_root_pose_to_sim(root[:, :7])
        robot.write_root_velocity_to_sim(root[:, 7:])
        robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
        box_reset = torch.zeros(2, 7, device=device)
        for e in (0, 1):
            box_reset[e, :3] = torch.tensor([2.0, 0.0, 0.785], device=device) + origins[e]
            box_reset[e, 3] = 1.0
        box.write_root_pose_to_sim(box_reset)
        box.write_root_velocity_to_sim(torch.zeros(2, 6, device=device))
        for c in ctrls:
            c.reset()

    reset_scene()
    scene.reset()
    masses_t = robot.root_physx_view.get_masses()[0].to(device)   # per-body masses (same both envs)
    origins_np = origins.cpu().numpy()   # the reference is WORLD-frame; gRITE must track to env origin
    T = args_cli.max_time
    t, count = 0.0, 0
    print(f"[INFO] tracking horizon {refs[0].t[-1]:.1f}s (loop={args_cli.loop})")

    while simulation_app.is_running() and (args_cli.loop or t <= T):
        if args_cli.loop and t > T:
            reset_scene()
            t = 0.0
        forces = torch.zeros(robot.num_instances, 1, 3, device=device)
        torques = torch.zeros_like(forces)
        q_target = robot.data.default_joint_pos.clone()
        for e in (0, 1):
            ref, ctrl = refs[e], ctrls[e]
            p_d, v_d, a_d, q_arm_d, R_d, omega_d, omega_d_dot = ref.at(t)
            p_d = p_d + origins_np[e]      # CRITICAL: offset the reference to THIS env (else ~7 m error)
            p = robot.data.root_pos_w[e].cpu().numpy()
            R = matrix_from_quat(robot.data.root_quat_w[e].unsqueeze(0))[0].cpu().numpy()
            v = robot.data.root_lin_vel_w[e].cpu().numpy()
            omega_b = robot.data.root_ang_vel_b[e].cpu().numpy()
            body_com = robot.data.body_com_pos_w[e]
            whole_com_w = (body_com * masses_t.unsqueeze(-1)).sum(0) / masses_t.sum()
            com_off_b = R.T @ (whole_com_w.cpu().numpy() - p)
            carrying = ref.transport_start <= t < ref.release_start
            ctrl.m = m_total + (box_mass if carrying else 0.0)
            f_body, tau_body = ctrl.compute(p, R, v, omega_b, com_off_b,
                                            p_d, v_d, a_d, R_d, omega_d, omega_d_dot)
            forces[e, 0, :] = torch.tensor(f_body, device=device, dtype=forces.dtype)
            torques[e, 0, :] = torch.tensor(tau_body, device=device, dtype=torques.dtype)
            if t >= ref.release_start:
                arm_cmd = ref.arm[-1].copy()
                arm_cmd[1:4] = ref.arm[0][1:4]
                arm_cmd[5:8] = ref.arm[0][5:8]
            else:
                arm_cmd = q_arm_d
            q_target[e, arm_ids] = torch.tensor(arm_cmd, device=device, dtype=q_target.dtype)
        robot.set_external_force_and_torque(forces, torques, body_ids=[0])
        robot.set_joint_position_target(q_target)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        t += sim_dt
        count += 1
        if count % 400 == 0:
            eA = robot.data.root_pos_w[0].cpu().numpy() - (refs[0].at(t)[0] + origins_np[0])
            eB = robot.data.root_pos_w[1].cpu().numpy() - (refs[1].at(t)[0] + origins_np[1])
            print(f"[t={t:5.2f}] A pos_err {np.linalg.norm(eA)*100:4.1f}cm  B pos_err {np.linalg.norm(eB)*100:4.1f}cm")


if __name__ == "__main__":
    main()
    simulation_app.close()

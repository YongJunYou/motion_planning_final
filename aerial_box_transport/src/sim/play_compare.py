"""Side-by-side kinematic playback of TWO planned paths for visual A/B comparison (am_isaac).

Spawns the scene TWICE (two cloned envs) and poses env 0's robot+box from --playA and env 1's from
--playB every frame, looping. Use it to eyeball the difference between two OCP variants, e.g. the
keyframe window OCP WITH vs WITHOUT the w_kf waypoint. Kinematic (poses written each frame), so what
you see is exactly the planned path -- no controller tracking error.

Run: conda run -n am_isaac python src/sim/play_compare.py \
        --playA results_play/with_wkf.npz --playB results_play/no_wkf.npz --proxy --loop
"""
import argparse
import os

from isaaclab.app import AppLauncher

_REPO = "/home/jaewoo/Research/motion_planning_final"
USD = f"{_REPO}/dual_arm_final.usd"

parser = argparse.ArgumentParser(description="Side-by-side kinematic A/B playback.")
parser.add_argument("--playA", required=True, help="left env play npz (export_play format)")
parser.add_argument("--playB", required=True, help="right env play npz")
parser.add_argument("--loop", action="store_true")
parser.add_argument("--proxy", action="store_true", help="spawn window keep-out volumes per env")
parser.add_argument("--hold", type=int, default=4, help="render steps per frame (slows playback)")
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

SPAWN = (0.0, 0.0, 1.5)
ARM_HOME = {f"dof_{s}{i}": 0.0 for s in ("l", "r") for i in range(1, 5)}
TILT_HOME = {f"dof_{s}{p}{i}": 0.0 for s in ("l", "r") for p in ("b", "f") for i in (1, 2)}
ARM_ORDER = [f"dof_{s}{i}" for s in ("l", "r") for i in range(1, 5)]
DESK_USD = f"{_REPO}/surroundings/desk_01/desk_01_inst_base.usd"
RACK_USD = f"{_REPO}/surroundings/rack_l01/rack_l01_inst_base.usd"
BOX_USD = f"{_REPO}/box/cubebox_a01/cubebox_a01.usd"
WINDOW_USD = f"{_REPO}/surroundings/awing_window.usd"
DESK_POS, RACK_POS, WINDOW_POS = (2.0, 0.0, 0.0), (-4.0, 0.0, 0.0), (-1.0, 0.0, 0.0)

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
                        spawn=sim_utils.UsdFileCfg(usd_path=DESK_USD),
                        init_state=AssetBaseCfg.InitialStateCfg(pos=DESK_POS))
    rack = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Rack",
                        spawn=sim_utils.UsdFileCfg(usd_path=RACK_USD),
                        init_state=AssetBaseCfg.InitialStateCfg(pos=RACK_POS))
    window = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Window",
                          spawn=sim_utils.UsdFileCfg(usd_path=WINDOW_USD),
                          init_state=AssetBaseCfg.InitialStateCfg(pos=WINDOW_POS))
    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim_utils.UsdFileCfg(
            usd_path=BOX_USD, scale=(1.0, 1.0, 1.5),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            variants={"PhysicsVariant": "RigidBody"},
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, 0.785)))


def spawn_proxy(d, origin, env_i):
    """Window keep-out volumes (4 wall borders + tilted sash) as solid, semi-transparent red colliders,
    placed at this env's origin. Static, collision OFF (visual reference only, since this is kinematic)."""
    full, cen, quat = d["proxy_full"], d["proxy_cen"], d["proxy_quat"]
    o = origin.cpu().numpy()
    for i in range(len(full)):
        cfg = sim_utils.CuboidCfg(
            size=tuple(float(v) for v in full[i]),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.12, 0.12), opacity=0.25))
        cfg.func(f"/World/WinProxy/e{env_i}_box{i}", cfg,
                 translation=tuple(float(v) for v in (cen[i] + o)),
                 orientation=tuple(float(v) for v in quat[i]))


def main():
    dA = {k: np.asarray(v) for k, v in np.load(args_cli.playA).items()}
    dB = {k: np.asarray(v) for k, v in np.load(args_cli.playB).items()}
    n = min(len(dA["base_pos"]), len(dB["base_pos"]))
    print(f"[INFO] A={args_cli.playA} ({len(dA['base_pos'])} frames)  B={args_cli.playB} "
          f"({len(dB['base_pos'])} frames)  -> playing {n}")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / 60.0,
                                                              gravity=(0.0, 0.0, 0.0)))
    scene = InteractiveScene(CmpSceneCfg(num_envs=2, env_spacing=args_cli.spacing))
    sim.reset()
    robot: Articulation = scene["robot"]
    box: RigidObject = scene["box"]
    device = robot.device
    arm_ids = [int(robot.find_joints(nm)[0][0]) for nm in ARM_ORDER]
    origins = scene.env_origins                                   # (2,3)
    cen = origins.mean(0).cpu().numpy()
    span = float((origins[1] - origins[0]).norm().cpu())
    sim.set_camera_view(eye=[cen[0] + 3.0, cen[1] - (span + 5.0), cen[2] + 3.5],
                        target=[cen[0] - 1.0, cen[1], 1.6])
    if args_cli.proxy:
        spawn_proxy(dA, origins[0], 0)
        spawn_proxy(dB, origins[1], 1)
        print("[INFO] spawned window keep-out volumes in both envs")
    print(f"[INFO] env A (left) = {os.path.basename(args_cli.playA)}, "
          f"env B (right) = {os.path.basename(args_cli.playB)}; origins span {span:.1f} m")

    def pose(k):
        root = torch.zeros(2, 7, device=device)
        bp = torch.zeros(2, 7, device=device)
        q = robot.data.default_joint_pos.clone()
        for e, d in ((0, dA), (1, dB)):
            root[e, :3] = torch.tensor(d["base_pos"][k], device=device) + origins[e]
            root[e, 3:7] = torch.tensor(d["base_quat"][k], device=device)
            q[e, arm_ids] = torch.tensor(d["arm"][k], device=device, dtype=q.dtype)
            bp[e, :3] = torch.tensor(d["box_pos"][k], device=device) + origins[e]
            bp[e, 3:7] = torch.tensor(d["box_quat"][k], device=device)
        robot.write_root_pose_to_sim(root)
        robot.write_root_velocity_to_sim(torch.zeros(2, 6, device=device))
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        robot.set_joint_position_target(q)
        box.write_root_pose_to_sim(bp)
        box.write_root_velocity_to_sim(torch.zeros(2, 6, device=device))

    hold = max(1, args_cli.hold)
    k = 0
    print(f"[INFO] playing {n} frames (loop={args_cli.loop})")
    while simulation_app.is_running():
        pose(k)
        scene.write_data_to_sim()
        for _ in range(hold):
            sim.step()
            scene.update(1.0 / 60.0)
        k += 1
        if k >= n:
            if not args_cli.loop:
                break
            k = 0


if __name__ == "__main__":
    main()
    simulation_app.close()

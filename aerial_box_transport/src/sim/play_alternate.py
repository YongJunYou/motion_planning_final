"""Single-env kinematic playback that ALTERNATES two planned paths in a loop (am_isaac).

ONE IsaacSim env: play --playA to completion, hold a beat, play --playB, hold, then repeat forever, so
the two window passages (sampler yaw-squeeze vs keyframe pitch-forward) are seen back-to-back in the
SAME view. Kinematic (poses written each frame) so what you see is exactly the planned path, with no
controller tracking error or grip slip. The console prints which path is currently playing.

Camera defaults to a side profile (look along +y) that shows the desk (x=+2), the window (x=-1) and the
rack (x=-4) at once; override with CAM_EYE / CAM_TARGET = "x,y,z".

Run: conda run -n am_isaac python src/sim/play_alternate.py \
        --playA /tmp/play_sampler_g2.npz --playB /tmp/play_keyframe_g2.npz --proxy
"""
import argparse
import os

from isaaclab.app import AppLauncher

_REPO = "/home/jaewoo/Research/motion_planning_final"
USD = f"{_REPO}/dual_arm_final.usd"

parser = argparse.ArgumentParser(description="Single-env alternating kinematic A/B playback.")
parser.add_argument("--playA", default="/tmp/play_sampler_g2.npz", help="first path (export_play npz)")
parser.add_argument("--playB", default="/tmp/play_keyframe_g2.npz", help="second path (export_play npz)")
parser.add_argument("--labelA", default="sampler_g2 (yaw-squeeze)")
parser.add_argument("--labelB", default="keyframe_g2 (pitch-forward)")
parser.add_argument("--proxy", action="store_true", help="spawn the window keep-out volumes")
parser.add_argument("--hold", type=int, default=3, help="render steps per frame (higher = slower)")
parser.add_argument("--gap", type=int, default=45, help="render steps to pause between the two paths")
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
class SceneCfg(InteractiveSceneCfg):
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


def spawn_proxy(d):
    """Window keep-out volumes (4 wall borders + tilted sash) as solid, semi-transparent red colliders.
    Static, collision OFF (visual reference only, since this is kinematic playback)."""
    full, cen, quat = d["proxy_full"], d["proxy_cen"], d["proxy_quat"]
    for i in range(len(full)):
        cfg = sim_utils.CuboidCfg(
            size=tuple(float(v) for v in full[i]),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.12, 0.12), opacity=0.25))
        cfg.func(f"/World/WinProxy/box{i}", cfg,
                 translation=tuple(float(v) for v in cen[i]),
                 orientation=tuple(float(v) for v in quat[i]))


def main():
    dA = {k: np.asarray(v) for k, v in np.load(args_cli.playA).items()}
    dB = {k: np.asarray(v) for k, v in np.load(args_cli.playB).items()}
    print(f"[INFO] A={os.path.basename(args_cli.playA)} ({len(dA['base_pos'])} frames)  "
          f"B={os.path.basename(args_cli.playB)} ({len(dB['base_pos'])} frames)")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / 60.0,
                                                              gravity=(0.0, 0.0, 0.0)))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=8.0))
    sim.reset()
    robot: Articulation = scene["robot"]
    box: RigidObject = scene["box"]
    device = robot.device
    arm_ids = [int(robot.find_joints(nm)[0][0]) for nm in ARM_ORDER]

    _ce, _ct = os.environ.get("CAM_EYE"), os.environ.get("CAM_TARGET")
    eye = [float(x) for x in _ce.split(",")] if _ce else [-1.0, -10.0, 2.6]
    tgt = [float(x) for x in _ct.split(",")] if _ct else [-1.0, 0.0, 1.7]
    sim.set_camera_view(eye=eye, target=tgt)
    if args_cli.proxy:
        spawn_proxy(dA)
        print("[INFO] spawned window keep-out volumes")

    def pose(d, k):
        root = torch.zeros(1, 7, device=device)
        bp = torch.zeros(1, 7, device=device)
        q = robot.data.default_joint_pos.clone()
        root[0, :3] = torch.tensor(d["base_pos"][k], device=device)
        root[0, 3:7] = torch.tensor(d["base_quat"][k], device=device)
        q[0, arm_ids] = torch.tensor(d["arm"][k], device=device, dtype=q.dtype)
        bp[0, :3] = torch.tensor(d["box_pos"][k], device=device)
        bp[0, 3:7] = torch.tensor(d["box_quat"][k], device=device)
        robot.write_root_pose_to_sim(root)
        robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        robot.set_joint_position_target(q)
        box.write_root_pose_to_sim(bp)
        box.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))

    segs = [(dA, args_cli.labelA), (dB, args_cli.labelB)]
    hold, gap = max(1, args_cli.hold), max(0, args_cli.gap)
    si, k = 0, 0
    print(f"[NOW PLAYING] A: {segs[0][1]}", flush=True)
    while simulation_app.is_running():
        d, _label = segs[si]
        n = len(d["base_pos"])
        pose(d, k)
        scene.write_data_to_sim()
        for _ in range(hold):
            sim.step()
            scene.update(1.0 / 60.0)
        k += 1
        if k >= n:
            for _ in range(gap):                 # hold the final (placed) frame, then switch paths
                sim.step()
                scene.update(1.0 / 60.0)
            si = (si + 1) % 2
            k = 0
            print(f"[NOW PLAYING] {'A' if si == 0 else 'B'}: {segs[si][1]}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()

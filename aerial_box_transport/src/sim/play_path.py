"""Kinematic playback of a planned path in IsaacSim, for VISUAL verification (not control).

Poses the robot + carried box EXACTLY at each precomputed config (from export_play.py) with the window
scene loaded, so you can see whether the planned whole-body motion threads the awning window. No gRITE
controller, no friction grip: the robot root pose, arm joints, and box pose are written every frame, so
what you see IS the plan (no tracking error, no slip). Loops for viewing.

Modes:
  (default)   live GUI loop, window mesh only (collision geometry OFF)
  --proxy     also spawn the planner's collision volumes (4 wall borders + tilted sash) as SOLID,
              semi-transparent colliders, since the teammate's window USD has NO colliders. This is
              "collision ON": you see the actual keep-out volumes and that the body threads them.
  --record P  headless, capture a video to P.mp4 (one pass), then exit.

Run: conda run -n am_isaac python src/sim/play_path.py --play /tmp/window_play.npz --loop
     conda run -n am_isaac python src/sim/play_path.py --proxy --loop
     conda run -n am_isaac python src/sim/play_path.py --record /tmp/window_off.mp4
"""
import argparse
import os

from isaaclab.app import AppLauncher

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = "/home/jaewoo/Research/motion_planning_final"
USD = f"{_REPO}/dual_arm_final.usd"

parser = argparse.ArgumentParser(description="Kinematic playback of a planned path.")
parser.add_argument("--play", default="/tmp/window_play.npz", help="npz from export_play.py")
parser.add_argument("--loop", action="store_true", help="replay continuously")
parser.add_argument("--proxy", action="store_true", help="spawn the window collision volumes (ON)")
parser.add_argument("--record", default="", help="record a video to this mp4 (headless, one pass)")
parser.add_argument("--hold", type=int, default=3, help="render steps per frame (slows playback)")
parser.add_argument("--fps", type=int, default=30, help="video fps when recording")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.record:                       # recording needs offscreen rendering, no GUI window
    args_cli.headless = True
    args_cli.enable_cameras = True
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
DESK_POS, RACK_POS, WINDOW_POS = (2.0, 0.0, 0.0), (-2.0, 0.0, 0.0), (-1.0, 0.0, 0.0)

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
class PlaySceneCfg(InteractiveSceneCfg):
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
    # box is KINEMATIC: we pose it at the gripper each frame (no gravity, no contact dynamics).
    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim_utils.UsdFileCfg(
            usd_path=BOX_USD, scale=(1.0, 1.0, 1.5),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            variants={"PhysicsVariant": "RigidBody"},      # select the variant carrying RigidBodyAPI
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, 0.785)))


def spawn_proxy(d, origin):
    """Spawn the planner's window keep-out volumes as SOLID, semi-transparent red colliders -- this
    is the collision geometry the teammate's window USD lacks. Static (no rigid body), collision ON."""
    full, cen, quat = d["proxy_full"], d["proxy_cen"], d["proxy_quat"]
    o = origin.cpu().numpy()
    for i in range(len(full)):
        cfg = sim_utils.CuboidCfg(
            size=tuple(float(v) for v in full[i]),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.12, 0.12),
                                                        opacity=0.28))
        cfg.func(f"/World/WinProxy/box_{i}", cfg,
                 translation=tuple(float(v) for v in (cen[i] + o)),
                 orientation=tuple(float(v) for v in quat[i]))
    print(f"[INFO] spawned {len(full)} window collision volumes (SOLID, semi-transparent)")


def main():
    d = np.load(args_cli.play)
    base_pos, base_quat = d["base_pos"], d["base_quat"]
    arm, box_pos, box_quat = d["arm"], d["box_pos"], d["box_quat"]
    clear = d["clear"] if "clear" in d.files else np.ones(len(base_pos), bool)
    n = len(base_pos)

    # gravity OFF: this is a KINEMATIC playback (poses written every frame). With gravity on, the
    # free-floating base has no holding force and sags ~1 cm between writes, then teleports back ->
    # ~20 Hz vibration. Zeroing gravity makes the written poses hold steady (no physics transients).
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / 60.0,
                                                              gravity=(0.0, 0.0, 0.0)))
    sim.set_camera_view(eye=[2.6, -4.0, 2.6], target=[-0.8, 0.0, 1.6])
    scene = InteractiveScene(PlaySceneCfg(num_envs=1, env_spacing=4.0))

    camera = None
    if args_cli.record:
        from isaaclab.sensors import Camera, CameraCfg
        camera = Camera(CameraCfg(
            prim_path="/World/RecordCam", height=720, width=1280, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 1e5)),
            update_period=0,
            offset=CameraCfg.OffsetCfg(pos=(2.6, -4.0, 2.6), rot=(1.0, 0.0, 0.0, 0.0),
                                       convention="world")))

    sim.reset()
    robot: Articulation = scene["robot"]
    box: RigidObject = scene["box"]
    device = robot.device
    arm_ids = [int(robot.find_joints(nm)[0][0]) for nm in ARM_ORDER]
    origin = scene.env_origins[0]
    if args_cli.proxy:
        spawn_proxy(d, origin)
    if camera is not None:
        camera.set_world_poses_from_view(
            torch.tensor([[2.6, -4.0, 2.6]], device=device),
            torch.tensor([[-0.8, 0.0, 1.6]], device=device))

    n_clip = int((~clear).sum())
    print(f"[INFO] playing {n} frames from {args_cli.play}  (loop={args_cli.loop}, proxy={args_cli.proxy})")
    print(f"[INFO] collision model: {n - n_clip}/{n} frames clear; {n_clip} minor inter-knot clips"
          + (f" at {list(np.where(~clear)[0])}" if n_clip else ""))

    def pose_frame(k):
        root = torch.zeros(1, 7, device=device)
        root[0, :3] = torch.tensor(base_pos[k], device=device) + origin
        root[0, 3:7] = torch.tensor(base_quat[k], device=device)        # wxyz
        robot.write_root_pose_to_sim(root)
        robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))
        q = robot.data.default_joint_pos.clone()
        q[0, arm_ids] = torch.tensor(arm[k], device=device, dtype=q.dtype)
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        robot.set_joint_position_target(q)
        bp = torch.zeros(1, 7, device=device)
        bp[0, :3] = torch.tensor(box_pos[k], device=device) + origin
        bp[0, 3:7] = torch.tensor(box_quat[k], device=device)
        box.write_root_pose_to_sim(bp)
        box.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))

    frames = []
    hold = 1 if args_cli.record else max(1, args_cli.hold)
    k = 0
    while simulation_app.is_running():
        pose_frame(k)
        scene.write_data_to_sim()
        for _ in range(hold):
            sim.step()
            scene.update(1.0 / 60.0)
        if camera is not None:
            camera.update(1.0 / 60.0)
            rgb = camera.data.output["rgb"][0, :, :, :3].detach().cpu().numpy().astype(np.uint8)
            frames.append(rgb)
        k += 1
        if k >= n:
            if args_cli.record:
                import imageio.v2 as imageio
                w = imageio.get_writer(args_cli.record, fps=args_cli.fps, macro_block_size=8)
                for f in frames:
                    w.append_data(f)
                w.close()
                print(f"[INFO] wrote {len(frames)} frames -> {args_cli.record}")
                break
            if not args_cli.loop:
                break
            k = 0


if __name__ == "__main__":
    main()
    simulation_app.close()

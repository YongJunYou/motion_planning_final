"""Dump the as-simulated inertial data and body rest poses from Isaac Sim.

Per-link center of mass and inertia are NOT authored in the USD (PhysX
auto-computes them from geometry), so we read what the simulator actually uses.
This complements the USD kinematic extraction (extract_usd.py) and is the
inertial data source for the M1 Pinocchio model. It also records body world
rest poses at the zero joint configuration for the FK cross-check.

Runs headless and terminates (no control loop). Reuses the proven scene config
from ../motion_planning_final/main.py.

Run (needs GPU, am_isaac):
  conda run -n am_isaac python src/sim/dump_model.py --headless
Output: config/isaac_model.json
"""
import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dump Isaac/PhysX model data.")
parser.add_argument("--usd", default=os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, "dual_arm_final.usd")))
parser.add_argument("--out", default=os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, "config", "isaac_model.json")))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.assets.articulation import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=args_cli.usd,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            articulation_enabled=True, fix_root_link=True,
            enabled_self_collisions=False,
            solver_position_iteration_count=8, solver_velocity_iteration_count=0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    actuators={
        "all": ImplicitActuatorCfg(joint_names_expr=["dof_.*"],
                                   stiffness=100.0, damping=10.0),
    },
)


@configclass
class DumpSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light",
                         spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def as_list(x):
    return np.asarray(x).astype(float).tolist()


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / 200.0))
    scene = InteractiveScene(DumpSceneCfg(num_envs=1, env_spacing=4.0))
    sim.reset()
    robot: Articulation = scene["robot"]

    view = robot.root_physx_view
    data = {
        "body_names": list(robot.body_names),
        "joint_names": list(robot.joint_names),
    }

    # PhysX inertial data (as the simulator uses it).
    for key, fn in [("masses", "get_masses"), ("inertias", "get_inertias"),
                    ("coms", "get_coms")]:
        try:
            arr = np.asarray(getattr(view, fn)()[0].cpu().numpy())
            data[key] = arr.tolist()
            data[key + "_shape"] = list(arr.shape)
        except Exception as exc:  # noqa: BLE001
            data[key + "_error"] = repr(exc)

    # Body world rest poses at the zero joint configuration (for FK cross-check).
    data["body_pos_w"] = as_list(robot.data.body_pos_w[0].cpu().numpy())
    data["body_quat_w"] = as_list(robot.data.body_quat_w[0].cpu().numpy())  # wxyz
    data["joint_pos"] = as_list(robot.data.joint_pos[0].cpu().numpy())
    data["root_pos_w"] = as_list(robot.data.root_pos_w[0].cpu().numpy())
    data["root_quat_w"] = as_list(robot.data.root_quat_w[0].cpu().numpy())

    with open(args_cli.out, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[DUMP] bodies: {len(data['body_names'])}  joints: {len(data['joint_names'])}")
    print(f"[DUMP] body_names: {data['body_names']}")
    print(f"[DUMP] joint_names: {data['joint_names']}")
    for k in ("masses", "inertias", "coms"):
        if k + "_shape" in data:
            print(f"[DUMP] {k} shape: {data[k + '_shape']}")
        elif k + "_error" in data:
            print(f"[DUMP] {k} ERROR: {data[k + '_error']}")
    print(f"[DUMP] total mass: {float(np.sum(data['masses'])):.4f} kg"
          if "masses" in data else "[DUMP] masses unavailable")
    print(f"[DUMP] wrote {args_cli.out}")


if __name__ == "__main__":
    main()
    simulation_app.close()

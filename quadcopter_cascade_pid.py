# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Closed-loop quadcopter control with a cascade PID controller.

Structure
---------
Outer loop  (position PID, world frame):  p, v  -> desired force vector F_des (world)
Map         (force -> attitude):           F_des -> total thrust (body z) + desired rotation R_des
Inner loop  (attitude PID, SO(3)):         R, R_des, w -> body torque tau

The total thrust (along body z) and the body torque are applied as an external
wrench on the root body, which is exactly how Isaac Lab's own quadcopter task
drives the Crazyflie. The reference trajectory is given in closed form
(analytic position / velocity / acceleration) and fed forward into the loops.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/quadcopter_cascade_pid.py --mode circle
    ./isaaclab.sh -p scripts/demos/quadcopter_cascade_pid.py --mode hover
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Closed-loop cascade-PID quadcopter control demo.")
parser.add_argument("--mode", type=str, default="circle", choices=["hover", "circle"],
                    help="Reference trajectory type.")
parser.add_argument("--radius", type=float, default=0.5, help="Circle radius [m].")
parser.add_argument("--period", type=float, default=6.0, help="Circle period [s].")
parser.add_argument("--ref_height", type=float, default=0.5, help="Reference altitude [m].")
parser.add_argument("--no_ros2", action="store_true", help="Disable ROS2 trajectory topics.")
parser.add_argument("--ros2_decim", type=int, default=4, help="Publish every N sim steps (200Hz / N).")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import matrix_from_quat

##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort:skip


def vee(skew: torch.Tensor) -> torch.Tensor:
    """Inverse of the hat map: extract the 3-vector from a (batched) skew matrix.

    skew: (..., 3, 3)  ->  (..., 3)
    """
    return torch.stack([skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], dim=-1)


class CascadePIDController:
    """Cascade PID (geometric) controller for a quadcopter.

    All quantities are batched over ``num_envs`` so the controller works for a
    single robot or many cloned robots without changes.
    """

    def __init__(self, num_envs: int, mass: float, gravity: float,
                 inertia: torch.Tensor, device: str):
        self.N = num_envs
        self.m = float(mass)          # total vehicle mass [kg]
        self.g = float(gravity)       # gravity magnitude [m/s^2]
        self.J = inertia              # body-frame inertia (3, 3) [kg m^2]
        self.device = device

        # --- Outer (position) loop gains, per world axis [x, y, z] -----------
        # Output is a desired acceleration -> gains are mass independent.
        self.Kp_pos = torch.tensor([6.0, 6.0, 16.0], device=device)
        self.Ki_pos = torch.tensor([1.0, 1.0, 4.0], device=device)
        self.Kd_pos = torch.tensor([4.5, 4.5, 8.0], device=device)

        # --- Inner (attitude) loop gains, per body axis [roll, pitch, yaw] ---
        # Output is a desired angular acceleration -> multiplied by J to get a
        # torque, so the gains stay interpretable and inertia independent.
        self.Kp_att = torch.tensor([250.0, 250.0, 80.0], device=device)
        self.Kd_att = torch.tensor([30.0, 30.0, 16.0], device=device)
        self.Ki_att = torch.tensor([0.0, 0.0, 0.0], device=device)

        # integrator states + anti-windup limits
        self._int_pos = torch.zeros(self.N, 3, device=device)
        self._int_att = torch.zeros(self.N, 3, device=device)
        self._pos_int_limit = 1.0
        self._att_int_limit = 0.5

        # constants
        self._z_w = torch.tensor([0.0, 0.0, 1.0], device=device)

    def reset(self):
        self._int_pos.zero_()
        self._int_att.zero_()

    def compute(self, pos, quat, lin_vel, ang_vel_b,
                pos_des, vel_des, acc_des, yaw_des, dt):
        """Return external (forces, torques) of shape (N, 1, 3) in the body frame.

        Args (all torch tensors on ``self.device``):
            pos        (N, 3)  world position
            quat       (N, 4)  world orientation, (w, x, y, z)
            lin_vel    (N, 3)  world linear velocity
            ang_vel_b  (N, 3)  body-frame angular velocity
            pos_des    (N, 3)  reference position (world)
            vel_des    (N, 3)  reference velocity (world)
            acc_des    (N, 3)  reference acceleration (world, feed-forward)
            yaw_des    (N,)    reference yaw [rad]
        """
        # ---------------- Outer loop: position PID -> world force -----------
        e_pos = pos_des - pos
        e_vel = vel_des - lin_vel
        self._int_pos = torch.clamp(self._int_pos + e_pos * dt,
                                    -self._pos_int_limit, self._pos_int_limit)
        acc_cmd = (self.Kp_pos * e_pos
                   + self.Ki_pos * self._int_pos
                   + self.Kd_pos * e_vel
                   + acc_des)
        # force the rotors must produce (world): m*a + m*g*z_hat (gravity comp.)
        F_des = self.m * acc_cmd + self.m * self.g * self._z_w  # (N, 3)

        # ------------- Map force -> total thrust + desired attitude ---------
        R = matrix_from_quat(quat)                # (N, 3, 3), columns = body axes in world
        b3 = R[:, :, 2]                           # current body z-axis (world)
        thrust = (F_des * b3).sum(dim=-1)         # project desired force onto body z  (N,)
        thrust = torch.clamp(thrust, min=0.0)     # rotors can only push

        F_norm = torch.linalg.norm(F_des, dim=-1, keepdim=True).clamp(min=1e-6)
        b3_des = F_des / F_norm                    # desired body z-axis
        b1_c = torch.stack([torch.cos(yaw_des), torch.sin(yaw_des),
                            torch.zeros_like(yaw_des)], dim=-1)   # desired heading
        b2_des = torch.cross(b3_des, b1_c, dim=-1)
        b2_des = b2_des / torch.linalg.norm(b2_des, dim=-1, keepdim=True).clamp(min=1e-6)
        b1_des = torch.cross(b2_des, b3_des, dim=-1)
        R_des = torch.stack([b1_des, b2_des, b3_des], dim=-1)     # (N, 3, 3), columns

        # ---------------- Inner loop: attitude PID -> torque ----------------
        # attitude error on SO(3), expressed in the body frame
        err_mat = 0.5 * (torch.bmm(R_des.transpose(1, 2), R)
                         - torch.bmm(R.transpose(1, 2), R_des))
        e_R = vee(err_mat)                         # (N, 3)
        e_omega = ang_vel_b                        # omega_des = 0
        self._int_att = torch.clamp(self._int_att + e_R * dt,
                                    -self._att_int_limit, self._att_int_limit)
        ang_acc_cmd = -(self.Kp_att * e_R
                        + self.Kd_att * e_omega
                        + self.Ki_att * self._int_att)
        # tau = J * alpha_cmd + omega x (J omega)  (gyroscopic feed-forward)
        J_omega = torch.einsum("ij,nj->ni", self.J, ang_vel_b)
        gyro = torch.cross(ang_vel_b, J_omega, dim=-1)
        tau = torch.einsum("ij,nj->ni", self.J, ang_acc_cmd) + gyro

        # ---------------- Pack external wrench (body frame) -----------------
        forces = torch.zeros(self.N, 1, 3, device=self.device)
        torques = torch.zeros(self.N, 1, 3, device=self.device)
        forces[:, 0, 2] = thrust
        torques[:, 0, :] = tau
        return forces, torques


def compute_reference(t, mode, radius, period, height, num_envs, device):
    """Closed-form reference trajectory.

    Returns pos_des (N,3), vel_des (N,3), acc_des (N,3), yaw_des (N,).
    """
    if mode == "hover":
        pos = torch.tensor([0.0, 0.0, height], device=device).repeat(num_envs, 1)
        vel = torch.zeros(num_envs, 3, device=device)
        acc = torch.zeros(num_envs, 3, device=device)
    else:  # circle
        w = 2.0 * math.pi / period
        c, s = math.cos(w * t), math.sin(w * t)
        pos = torch.tensor([radius * c, radius * s, height], device=device).repeat(num_envs, 1)
        vel = torch.tensor([-radius * w * s, radius * w * c, 0.0], device=device).repeat(num_envs, 1)
        acc = torch.tensor([-radius * w * w * c, -radius * w * w * s, 0.0], device=device).repeat(num_envs, 1)
    yaw = torch.zeros(num_envs, device=device)
    return pos, vel, acc, yaw


def get_base_inertia(robot, base_idx, device):
    """Read the base-link inertia from PhysX, with a Crazyflie fallback."""
    try:
        inertias = robot.root_physx_view.get_inertias()      # (N, num_bodies, 9)
        J = inertias[0, base_idx].reshape(3, 3).to(device).float()
        if torch.count_nonzero(J) == 0:
            raise ValueError("zero inertia")
        return J
    except Exception:
        # Crazyflie 2.0 typical inertia [kg m^2]
        return torch.diag(torch.tensor([1.4e-5, 1.4e-5, 2.17e-5], device=device))


class Ros2TrajPublisher:
    """Lightweight rclpy publisher for live trajectory visualization.

    Topics (frame "world"):
      /quad/odom       nav_msgs/Odometry      actual pose + twist
      /quad/pose_des   geometry_msgs/PoseStamped   reference pose
      /quad/path       nav_msgs/Path          actual trajectory trail
      /quad/path_des   nav_msgs/Path          reference trajectory trail
    """

    def __init__(self, frame_id="world", path_max=4000):
        import math as _math
        # Pull in Isaac Sim's *internal* ROS2 Humble libs (built for Python 3.11),
        # so rclpy resolves to the bundled 3.11 build instead of the apt 3.10 one.
        try:
            from isaacsim.core.utils.extensions import enable_extension
            enable_extension("isaacsim.ros2.bridge")
        except Exception:
            pass
        import rclpy
        from nav_msgs.msg import Odometry, Path
        from geometry_msgs.msg import PoseStamped
        from std_msgs.msg import Float32MultiArray

        self._math = _math
        self.rclpy = rclpy
        self.Odometry, self.Path, self.PoseStamped = Odometry, Path, PoseStamped
        self.Float32MultiArray = Float32MultiArray

        rclpy.init()
        self.node = rclpy.create_node("isaaclab_quadcopter")
        self.pub_odom = self.node.create_publisher(Odometry, "/quad/odom", 10)
        self.pub_des = self.node.create_publisher(PoseStamped, "/quad/pose_des", 10)
        self.pub_path = self.node.create_publisher(Path, "/quad/path", 10)
        self.pub_path_des = self.node.create_publisher(Path, "/quad/path_des", 10)
        self.pub_motor = self.node.create_publisher(Float32MultiArray, "/quad/motor_thrusts", 10)

        self.frame_id = frame_id
        self.path_max = path_max
        self.path = Path(); self.path.header.frame_id = frame_id
        self.path_des = Path(); self.path_des.header.frame_id = frame_id

    def _pose(self, stamp, p, q_xyzw):
        ps = self.PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = self.frame_id
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, p)
        (ps.pose.orientation.x, ps.pose.orientation.y,
         ps.pose.orientation.z, ps.pose.orientation.w) = map(float, q_xyzw)
        return ps

    def clear_paths(self):
        self.path.poses.clear()
        self.path_des.poses.clear()

    def publish_motors(self, f):
        msg = self.Float32MultiArray()
        msg.data = [float(v) for v in f]
        self.pub_motor.publish(msg)

    def publish(self, pos, quat_wxyz, lin_vel, ang_vel, pos_des, yaw_des):
        stamp = self.node.get_clock().now().to_msg()
        # actual orientation: Isaac (w,x,y,z) -> ROS (x,y,z,w)
        q_act = (quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0])
        # desired orientation: yaw only
        qz, qw = self._math.sin(yaw_des * 0.5), self._math.cos(yaw_des * 0.5)
        q_des = (0.0, 0.0, qz, qw)

        # --- actual odometry ---
        odom = self.Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.frame_id
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = map(float, pos)
        (odom.pose.pose.orientation.x, odom.pose.pose.orientation.y,
         odom.pose.pose.orientation.z, odom.pose.pose.orientation.w) = map(float, q_act)
        odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z = map(float, lin_vel)
        odom.twist.twist.angular.x, odom.twist.twist.angular.y, odom.twist.twist.angular.z = map(float, ang_vel)
        self.pub_odom.publish(odom)

        # --- desired pose ---
        ps_des = self._pose(stamp, pos_des, q_des)
        self.pub_des.publish(ps_des)

        # --- path trails ---
        self.path.poses.append(self._pose(stamp, pos, q_act))
        self.path_des.poses.append(ps_des)
        if len(self.path.poses) > self.path_max:
            self.path.poses = self.path.poses[-self.path_max:]
            self.path_des.poses = self.path_des.poses[-self.path_max:]
        self.path.header.stamp = stamp
        self.path_des.header.stamp = stamp
        self.pub_path.publish(self.path)
        self.pub_path_des.publish(self.path_des)

        self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def shutdown(self):
        self.node.destroy_node()
        self.rclpy.shutdown()


def build_allocation(robot, device, k_torque=0.006, spin=(1.0, -1.0, 1.0, -1.0)):
    """Build the inverse allocation matrix mapping [T, tx, ty, tz] -> 4 rotor thrusts.

    Each rotor i at body-frame offset (x_i, y_i) producing thrust f_i (+z) gives:
        T  = sum f_i
        tx = sum  y_i f_i           (roll)
        ty = sum -x_i f_i           (pitch)
        tz = sum spin_i * k_torque * f_i   (yaw from rotor drag)
    So [T, tx, ty, tz]^T = A f, and f = A^{-1} [T, tx, ty, tz]^T.

    NOTE: `spin` (rotor turn directions) and `k_torque` (drag/thrust ratio) are
    model assumptions; only the yaw row depends on them. Flip `spin` if yaw looks
    inverted. Geometry (x_i, y_i) is read from the simulation.
    """
    prop_ids, prop_names = robot.find_bodies("m.*_prop")
    try:
        prop_pos_w = robot.data.body_pos_w[0, prop_ids]      # (4, 3) world, level state
        r = (prop_pos_w - robot.data.root_pos_w[0]).cpu()    # body ~ world at level
        x, y = r[:, 0].clone(), r[:, 1].clone()
        if torch.linalg.norm(r) < 1e-6:
            raise ValueError
    except Exception:
        # Crazyflie 2.0 fallback: motors on a ~0.0325 m square (X-config)
        d = 0.0325
        x = torch.tensor([d, -d, -d, d])
        y = torch.tensor([d, d, -d, -d])
        prop_names = ["m1_prop", "m2_prop", "m3_prop", "m4_prop"]

    sp = torch.tensor(spin)
    A = torch.stack([torch.ones(4), y, -x, sp * k_torque], dim=0)   # (4, 4): f -> [T,tx,ty,tz]
    A_inv = torch.linalg.inv(A).to(device)
    return A_inv, prop_names


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.0, 0.0, 0.5])

    # Spawn things into stage
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Robots
    robot_cfg = CRAZYFLIE_CFG.replace(prim_path="/World/Crazyflie")
    robot_cfg.spawn.func("/World/Crazyflie", robot_cfg.spawn, translation=robot_cfg.init_state.pos)

    # create handles for the robots
    robot = Articulation(robot_cfg)

    # Play the simulator
    sim.reset()

    # ---- Build the controller ------------------------------------------------
    device = sim.device
    num_envs = robot.num_instances

    # base (root) body that we apply the wrench on
    base_ids, _ = robot.find_bodies("body")
    if len(base_ids) == 0:
        base_ids = [0]
    base_idx = base_ids[0]

    robot_mass = robot.root_physx_view.get_masses()[0].sum().item()
    gravity = torch.tensor(sim.cfg.gravity, device=device).norm().item()
    inertia = get_base_inertia(robot, base_idx, device)

    controller = CascadePIDController(
        num_envs=num_envs, mass=robot_mass, gravity=gravity,
        inertia=inertia, device=device,
    )

    print("[INFO]: Setup complete...")
    print(f"[INFO]: mass = {robot_mass:.5f} kg, g = {gravity:.3f} m/s^2")
    print(f"[INFO]: base inertia diag = {torch.diagonal(inertia).tolist()}")
    print(f"[INFO]: reference mode = '{args_cli.mode}'")

    # ---- Optional ROS2 trajectory publisher ---------------------------------
    ros_pub = None
    if not args_cli.no_ros2:
        try:
            ros_pub = Ros2TrajPublisher()
            print("[INFO]: ROS2 publisher up: /quad/odom, /quad/pose_des, /quad/path, /quad/path_des")
        except Exception as exc:  # rclpy missing / ROS2 not sourced
            print(f"[WARN]: ROS2 disabled ({exc}). Did you 'source /opt/ros/humble/setup.bash'?")
            ros_pub = None

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # one-time initial state (no periodic resets)
    joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
    robot.write_root_velocity_to_sim(robot.data.default_root_state[:, 7:])
    robot.reset()
    controller.reset()

    # control allocation: [T, tx, ty, tz] -> per-rotor thrusts (display only)
    A_inv, prop_names = build_allocation(robot, device)
    print(f"[INFO]: rotor order = {prop_names}")

    # Simulate physics
    while simulation_app.is_running():
        # closed-form reference at the current time
        pos_des, vel_des, acc_des, yaw_des = compute_reference(
            sim_time, args_cli.mode, args_cli.radius, args_cli.period,
            args_cli.ref_height, num_envs, device,
        )

        # closed-loop cascade-PID control law
        forces, torques = controller.compute(
            pos=robot.data.root_pos_w,
            quat=robot.data.root_quat_w,
            lin_vel=robot.data.root_lin_vel_w,
            ang_vel_b=robot.data.root_ang_vel_b,
            pos_des=pos_des, vel_des=vel_des, acc_des=acc_des, yaw_des=yaw_des,
            dt=sim_dt,
        )

        # apply the wrench on the root body (body frame: thrust along +z, torque tau)
        robot.set_external_force_and_torque(forces, torques, body_ids=base_ids)
        robot.write_data_to_sim()

        # per-rotor thrusts via control allocation:  f = A^{-1} [T, tx, ty, tz]
        u = torch.cat([forces[:, 0, 2:3], torques[:, 0, :]], dim=-1)   # (N, 4)
        motor_thrusts = torch.einsum("ij,nj->ni", A_inv, u)            # (N, 4)
        motor_thrusts = torch.clamp(motor_thrusts, min=0.0)

        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        robot.update(sim_dt)

        # publish trajectories + per-rotor thrusts for live plotting
        if ros_pub is not None and (count % args_cli.ros2_decim == 0):
            ros_pub.publish(
                pos=robot.data.root_pos_w[0].tolist(),
                quat_wxyz=robot.data.root_quat_w[0].tolist(),
                lin_vel=robot.data.root_lin_vel_w[0].tolist(),
                ang_vel=robot.data.root_ang_vel_b[0].tolist(),
                pos_des=pos_des[0].tolist(),
                yaw_des=float(yaw_des[0]),
            )
            ros_pub.publish_motors(motor_thrusts[0].tolist())

        # console readout of per-rotor thrusts (every ~0.5 s)
        if count % 100 == 0:
            f = motor_thrusts[0].tolist()
            T = float(forces[0, 0, 2])
            print(f"[t={sim_time:6.2f}s] T={T:.4f} N | rotors [N] = "
                  f"{f[0]:.4f} {f[1]:.4f} {f[2]:.4f} {f[3]:.4f}")

    if ros_pub is not None:
        ros_pub.shutdown()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
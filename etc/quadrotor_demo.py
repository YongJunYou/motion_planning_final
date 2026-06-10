import argparse
import math
import os
from dataclasses import dataclass

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Tiltrotor flight demo.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--mode", type=str, default="hover", choices=["hover", "circle"])
parser.add_argument("--radius", type=float, default=2.0, help="Circle radius [m].")
parser.add_argument("--period", type=float, default=8.0, help="Circle period [s].")
parser.add_argument("--ref_height", type=float, default=1.8, help="Hover altitude [m].")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import omni.usd
from pxr import Gf, Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat


# ----------------------------------------------------------------------------
# 로봇 home 상태로 각도기준 설정
# ----------------------------------------------------------------------------
ARM_HOME = {
    "dof_l1": 0.0, "dof_l2": 0.0, "dof_l3": 0.0, "dof_l4": 0.0,
    "dof_r1": 0.0, "dof_r2": 0.0, "dof_r3": 0.0, "dof_r4": 0.0,
}

TILT_HOME = {
    "dof_lb1": 0.0, "dof_lb2": 0.0,
    "dof_lf1": 0.0, "dof_lf2": 0.0,
    "dof_rb1": 0.0, "dof_rb2": 0.0,
    "dof_rf1": 0.0, "dof_rf2": 0.0,
}

# ----------------------------------------------------------------------------
# 로터 위치
# ----------------------------------------------------------------------------
ROTOR_XY = torch.tensor([
    [-0.18, +0.18],   # lb (lb_link2_01)
    [-0.18, -0.18],   # lf (lf_link2_01)
    [+0.18, +0.18],   # rb (rb_link2_01)
    [+0.18, -0.18],   # rf (rf_link2_01)
])

# ----------------------------------------------------------------------------
# moment of inertia of the base (kg*m^2), onshape에서 받아온 값
# ---------------------------------------------------------------------------
J_BASE_DIAG = torch.tensor([0.0819, 0.1563, 0.2341])

# ----------------------------------------------------------------------------
 # rotor drag torque / thrust ratio [m], 대충 넣음
# ---------------------------------------------------------------------------
K_DRAG = 0.02 

# ----------------------------------------------------------------------------
# Visual propeller spin (physics에 영향 없음)
# ----------------------------------------------------------------------------
PROP_DUCTS = ["lb", "lf", "rb", "rf"]            # thrusts의 열 인덱스 순서
PROP_LEAVES = ["_prop_up", "_prop_dn"]           # 덕트당 동축 프롭 2개 (USD prim 이름 suffix)
PROP_SIGN = [+1, -1]                             # 동축 쌍(up/dn)은 서로 반대 회전
VIS_RATE = 1500.0                                # 호버 추력 기준 회전속도 [deg/s], 취향껏

# ----------------------------------------------------------------------------
# Visual propeller spin (physics에 영향 없음)
# ----------------------------------------------------------------------------
TILTROTOR_CFG = ArticulationCfg(
    # 스폰방식 지정
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/yyj/motion_planning_final/dual_arm_final.usd",   
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            articulation_enabled=True,           # PhysX articulation(reduced-coordinate 강체 체인)으로 시뮬레이션(고정값)
            fix_root_link=False,                 # 고정 또는 비행
            enabled_self_collisions=False,       # 팔 사이 충돌 끄기
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.5),   # 스폰위치
        rot=(1.0, 0.0, 0.0, 0.0), # 스폰자세
        joint_pos={**TILT_HOME, **ARM_HOME}, #스폰조인트값
    ),
    actuators={
        "tilt": ImplicitActuatorCfg(
            joint_names_expr=["dof_[lr][bf][12]"],
            stiffness=100.0, damping=10.0, effort_limit_sim=20.0, # 관절 PD 제어, P=stiffness, D=damping, 각도및 각속도는 라디안 단위일듯, 20N 리밋
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=["dof_[lr][1-4]"],
            stiffness=1000.0, damping=100.0, effort_limit_sim=60.0, # 관절 PD 제어, P=stiffness, D=damping, 각도및 각속도는 라디안 단위일듯, 60N 리밋
        ),
    },
)
# ----------------------------------------------------------------------------
# isaaclab config
# ---------------------------------------------------------------------------
@configclass
class FlightSceneCfg(InteractiveSceneCfg):
    # 씬 관련 설정
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)))
    robot: ArticulationCfg = TILTROTOR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

# ----------------------------------------------------------------------------
# control loop state
# ----------------------------------------------------------------------------
@dataclass
class ControlLoopState:
    t: float
    count: int
    device: torch.device
    N: int
    default_q: torch.Tensor
    spins: torch.Tensor
    A_inv: torch.Tensor
    thrust_hover: float
    thrust_max: float
    p0: torch.Tensor
    ramp_time: float
    sim_dt: float

# ----------------------------------------------------------------------------
# 프로펠러 시각화용
# ----------------------------------------------------------------------------
class PropSpinner:
    def __init__(self, stage, num_envs):
        self.items = []    # (xform_op, pivot, spin_axis, env, duct, sign)
        self.angles = []
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])

        name2info = {}
        for d, duct in enumerate(PROP_DUCTS):
            for j, suffix in enumerate(PROP_LEAVES):
                name2info[duct + suffix] = (d, PROP_SIGN[j])

        for e in range(num_envs):
            root_path = f"/World/envs/env_{e}/Robot"
            root = stage.GetPrimAtPath(root_path)
            if not root.IsValid():
                print(f"[WARN] robot root not found: {root_path}")
                continue
            n_before = len(self.items)
            for prim in Usd.PrimRange(root):
                info = name2info.get(prim.GetName())
                if info is None or not prim.IsA(UsdGeom.Xformable):
                    continue  # 이름이 {duct}_prop_{up,dn} 인 Xformable prim만
                duct, sgn = info
                if f"/{PROP_DUCTS[duct]}_link2_01/" not in str(prim.GetPath()):
                    print(f"[SKIP] name matched but outside duct subtree: {prim.GetPath()}")
                    continue
                for schema in list(prim.GetAppliedSchemas()):
                    if "Collision" in schema or "Physx" in schema:
                        prim.RemoveAppliedSchema(schema)
                pivot, spin_axis = self._axis_from_mesh(prim, cache)
                xf = UsdGeom.Xformable(prim)
                op = xf.AddTransformOp(opSuffix="propspin")  # 맨 뒤 = 가장 안쪽(geometry에 먼저 적용)
                self.items.append((op, pivot, spin_axis, e, duct, sgn))
                self.angles.append(0.0)
                print(f"[SPIN] env_{e} duct={PROP_DUCTS[duct]} sign={sgn:+d} -> {prim.GetPath()}")
            n_expected = len(PROP_DUCTS) * len(PROP_LEAVES)
            if len(self.items) - n_before != n_expected:
                print(f"[WARN] env_{e}: expected {n_expected} prop prims, "
                      f"got {len(self.items) - n_before}")
        print(f"[INFO]: PropSpinner attached to {len(self.items)} visual prims")

    @staticmethod
    def _axis_from_mesh(prim, cache):
        pts_list = []
        xf_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        L2W = xf_cache.GetLocalToWorldTransform(prim)
        base_inv = L2W.GetInverse()
        for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if not p.IsA(UsdGeom.Mesh):
                continue
            attr = UsdGeom.Mesh(p).GetPointsAttr()
            raw = attr.Get() if attr else None
            if raw is None or len(raw) < 3:
                continue
            # 메시 -> 프롭 prim 프레임 변환 (Gf는 row-vector 컨벤션: x' = x * M)
            M = xf_cache.GetLocalToWorldTransform(p) * base_inv
            R = np.array([[M[i][j] for j in range(3)] for i in range(3)])
            t = np.array([M[3][0], M[3][1], M[3][2]])
            pts_list.append(np.asarray(raw, dtype=np.float64) @ R + t)
        if pts_list:
            pts = np.concatenate(pts_list, axis=0)
            ctr = pts.mean(axis=0)
            _, _, Vt = np.linalg.svd(pts - ctr, full_matrices=False)
            axis = Vt[-1]  # 최소 특이방향 = 디스크 법선
            Rw = np.array([[L2W[i][j] for j in range(3)] for i in range(3)])
            if (axis @ Rw)[2] < 0:
                axis = -axis
            return Gf.Vec3d(*ctr.tolist()), Gf.Vec3d(*axis.tolist())
        print(f"[WARN] no mesh points under {prim.GetPath()}, fallback to bbox+z")
        bound = cache.ComputeUntransformedBound(prim)
        pivot = Gf.Vec3d(bound.ComputeAlignedBox().GetMidpoint())
        return pivot, Gf.Vec3d(0, 0, 1)

    def update(self, thrusts, spins, thrust_hover, dt):
        for i, (op, c, axis, e, duct, sgn) in enumerate(self.items):
            rate = VIS_RATE * float(thrusts[e, duct]) / thrust_hover
            self.angles[i] = (self.angles[i] + sgn * float(spins[duct]) * rate * dt) % 360.0
            R = Gf.Matrix4d().SetRotate(Gf.Rotation(axis, self.angles[i]))
            M = Gf.Matrix4d().SetTranslate(-c) * R * Gf.Matrix4d().SetTranslate(c)
            op.Set(M)

# ----------------------------------------------------------------------------
# ROS2 debug publisher
# ----------------------------------------------------------------------------
class Ros2CurrentPublisher:
    """Publish current sim values for ROS2 debugging.

    Topics:
      /tiltrotor/current/odom             nav_msgs/Odometry
      /tiltrotor/current/pose             geometry_msgs/PoseStamped
      /tiltrotor/current/path             nav_msgs/Path
      /tiltrotor/current/rotor_thrusts    std_msgs/Float32MultiArray
      /tiltrotor/current/wrench           geometry_msgs/WrenchStamped, CAD-origin body wrench
    """

    def __init__(self, frame_id="world", child_frame_id="base_link", path_max=4000,
                 spins=None, k_drag=K_DRAG):
        # Pull in Isaac Sim's internal ROS2 Humble libs before importing rclpy.
        # This avoids loading a system ROS2 rclpy built for a different Python.
        os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
        try:
            from isaacsim.core.utils.extensions import enable_extension
            enable_extension("isaacsim.ros2.bridge")
            for _ in range(5):
                simulation_app.update()
        except Exception as exc:
            print(f"[WARN]: Failed to enable Isaac Sim ROS2 bridge extension ({exc}).")

        import rclpy
        from geometry_msgs.msg import PoseStamped, WrenchStamped
        from nav_msgs.msg import Odometry, Path
        from std_msgs.msg import Float32MultiArray

        if not rclpy.ok():
            rclpy.init(args=[])

        self.rclpy = rclpy
        self.Odometry = Odometry
        self.Path = Path
        self.PoseStamped = PoseStamped
        self.WrenchStamped = WrenchStamped
        self.Float32MultiArray = Float32MultiArray

        self.node = rclpy.create_node("isaaclab_tiltrotor_debug")
        self.pub_odom = self.node.create_publisher(Odometry, "/tiltrotor/current/odom", 10)
        self.pub_pose = self.node.create_publisher(PoseStamped, "/tiltrotor/current/pose", 10)
        self.pub_path = self.node.create_publisher(Path, "/tiltrotor/current/path", 10)
        self.pub_thrusts = self.node.create_publisher(
            Float32MultiArray, "/tiltrotor/current/rotor_thrusts", 10)
        self.pub_wrench = self.node.create_publisher(
            WrenchStamped, "/tiltrotor/current/wrench", 10)

        self.frame_id = frame_id
        self.child_frame_id = child_frame_id
        self.path_max = path_max
        self.spins = (spins.detach().clone().float().cpu()
                      if spins is not None else torch.tensor([+1.0, -1.0, -1.0, +1.0]))
        self.k_drag = float(k_drag)
        self.path = Path()
        self.path.header.frame_id = frame_id

        print(f"[INFO]: rclpy loaded from: {getattr(rclpy, '__file__', '<unknown>')}")
        print(f"[INFO]: ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '0')} "
              f"RMW_IMPLEMENTATION={os.environ.get('RMW_IMPLEMENTATION', '<unset>')}")
        self.rclpy.spin_once(self.node, timeout_sec=0.0)
        self.print_local_topics()

    def print_local_topics(self):
        topic_names = [name for name, _ in self.node.get_topic_names_and_types()
                       if name.startswith("/tiltrotor")]
        if topic_names:
            print("[INFO]: ROS2 topics visible from publisher node:")
            for name in sorted(topic_names):
                print(f"        {name}")
        else:
            print("[WARN]: ROS2 publisher node does not see /tiltrotor topics yet.")

    def _pose_msg(self, stamp, pos, quat_wxyz):
        pose = self.PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.frame_id
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = map(float, pos)

        # Isaac stores quaternions as (w, x, y, z), ROS messages use (x, y, z, w).
        pose.pose.orientation.x = float(quat_wxyz[1])
        pose.pose.orientation.y = float(quat_wxyz[2])
        pose.pose.orientation.z = float(quat_wxyz[3])
        pose.pose.orientation.w = float(quat_wxyz[0])
        return pose

    def _wrench_msg(self, stamp, thrusts):
        thrusts_cpu = thrusts.detach().float().cpu()
        forces, torques = compute_cad_wrench_from_thrusts(
            thrusts_cpu, self.spins, self.k_drag, thrusts_cpu.device)
        force = forces[0, 0]
        torque = torques[0, 0]

        wrench = self.WrenchStamped()
        wrench.header.stamp = stamp
        wrench.header.frame_id = self.child_frame_id
        wrench.wrench.force.x, wrench.wrench.force.y, wrench.wrench.force.z = map(float, force)
        wrench.wrench.torque.x, wrench.wrench.torque.y, wrench.wrench.torque.z = map(float, torque)
        return wrench

    def publish(self, robot, thrusts):
        stamp = self.node.get_clock().now().to_msg()
        pos = robot.data.root_pos_w[0].tolist()
        quat = robot.data.root_quat_w[0].tolist()
        lin_vel = robot.data.root_lin_vel_w[0].tolist()
        ang_vel_b = robot.data.root_ang_vel_b[0].tolist()

        pose = self._pose_msg(stamp, pos, quat)
        self.pub_pose.publish(pose)

        odom = self.Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.frame_id
        odom.child_frame_id = self.child_frame_id
        odom.pose.pose = pose.pose
        odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z = map(float, lin_vel)
        odom.twist.twist.angular.x, odom.twist.twist.angular.y, odom.twist.twist.angular.z = map(float, ang_vel_b)
        self.pub_odom.publish(odom)

        thrust_msg = self.Float32MultiArray()
        thrust_msg.data = [float(v) for v in thrusts[0].tolist()]
        self.pub_thrusts.publish(thrust_msg)
        self.pub_wrench.publish(self._wrench_msg(stamp, thrusts))

        self.path.poses.append(pose)
        if len(self.path.poses) > self.path_max:
            self.path.poses = self.path.poses[-self.path_max:]
        self.path.header.stamp = stamp
        self.pub_path.publish(self.path)

        self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def shutdown(self):
        self.node.destroy_node()
        if self.rclpy.ok():
            self.rclpy.shutdown()

# ----------------------------------------------------------------------------
# Drone cascade PID (position -> attitude) producing [thrust_total, tau]
# ----------------------------------------------------------------------------
class DronePID:
    def __init__(self, num_envs, mass, gravity, inertia, device):
        self.N, self.m, self.g = num_envs, float(mass), float(gravity)
        self.J = inertia
        self.device = device

        # 위치제어 PID Gains 
        self.Kp_pos = torch.tensor([7.5, 7.5, 18.0], device=device)
        self.Ki_pos = torch.tensor([1.2, 1.2, 4.5], device=device)
        self.Kd_pos = torch.tensor([5.2, 5.2, 9.0], device=device)

        # 자세제어 PID Gains 
        self.Kp_att = torch.tensor([300.0, 300.0, 90.0], device=device)
        self.Kd_att = torch.tensor([35.0, 35.0, 18.0], device=device)

        # 내부변수
        self._int_e_p = torch.zeros(num_envs, 3, device=device) # I제어용 적분기
        self._e_3 = torch.tensor([0.0, 0.0, 1.0], device=device) # e_3

    def reset(self):
        self._int_e_p.zero_()

    def compute(self, pos, quat, vel, omega_b, pos_d, vel_d, acc_d, yaw_d, dt):  #TODO: 쿼드로터로 되어있는데 틸트로터에 맞게 변경.
        R = matrix_from_quat(quat)

        # 위치 제어기
        e_p, e_v = pos_d - pos, vel_d - vel
        self._int_e_p = torch.clamp(self._int_e_p + e_p * dt, -1.5, 1.5)
        a_cmd = self.Kp_pos*e_p + self.Ki_pos*self._int_e_p + self.Kd_pos*e_v + acc_d

        # 다이나믹스
        F_des = self.m*a_cmd + self.m*self.g*self._e_3          
        b3 = R @ self._e_3                                        
        thrust_total = torch.clamp((F_des * b3).sum(-1), min=0.0)   # total thrust, (N,) / num_envs=1이면 (1,)

        # 자세계산용 방향벡터
        Fn = torch.linalg.norm(F_des, dim=-1, keepdim=True).clamp(min=1e-6)
        b3d = F_des / Fn
        b1c = torch.stack([torch.cos(yaw_d), torch.sin(yaw_d), torch.zeros_like(yaw_d)], -1)
        b2d = torch.cross(b3d, b1c, dim=-1)
        b2d = b2d / torch.linalg.norm(b2d, dim=-1, keepdim=True).clamp(min=1e-6)
        b1d = torch.cross(b2d, b3d, dim=-1)
        Rd = torch.stack([b1d, b2d, b3d], dim=-1)
        eR = vee(0.5 * (torch.bmm(Rd.transpose(1, 2), R) - torch.bmm(R.transpose(1, 2), Rd)))
        a_ang = -(self.Kp_att * eR + self.Kd_att * omega_b)
        Jw = torch.einsum("ij,nj->ni", self.J, omega_b)

        # 자세제어기
        tau = torch.einsum("ij,nj->ni", self.J, a_ang) + torch.cross(omega_b, Jw, dim=-1)
        return thrust_total, tau

# ----------------------------------------------------------------------------
# math helpers
# ----------------------------------------------------------------------------
def vee(s):
    return torch.stack([s[..., 2, 1], s[..., 0, 2], s[..., 1, 0]], dim=-1)

# ----------------------------------------------------------------------------
# main loop helpers
# ----------------------------------------------------------------------------
def setup_sim_scene_and_robot():
    """시뮬레이션/씬/스피너를 만들고 reset까지 수행한다."""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / 200.0)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.5, 2.5, 2.0], target=[0.0, 0.0, 1.0])

    scene = InteractiveScene(FlightSceneCfg(num_envs=args_cli.num_envs, env_spacing=4.0))

    # AddTransformOp / collision 제거는 USD 구조 변경이라 sim.reset() 이전에 해야 함.
    stage = omni.usd.get_context().get_stage()
    spinner = PropSpinner(stage, args_cli.num_envs)

    sim.reset()
    robot: Articulation = scene["robot"]
    return sim, scene, spinner, robot

def find_robot_handles(robot):
    """루프에서 필요한 body/joint id를 한곳에서 찾는다."""
    root_body_ids = [0]
    base_ids, base_names = robot.find_bodies("base_link_01")
    _rotor_ids, rotor_names = robot.find_bodies(
        ["lb_link2_01", "lf_link2_01", "rb_link2_01", "rf_link2_01"])
    robot.find_joints(["dof_[lr][1-4]"])
    robot.find_joints(["dof_[lr][bf][12]"])
    root_name = robot.body_names[0] if robot.body_names else "<unknown>"
    print(f"[INFO]: wrench body = root body[0] '{root_name}'")
    if len(base_ids) > 0:
        print(f"[INFO]: base_link_01 body match = {list(zip(base_ids, base_names))}")
    else:
        print("[WARN]: base_link_01 body match not found")
    return root_body_ids, rotor_names

def get_mass_and_base_inertia(robot, device):
    """전체 질량과 팔 2개를 제외한 하드코딩 base 관성행렬을 반환한다."""
    m_total = robot.root_physx_view.get_masses()[0].sum().item()
    g = 9.81

    J_base = torch.diag(J_BASE_DIAG).to(device=device, dtype=torch.float32)
    return m_total, g, J_base

def print_wrench_body_com_debug(robot, body_ids):
    """wrench 적용 body 원점 기준 inertial CoM 위치를 한 번 출력한다."""
    base_idx = int(body_ids[0])
    masses = robot.root_physx_view.get_masses()[0]
    body_name = robot.body_names[base_idx] if robot.body_names else f"body_{base_idx}"

    link_pos_w = robot.data.body_pos_w[0, base_idx]
    link_quat_w = robot.data.body_quat_w[0, base_idx]
    com_pos_w = robot.data.body_com_pos_w[0, base_idx]

    R_wb = matrix_from_quat(link_quat_w.unsqueeze(0))[0]
    com_offset_b = R_wb.transpose(0, 1) @ (com_pos_w - link_pos_w)

    print(f"[INFO]: wrench body '{body_name}' mass properties:")
    print(f"        mass = {float(masses[base_idx]):.6f} kg")
    print(f"        link origin world = {link_pos_w.detach().cpu().numpy().round(6)}")
    print(f"        inertial CoM world = {com_pos_w.detach().cpu().numpy().round(6)}")
    print(f"        CoM offset in wrench body frame = "
          f"{com_offset_b.detach().cpu().numpy().round(6)} m")

def print_whole_body_com_debug(robot):
    """전체 articulation CoM이 root link 원점에서 얼마나 떨어졌는지 출력한다."""
    masses = robot.root_physx_view.get_masses()[0].to(robot.device)
    body_com = robot.data.body_com_pos_w[0]
    whole_com_w = (body_com * masses.unsqueeze(-1)).sum(0) / masses.sum()
    root_pos_w = robot.data.root_pos_w[0]
    root_quat_w = robot.data.root_quat_w[0]
    R_wr = matrix_from_quat(root_quat_w.unsqueeze(0))[0]
    com_offset_root = R_wr.transpose(0, 1) @ (whole_com_w - root_pos_w)

    print("[INFO]: whole-body mass properties:")
    print(f"        whole CoM world = {whole_com_w.detach().cpu().numpy().round(6)}")
    print(f"        root link world = {root_pos_w.detach().cpu().numpy().round(6)}")
    print(f"        whole CoM offset in root frame = "
          f"{com_offset_root.detach().cpu().numpy().round(6)} m")

def reset_robot_to_default(robot, scene):
    """초기 root/joint 상태를 쓰고 scene reset을 수행한다."""
    default_q = robot.data.default_joint_pos.clone()
    lims = robot.data.soft_joint_pos_limits[0]
    default_q[0] = default_q[0].clamp(lims[:, 0] + 0.02, lims[:, 1] - 0.02)

    root = robot.data.default_root_state.clone()
    root[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_root_velocity_to_sim(root[:, 7:])
    robot.write_joint_state_to_sim(default_q, torch.zeros_like(default_q))
    scene.reset()
    return default_q

def build_quadrotor_allocator(m_total, g, device): # 의심스러움
    """[thrust_total, tau_x, tau_y, tau_z] -> rotor thrust 4개로 바꾸는 allocation matrix를 만든다."""
    x, y = ROTOR_XY[:, 0].clone(), ROTOR_XY[:, 1].clone()
    spins = torch.tensor([+1.0, -1.0, -1.0, +1.0])  # lb, lf, rb, rf
    A = torch.stack([torch.ones(4), y, -x, spins * K_DRAG], dim=0)
    A_inv = torch.linalg.inv(A).to(device)
    thrust_hover = m_total * g / 4.0
    thrust_max = 3.0 * thrust_hover
    return spins, A_inv, thrust_hover, thrust_max

def make_smooth_reference(t, p0, num_envs, device, ramp_time):
    """takeoff ramp가 적용된 position/velocity/acceleration/yaw reference를 만든다."""
    t_eff = max(t - ramp_time, 0.0)
    p_d, v_d, a_d, yaw_d = reference(
        t_eff, args_cli.mode, args_cli.radius, args_cli.period,
        args_cli.ref_height, num_envs, device)

    s = min(t / ramp_time, 1.0)
    s = s * s * (3.0 - 2.0 * s)  # smoothstep
    p_d = (1.0 - s) * p0 + s * p_d  
    v_d = s * v_d
    a_d = s * a_d
    yaw_d = s * yaw_d
    return p_d, v_d, a_d, yaw_d

def reference(t, mode, radius, period, height, N, device):
    if mode == "hover":
        p_d = torch.tensor([0.0, 0.0, height], device=device).repeat(N, 1)
        v_d = torch.zeros(N, 3, device=device)
        a_d = torch.zeros(N, 3, device=device)
        yaw_d = torch.zeros(N, device=device)
    else: # circle mode
        w = 2 * math.pi / period
        c, sin_t = math.cos(w * t), math.sin(w * t)
        vx = -radius * w * sin_t
        vy = radius * w * c
        p_d = torch.tensor([radius * c, radius * sin_t, height], device=device).repeat(N, 1)
        v_d = torch.tensor([vx, vy, 0.0], device=device).repeat(N, 1)
        a_d = torch.tensor([-radius * w * w * c, -radius * w * w * sin_t, 0.0], device=device).repeat(N, 1)
        speed_xy = math.hypot(vx, vy)
        velocity_heading = math.atan2(vy, vx) if speed_xy > 1.0e-9 else 0.0
        # Desired body -Y axis points along the velocity direction.
        yaw_value = velocity_heading + 0.5 * math.pi
        yaw_d = torch.full((N,), yaw_value, device=device)
    return p_d, v_d, a_d, yaw_d

def compute_rotor_thrusts(ctrl, robot, p_d, v_d, a_d, yaw_d, A_inv, thrust_max, sim_dt):
    """cascade PID 출력 [thrust_total, tau]를 rotor별 thrusts로 변환한다."""
    thrust_total, tau = ctrl.compute(
        robot.data.root_pos_w, robot.data.root_quat_w,
        robot.data.root_link_lin_vel_w, robot.data.root_ang_vel_b,
        p_d, v_d, a_d, yaw_d, sim_dt)

    u = torch.cat([thrust_total.unsqueeze(-1), tau], dim=-1)    # (N, 4)
    thrusts = torch.einsum("ij,nj->ni", A_inv, u).clamp(0.0, thrust_max)
    return thrusts                                                   # (N, 4) / num_envs=1이면 (1, 4)

def compute_body_wrench_and_visual_thrusts(ctrl, robot, p_d, v_d, a_d, yaw_d,
                                           A_inv, thrust_max, sim_dt):
    """controller wrench를 직접 적용하고, rotor thrust는 시각화/로그용으로만 만든다."""
    thrust_total, tau = ctrl.compute(
        robot.data.root_pos_w, robot.data.root_quat_w,
        robot.data.root_link_lin_vel_w, robot.data.root_ang_vel_b,
        p_d, v_d, a_d, yaw_d, sim_dt)

    forces = torch.zeros(robot.num_instances, 1, 3, device=robot.device)
    torques = torch.zeros_like(forces)
    forces[:, 0, 2] = thrust_total
    torques[:, 0, :] = tau

    u = torch.cat([thrust_total.unsqueeze(-1), tau], dim=-1)
    thrusts = torch.einsum("ij,nj->ni", A_inv, u).clamp(0.0, thrust_max)
    return forces, torques, thrusts

def compute_cad_wrench_from_thrusts(thrusts, spins, k_drag, device):
    """ROTOR_XY 기준 rotor thrusts를 CAD 원점 기준 body-frame wrench로 합친다."""
    rotor_xy = ROTOR_XY.to(device=device, dtype=thrusts.dtype)
    spins = spins.to(device=device, dtype=thrusts.dtype)

    forces = torch.zeros(thrusts.shape[0], 1, 3, device=device, dtype=thrusts.dtype)
    torques = torch.zeros_like(forces)
    forces[:, 0, 2] = thrusts.sum(dim=-1)
    torques[:, 0, 0] = (rotor_xy[:, 1] * thrusts).sum(dim=-1)
    torques[:, 0, 1] = (-rotor_xy[:, 0] * thrusts).sum(dim=-1)
    torques[:, 0, 2] = (spins * k_drag * thrusts).sum(dim=-1)
    return forces, torques

def apply_cad_wrench(robot, thrusts, spins, base_ids, k_drag, device):
    """body-frame +z thrust와 body-frame torque를 base body에 직접 적용한다."""
    forces, torques = compute_cad_wrench_from_thrusts(thrusts, spins, k_drag, device)
    robot.set_external_force_and_torque(forces, torques, body_ids=base_ids)

def apply_body_wrench(robot, forces, torques, body_ids):
    """controller가 만든 body-frame +z thrust와 body-frame torque를 그대로 적용한다."""
    robot.set_external_force_and_torque(forces, torques, body_ids=body_ids)

def step_simulation(sim, scene, sim_dt):
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim_dt)

def quat_to_roll_pitch_deg(quat):
    w_, x_, y_, z_ = quat.tolist()
    roll = math.degrees(math.atan2(2 * (w_ * x_ + y_ * z_), 1 - 2 * (x_ * x_ + y_ * y_)))
    pitch = math.degrees(math.asin(max(-1, min(1, 2 * (w_ * y_ - z_ * x_)))))
    return roll, pitch

# ----------------------------------------------------------------------------
# main loop debugging helpers
# ----------------------------------------------------------------------------
def print_startup_log(m_total, thrust_hover, J_base, rotor_names):
    print(f"[INFO]: mass={m_total:.3f} kg, hover/rotor={thrust_hover:.2f} N")
    print(f"[INFO]: J_base diag = {torch.diagonal(J_base).tolist()} kg*m^2")
    print(f"[INFO]: rotors={rotor_names}")
    print(f"[INFO]: rotor xy (CAD, visual/log allocation only):\n{ROTOR_XY.numpy().round(3)}")
    print("[INFO]: control mode = direct body +z thrust and body torque on root body")

def setup_ros2_debug_publisher(spins):
    try:
        ros_pub = Ros2CurrentPublisher(spins=spins)
    except Exception as exc:
        print(f"[WARN]: ROS2 debug topics disabled ({exc}).")
        return None

    print("[INFO]: ROS2 current topics:")
    print("        /tiltrotor/current/odom")
    print("        /tiltrotor/current/pose")
    print("        /tiltrotor/current/path")
    print("        /tiltrotor/current/rotor_thrusts")
    print("        /tiltrotor/current/wrench")
    return ros_pub

def print_periodic_status(t, p_d, robot, thrusts):
    pe = (p_d - robot.data.root_pos_w)[0]
    thrusts_row = thrusts[0].tolist()
    roll, pitch = quat_to_roll_pitch_deg(robot.data.root_quat_w[0])
    print(f"[t={t:6.2f}] pos err=({pe[0]:+.3f},{pe[1]:+.3f},{pe[2]:+.3f}) m | "
          f"rp=({roll:+.1f},{pitch:+.1f}) deg | "
          f"thrusts[N]={thrusts_row[0]:.2f} {thrusts_row[1]:.2f} "
          f"{thrusts_row[2]:.2f} {thrusts_row[3]:.2f}")

def run_control_step(robot, ctrl, wrench_body_ids, spinner, state):
    """while 루프의 한 스텝을 기능 단위로 묶은 함수."""
    robot.set_joint_position_target(state.default_q)

    p_d, v_d, a_d, yaw_d = make_smooth_reference(
        state.t, state.p0, state.N, state.device, state.ramp_time)

    forces, torques, thrusts = compute_body_wrench_and_visual_thrusts(
        ctrl, robot, p_d, v_d, a_d, yaw_d,
        state.A_inv, state.thrust_max, state.sim_dt)

    apply_body_wrench(robot, forces, torques, wrench_body_ids)

    spinner.update(thrusts, state.spins, state.thrust_hover, state.sim_dt)
    return p_d, thrusts

# ----------------------------------------------------------------------------
# main loop
# ----------------------------------------------------------------------------
def main():
    # 시작시 설정
    sim, scene, spinner, robot = setup_sim_scene_and_robot()
    device = robot.device #계산 디바이스 (CPU or GPU)
    robot_num = robot.num_instances # 로봇 대수, 1

    wrench_body_ids, rotor_names = find_robot_handles(robot)
    m_total, g, J_base = get_mass_and_base_inertia(robot, device)
    default_q = reset_robot_to_default(robot, scene)
    print_wrench_body_com_debug(robot, wrench_body_ids)
    print_whole_body_com_debug(robot)

    spins, A_inv, thrust_hover, thrust_max = build_quadrotor_allocator(m_total, g, device)

    ctrl = DronePID(robot_num, m_total, g, J_base, device)

    ramp_time = 4.0
    p0 = robot.data.root_pos_w.clone()
    sim_dt = sim.get_physics_dt()

    print_startup_log(m_total, thrust_hover, J_base, rotor_names)
    ros_pub = setup_ros2_debug_publisher(spins)

    state = ControlLoopState(
        t=0.0,
        count=0,
        device=device,
        N=robot_num,
        default_q=default_q,
        spins=spins,
        A_inv=A_inv,
        thrust_hover=thrust_hover,
        thrust_max=thrust_max,
        p0=p0,
        ramp_time=ramp_time,
        sim_dt=sim_dt,
    )

    if ros_pub is not None:
        ros_pub.publish(robot, torch.zeros(robot_num, 4, device=device))
        ros_pub.print_local_topics()

    # 루프
    try:
        while simulation_app.is_running():
            p_d, thrusts = run_control_step(
                robot, ctrl, wrench_body_ids, spinner, state)

            step_simulation(sim, scene, sim_dt)
            state.t += sim_dt
            state.count += 1

            if ros_pub is not None:
                ros_pub.publish(robot, thrusts)

            if state.count % 200 == 0:
                print_periodic_status(state.t, p_d, robot, thrusts)
    finally:
        if ros_pub is not None:
            ros_pub.shutdown()


if __name__ == "__main__":
    main()
    simulation_app.close()

import argparse
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tiltrotor flight demo (quadrotor mode).")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--mode", type=str, default="hover", choices=["hover", "circle"])
parser.add_argument("--radius", type=float, default=0.8, help="Circle radius [m].")
parser.add_argument("--period", type=float, default=8.0, help="Circle period [s].")
parser.add_argument("--ref_height", type=float, default=1.8, help="Hover altitude [m].")
parser.add_argument("--k_drag", type=float, default=0.02,
                    help="Rotor drag torque / thrust ratio [m] (yaw model).")
parser.add_argument("--max_time", type=float, default=0.0,
                    help="If >0, stop the sim after this many seconds (for tuning).")
parser.add_argument("--arm_mode", type=str, default="sine", choices=["hold", "sine"],
                    help="hold: 팔 고정(기존 동작), sine: 팔 동작으로 호버 외란 테스트.")
parser.add_argument("--arm_profile", type=str, default="outward", choices=["outward", "center_sine"],
                    help="outward: 시작각에서 +방향으로만 움직임(몸 안쪽으로 되돌아가는 명령 방지), "
                         "center_sine: 기존처럼 limit 중앙 기준 사인 스윕.")
parser.add_argument("--arm_amp", type=float, default=0.6,
                    help="팔 3,4번 관절 기본 진폭 [rad] (~34 deg).")
parser.add_argument("--arm_j1_amp", type=float, default=0.5,
                    help="1번 관절 진폭 [rad]. 0이면 1번 관절 고정.")
parser.add_argument("--arm_j2_scale", type=float, default=6.0,
                    help="2번 관절 진폭 배율. 실제 진폭은 arm_amp * arm_j2_scale로 계산 후 limit 안에서 clamp.")
parser.add_argument("--arm_period", type=float, default=3.0,
                    help="팔 사인 주기 [s].")
parser.add_argument("--arm_phase", type=float, default=0.0,
                    help="오른팔 위상차 [deg]. 0=동위상(CoM이 같이 흔들려 외란 최대), "
                         "180=미러(좌우 대칭이라 횡방향 CoM 상쇄).")
parser.add_argument("--arm_start_delay", type=float, default=0.0,
                    help="호버 안정화 후 팔 동작 시작까지 지연 [s] (T_RAMP 기준). 기본 0이면 takeoff ramp 직후 바로 시작.")
parser.add_argument("--arm_direction", type=float, default=1.0,
                    help="이전 실행 옵션 호환용. dof_[lr][2-4]는 항상 시작각 기준 [start, start+span]으로 제한한다.")
parser.add_argument("--arm_limit_span_deg", type=float, default=180.0,
                    help="dof_[lr][2-4]의 허용 범위: 시작 각도 기준 +이 값[deg].")
parser.add_argument("--arm_limit_margin", type=float, default=0.03,
                    help="팔 2~4번 사인궤적이 hard limit 안쪽에서 움직이도록 두는 마진 [rad].")
parser.add_argument("--disable_usd_limit_patch", action="store_true",
                    help="켜면 sim.reset() 전 USD RevoluteJoint limit 패치를 하지 않고 코드 target clamp만 사용.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.arm_direction != 1.0:
    print("[WARN]: --arm_direction is kept only for launch compatibility and is ignored. "
          "dof_[lr][2-4] limits are forced to [start, start+span] = [0,+180] deg.")
args_cli.arm_direction = 1.0

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

from pxr import Usd, UsdGeom, UsdPhysics, Gf
import omni.usd

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat

# ----------------------------------------------------------------------------
# 로봇 설정값
# ----------------------------------------------------------------------------
ARM_HOME = {
    "dof_l1": 0.0, "dof_l2": 0.0, "dof_l3": 0.0, "dof_l4": 0.0,
    "dof_r1": 0.0, "dof_r2": 0.0, "dof_r3": 0.0, "dof_r4": 0.0,
}
TILT_HOME = {j: 0.0 for j in (
    "dof_lb1", "dof_lb2", "dof_lf1", "dof_lf2",
    "dof_rb1", "dof_rb2", "dof_rf1", "dof_rf2",
)}

# 2, 3, 4번 매니퓰레이터 조인트는 "시작 각도 <= q <= 시작 각도 + 180 deg"로 제한한다.
# dof_[lr]1은 기존처럼 연속/회전축 조인트로 취급한다.
ARM_REL_LIMIT_JOINTS = tuple(
    f"dof_{side}{idx}" for side in ("l", "r") for idx in (2, 3, 4)
)

# ----------------------------------------------------------------------------
# 로터 위치
# ----------------------------------------------------------------------------
ROTOR_XY = torch.tensor([
    [-0.18, +0.18],   # lb (lb_link2_01)
    [-0.18, -0.18],   # lf (lf_link2_01)
    [+0.18, +0.18],   # rb (rb_link2_01)
    [+0.18, -0.18],   # rf (rf_link2_01)
])

J_BASE_DIAG = torch.tensor([0.0819, 0.1563, 0.2341])

DAAM_CFG = ArticulationCfg(
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


@configclass
class FlightSceneCfg(InteractiveSceneCfg):
    # 씬 관련 설정
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light",
                         spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)))
    robot: ArticulationCfg = DAAM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# ----------------------------------------------------------------------------
# math helpers
# ----------------------------------------------------------------------------
def vee(s):
    return torch.stack([s[..., 2, 1], s[..., 0, 2], s[..., 1, 0]], dim=-1)


def is_relative_limited_arm_joint(joint_name):
    """True이면 시작 각 기준 [0, +span] deg만 허용하는 팔 조인트이다."""
    return joint_name in ARM_REL_LIMIT_JOINTS


def is_revolute_joint_prim(prim):
    """USD prim이 RevoluteJoint인지 최대한 보수적으로 판정한다."""
    return prim.IsA(UsdPhysics.RevoluteJoint) or prim.GetTypeName() == "PhysicsRevoluteJoint"


def patch_manipulator_limits_in_usd(stage, num_envs, start_angles, span_deg, direction=1.0):
    """PhysX가 실제 hard stop을 갖도록 sim.reset() 전에 USD revolute joint limit을 덮어쓴다.

    USD Physics의 RevoluteJoint limit attribute는 degree 단위이고,
    Isaac/PhysX tensor의 joint position은 radian 단위이다.
    이 함수는 원본 .usd 파일을 저장하지 않고, 현재 열린 stage의 env 복제본만 수정한다.
    """
    total_patched = 0
    for env_i in range(num_envs):
        root_path = f"/World/envs/env_{env_i}/Robot"
        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            print(f"[WARN] robot root not found while patching arm limits: {root_path}")
            continue

        for joint_name in ARM_REL_LIMIT_JOINTS:
            start_rad = float(start_angles.get(joint_name, 0.0))
            start_deg = math.degrees(start_rad)
            end_deg = start_deg + float(direction) * float(span_deg)
            lower_deg = min(start_deg, end_deg)
            upper_deg = max(start_deg, end_deg)
            matched = 0

            for prim in Usd.PrimRange(root):
                prim_path = str(prim.GetPath())
                prim_name = prim.GetName()
                if joint_name != prim_name and joint_name not in prim_name and f"/{joint_name}" not in prim_path:
                    continue
                if not is_revolute_joint_prim(prim):
                    continue

                joint = UsdPhysics.RevoluteJoint(prim)
                joint.CreateLowerLimitAttr().Set(lower_deg)
                joint.CreateUpperLimitAttr().Set(upper_deg)
                matched += 1
                total_patched += 1
                print(f"[LIMIT][USD] env_{env_i} {joint_name}: "
                      f"[{lower_deg:+.2f}, {upper_deg:+.2f}] deg -> {prim_path}")

            if matched == 0:
                print(f"[WARN] no RevoluteJoint prim matched {joint_name} under {root_path}. "
                      "Code-level target/state clamp will still be used.")

    print(f"[INFO]: patched {total_patched} USD revolute joint limit attributes "
          f"for {len(ARM_REL_LIMIT_JOINTS)} relative-limited arm joints")


def make_inner_limits(lo, hi, margin):
    """finite limit만 margin만큼 안쪽으로 줄인다. 너무 좁으면 midpoint로 collapse한다."""
    lo_i = lo.clone()
    hi_i = hi.clone()
    finite_lo = torch.isfinite(lo_i)
    finite_hi = torch.isfinite(hi_i)
    lo_i[finite_lo] += margin
    hi_i[finite_hi] -= margin

    bad = finite_lo & finite_hi & (lo_i > hi_i)
    if bad.any():
        mid = 0.5 * (lo[bad] + hi[bad])
        lo_i[bad] = mid
        hi_i[bad] = mid
    return lo_i, hi_i


def clamp_joint_tensor(q, lo, hi):
    """q: (N, num_joints), lo/hi: (num_joints,)를 broadcast해서 clamp한다."""
    return torch.maximum(torch.minimum(q, hi.unsqueeze(0)), lo.unsqueeze(0))


def build_joint_command_limits(default_q0, usd_lims, arm_ids, arm_names, span_rad, device, direction=1.0):
    """USD limit과 사용자 조건을 합쳐 코드가 보낼 수 있는 최종 command limit을 만든다.

    dof_[lr][2-4]는 반드시 [초기각, 초기각 + span_rad]로 제한한다.
    기존 USD limit이 더 좁은 경우에는 그 교집합을 사용한다.
    """
    cmd_lo = usd_lims[:, 0].clone()
    cmd_hi = usd_lims[:, 1].clone()
    relative_mask = torch.zeros(len(arm_ids), dtype=torch.bool, device=device)
    constrained_ids = []

    for k, (jid, name) in enumerate(zip(arm_ids, arm_names)):
        jid = int(jid)
        if not is_relative_limited_arm_joint(name):
            continue

        relative_mask[k] = True
        constrained_ids.append(jid)
        start = default_q0[jid]
        end = start + float(direction) * span_rad
        lo = torch.minimum(start, end)
        hi = torch.maximum(start, end)

        # USD가 이미 더 좁은 hard/soft limit을 갖고 있으면 그 안쪽으로만 명령한다.
        if torch.isfinite(cmd_lo[jid]):
            lo = torch.maximum(lo, cmd_lo[jid])
        if torch.isfinite(cmd_hi[jid]):
            hi = torch.minimum(hi, cmd_hi[jid])

        if bool((hi < lo).item()):
            print(f"[WARN] empty limit intersection for {name}; holding at start angle. "
                  f"start={math.degrees(float(start)):+.2f} deg, "
                  f"usd=[{math.degrees(float(usd_lims[jid, 0])):+.2f}, "
                  f"{math.degrees(float(usd_lims[jid, 1])):+.2f}] deg")
            hi = lo.clone()

        cmd_lo[jid] = lo
        cmd_hi[jid] = hi

    constrained_ids_t = torch.tensor(constrained_ids, device=device, dtype=torch.long)
    return cmd_lo, cmd_hi, relative_mask, constrained_ids_t


def clamp_actual_joint_state_to_limits(robot, lo, hi, joint_ids_t, tol=1.0e-5):
    """PhysX limit 패치가 실패한 경우에도 실제 joint state가 범위 밖으로 나가지 않게 하는 안전망."""
    if joint_ids_t.numel() == 0:
        return 0

    q = robot.data.joint_pos.clone()
    qd = robot.data.joint_vel.clone()
    lo_j = lo[joint_ids_t].unsqueeze(0)
    hi_j = hi[joint_ids_t].unsqueeze(0)
    q_j = q[:, joint_ids_t]
    q_j_clamped = torch.maximum(torch.minimum(q_j, hi_j), lo_j)
    violated = (q_j < lo_j - tol) | (q_j > hi_j + tol)

    if not violated.any():
        return 0

    q[:, joint_ids_t] = q_j_clamped
    qd[:, joint_ids_t] = 0.0
    robot.write_joint_state_to_sim(q, qd)
    return int(violated.sum().item())


# ----------------------------------------------------------------------------
# Visual propeller spin (physics에 영향 없음)
# ----------------------------------------------------------------------------
PROP_DUCTS = ["lb", "lf", "rb", "rf"]            
PROP_LEAVES = ["_prop_up", "_prop_dn"]          
PROP_SIGN = [+1, -1]                            
VIS_RATE = 1500.0                                # 호버 추력 기준 회전속도 [deg/s], 취향껏


class PropSpinner: # 회전 시각화
    def __init__(self, stage, num_envs):
        self.items = []
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
                    continue 
                duct, sgn = info
                if f"/{PROP_DUCTS[duct]}_link2_01/" not in str(prim.GetPath()):
                    print(f"[SKIP] name matched but outside duct subtree: {prim.GetPath()}")
                    continue

                for schema in list(prim.GetAppliedSchemas()):
                    if "Collision" in schema or "Physx" in schema:
                        prim.RemoveAppliedSchema(schema)
                pivot, spin_axis = self._axis_from_mesh(prim, cache)
                xf = UsdGeom.Xformable(prim)
                op = xf.AddTransformOp(opSuffix="propspin")
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
            M = xf_cache.GetLocalToWorldTransform(p) * base_inv
            R = np.array([[M[i][j] for j in range(3)] for i in range(3)])
            t = np.array([M[3][0], M[3][1], M[3][2]])
            pts_list.append(np.asarray(raw, dtype=np.float64) @ R + t)
        if pts_list:
            pts = np.concatenate(pts_list, axis=0)
            ctr = pts.mean(axis=0)
            _, _, Vt = np.linalg.svd(pts - ctr, full_matrices=False)
            axis = Vt[-1]
            Rw = np.array([[L2W[i][j] for j in range(3)] for i in range(3)])
            if (axis @ Rw)[2] < 0:
                axis = -axis
            return Gf.Vec3d(*ctr.tolist()), Gf.Vec3d(*axis.tolist())
        print(f"[WARN] no mesh points under {prim.GetPath()}, fallback to bbox+z")
        bound = cache.ComputeUntransformedBound(prim)
        pivot = Gf.Vec3d(bound.ComputeAlignedBox().GetMidpoint())
        return pivot, Gf.Vec3d(0, 0, 1)

    def update(self, f, spins, f_hover, dt):
        for i, (op, c, axis, e, duct, sgn) in enumerate(self.items):
            rate = VIS_RATE * float(f[e, duct]) / f_hover
            self.angles[i] = (self.angles[i] + sgn * float(spins[duct]) * rate * dt) % 360.0
            R = Gf.Matrix4d().SetRotate(Gf.Rotation(axis, self.angles[i]))
            M = Gf.Matrix4d().SetTranslate(-c) * R * Gf.Matrix4d().SetTranslate(c)
            op.Set(M)


# ----------------------------------------------------------------------------
# ROS2 debug publisher
# ----------------------------------------------------------------------------
class Ros2CurrentPublisher:
    """main.py와 같은 current topics에 EE 위치와 joint 값을 추가로 publish한다."""

    def __init__(self, robot, ee_body_ids, ee_names, frame_id="world",
                 child_frame_id="base_link", path_max=4000):
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
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float32MultiArray

        if not rclpy.ok():
            rclpy.init(args=[])

        self.rclpy = rclpy
        self.Odometry = Odometry
        self.Path = Path
        self.PoseStamped = PoseStamped
        self.WrenchStamped = WrenchStamped
        self.JointState = JointState
        self.Float32MultiArray = Float32MultiArray

        self.node = rclpy.create_node("isaaclab_tiltrotor_debug")
        self.pub_odom = self.node.create_publisher(Odometry, "/tiltrotor/current/odom", 10)
        self.pub_pose = self.node.create_publisher(PoseStamped, "/tiltrotor/current/pose", 10)
        self.pub_path = self.node.create_publisher(Path, "/tiltrotor/current/path", 10)
        self.pub_thrusts = self.node.create_publisher(
            Float32MultiArray, "/tiltrotor/current/rotor_thrusts", 10)
        self.pub_wrench = self.node.create_publisher(
            WrenchStamped, "/tiltrotor/current/wrench", 10)
        self.pub_ee_positions = self.node.create_publisher(
            Float32MultiArray, "/tiltrotor/current/ee_positions", 10)
        self.pub_joint_states = self.node.create_publisher(
            JointState, "/tiltrotor/current/joint_states", 10)
        self.pub_joint_positions = self.node.create_publisher(
            Float32MultiArray, "/tiltrotor/current/joint_positions", 10)

        self.frame_id = frame_id
        self.child_frame_id = child_frame_id
        self.path_max = path_max
        self.ee_body_ids = ee_body_ids
        self.ee_names = ee_names
        self.joint_names = list(robot.joint_names)
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
        pose.pose.orientation.x = float(quat_wxyz[1])
        pose.pose.orientation.y = float(quat_wxyz[2])
        pose.pose.orientation.z = float(quat_wxyz[3])
        pose.pose.orientation.w = float(quat_wxyz[0])
        return pose

    def _wrench_msg(self, stamp, forces, torques):
        wrench = self.WrenchStamped()
        wrench.header.stamp = stamp
        wrench.header.frame_id = self.child_frame_id
        force = forces.detach().float().cpu()[0, 0]
        torque = torques.detach().float().cpu()[0, 0]
        wrench.wrench.force.x, wrench.wrench.force.y, wrench.wrench.force.z = map(float, force)
        wrench.wrench.torque.x, wrench.wrench.torque.y, wrench.wrench.torque.z = map(float, torque)
        return wrench

    def _ee_positions_msg(self, robot):
        msg = self.Float32MultiArray()
        data = []
        for body_id in self.ee_body_ids:
            pos = robot.data.body_pos_w[0, int(body_id)].detach().float().cpu().tolist()
            data.extend(float(v) for v in pos)
        msg.data = data
        return msg

    def _joint_state_msg(self, stamp, robot):
        msg = self.JointState()
        msg.header.stamp = stamp
        msg.name = self.joint_names
        msg.position = [float(v) for v in robot.data.joint_pos[0].detach().float().cpu().tolist()]
        msg.velocity = [float(v) for v in robot.data.joint_vel[0].detach().float().cpu().tolist()]
        return msg

    def publish(self, robot, thrusts, forces, torques):
        stamp = self.node.get_clock().now().to_msg()
        pos = robot.data.root_pos_w[0].tolist()
        quat = robot.data.root_quat_w[0].tolist()
        lin_vel = robot.data.root_link_lin_vel_w[0].tolist()
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
        self.pub_wrench.publish(self._wrench_msg(stamp, forces, torques))
        self.pub_ee_positions.publish(self._ee_positions_msg(robot))
        self.pub_joint_states.publish(self._joint_state_msg(stamp, robot))

        joint_positions = self.Float32MultiArray()
        joint_positions.data = [float(v) for v in robot.data.joint_pos[0].detach().float().cpu().tolist()]
        self.pub_joint_positions.publish(joint_positions)

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
# Drone cascade PID (position -> attitude) producing [T, tau]
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

        e_p, e_v = pos_d - pos, vel_d - vel
        self._int_e_p = torch.clamp(self._int_e_p + e_p * dt, -1.5, 1.5)
        a_cmd = self.Kp_pos*e_p + self.Ki_pos*self._int_e_p + self.Kd_pos*e_v + acc_d
        F_des = self.m*a_cmd + self.m*self.g*self._e_3          

        b3 = R @ self._e_3                                        
        T = torch.clamp((F_des * b3).sum(-1), min=0.0)              # total thrust

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
        tau = torch.einsum("ij,nj->ni", self.J, a_ang) + torch.cross(omega_b, Jw, dim=-1)
        return T, tau


# ----------------------------------------------------------------------------
def reference(t, mode, radius, period, height, N, device):
    if mode == "hover":
        p = torch.tensor([0.0, 0.0, height], device=device).repeat(N, 1)
        v = torch.zeros(N, 3, device=device)
        a = torch.zeros(N, 3, device=device)
    else:
        w = 2 * math.pi / period
        c, s = math.cos(w * t), math.sin(w * t)
        p = torch.tensor([radius * c, radius * s, height], device=device).repeat(N, 1)
        v = torch.tensor([-radius * w * s, radius * w * c, 0.0], device=device).repeat(N, 1)
        a = torch.tensor([-radius * w * w * c, -radius * w * w * s, 0.0], device=device).repeat(N, 1)
    yaw = torch.zeros(N, device=device)
    return p, v, a, yaw


def find_ee_body_ids(robot):
    """좌/우 매니퓰레이터 EE body를 body name에서 자동으로 찾는다."""
    ee_ids = []
    ee_names = []
    keywords = ("ee", "end", "tip", "tool", "gripper", "hand", "wrist", "tcp")

    for side in ("l", "r"):
        side_candidates = []
        fallback_candidates = []
        for idx, name in enumerate(robot.body_names):
            n = name.lower()
            if any(token in n for token in ("prop", "duct")):
                continue
            if side == "l" and (n.startswith("lb") or n.startswith("lf")):
                continue
            if side == "r" and (n.startswith("rb") or n.startswith("rf")):
                continue
            side_match = (
                n.startswith(f"{side}_")
                or n.startswith(f"{side}link")
                or n.startswith(f"{side}arm")
                or any(n.startswith(f"{side}{i}") for i in range(1, 10))
                or n.startswith("left" if side == "l" else "right")
                or f"_{side}" in n
            )
            if not side_match:
                continue
            if any(k in n for k in keywords):
                side_candidates.append((idx, name))
            if "link" in n or "arm" in n:
                fallback_candidates.append((idx, name))

        candidates = side_candidates or fallback_candidates
        if candidates:
            body_id, body_name = candidates[-1]
            ee_ids.append(body_id)
            ee_names.append(body_name)
        else:
            print(f"[WARN]: {side}-arm EE body not found. Available bodies: {robot.body_names}")

    print(f"[INFO]: EE bodies for ROS2 = {list(zip(ee_ids, ee_names))}")
    return ee_ids, ee_names


def setup_ros2_debug_publisher(robot, ee_body_ids, ee_names):
    try:
        ros_pub = Ros2CurrentPublisher(robot, ee_body_ids, ee_names)
    except Exception as exc:
        print(f"[WARN]: ROS2 debug topics disabled ({exc}).")
        return None

    print("[INFO]: ROS2 current topics:")
    print("        /tiltrotor/current/odom")
    print("        /tiltrotor/current/pose")
    print("        /tiltrotor/current/path")
    print("        /tiltrotor/current/rotor_thrusts")
    print("        /tiltrotor/current/wrench")
    print("        /tiltrotor/current/ee_positions    # [left xyz, right xyz]")
    print("        /tiltrotor/current/joint_states")
    print("        /tiltrotor/current/joint_positions")
    return ros_pub


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / 200.0)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.5, 2.5, 2.0], target=[0.0, 0.0, 1.0])

    scene = InteractiveScene(FlightSceneCfg(num_envs=args_cli.num_envs, env_spacing=4.0))

    # ---- USD hard limits -------------------------------------------------------
    # 중요: PhysX articulation이 만들어지는 sim.reset() 전에 stage의 revolute joint limit을 고친다.
    stage = omni.usd.get_context().get_stage()
    if not args_cli.disable_usd_limit_patch:
        patch_manipulator_limits_in_usd(
            stage, args_cli.num_envs, ARM_HOME, args_cli.arm_limit_span_deg,
            direction=args_cli.arm_direction)

    # ---- visual prop spinner --------------------------------------------------
    # AddTransformOp / collision 제거는 USD 구조 변경이라 sim.reset() "이전"에 해야 함.
    # (sim 시작 후 하면 PhysX tensor view가 invalidate됨)
    spinner = PropSpinner(stage, args_cli.num_envs)

    sim.reset()

    robot: Articulation = scene["robot"]
    device = robot.device
    N = robot.num_instances

    # ---- ids ---------------------------------------------------------------
    # main.py에서 안정화된 방식과 동일하게 controller wrench는 articulation root body에 건다.
    wrench_body_ids = [0]
    root_body_name = robot.body_names[0] if robot.body_names else "<unknown>"
    print(f"[INFO]: wrench body = root body[0] '{root_body_name}'")
    base_ids, base_names = robot.find_bodies("base_link_01")
    if len(base_ids) > 0:
        print(f"[INFO]: base_link_01 body match = {list(zip(base_ids, base_names))}")
    else:
        print("[WARN]: base_link_01 body match not found")
    # inner rotor assemblies (motor+prop) = tilt link2 bodies
    rotor_ids, rotor_names = robot.find_bodies(
        ["lb_link2_01", "lf_link2_01", "rb_link2_01", "rf_link2_01"])
    arm_ids, arm_names = robot.find_joints(["dof_[lr][1-4]"])
    tilt_ids, _ = robot.find_joints(["dof_[lr][bf][12]"])
    ee_body_ids, ee_names = find_ee_body_ids(robot)

    # ---- model parameters ----------------------------------------------------
    masses = robot.root_physx_view.get_masses()[0]
    m_tot = masses.sum().item()
    g = 9.81
    J = torch.diag(J_BASE_DIAG).to(device=device, dtype=torch.float32)

    body_com = robot.data.body_com_pos_w[0].cpu()
    masses_cpu = masses.cpu()
    com0 = (body_com * masses_cpu.unsqueeze(-1)).sum(0) / m_tot
    root_pos_w = robot.data.root_pos_w[0].cpu()
    root_quat_w = robot.data.root_quat_w[0]
    R_wr = matrix_from_quat(root_quat_w.unsqueeze(0))[0].cpu()
    com_offset_root = R_wr.transpose(0, 1) @ (com0 - root_pos_w)
    print("[INFO]: whole-body mass properties:")
    print(f"        whole CoM world = {com0.numpy().round(6)}")
    print(f"        root link world = {root_pos_w.numpy().round(6)}")
    print(f"        whole CoM offset in root frame = {com_offset_root.numpy().round(6)} m")

    # ---- initial state -------------------------------------------------------
    default_q = robot.data.default_joint_pos.clone()
    lims = robot.data.soft_joint_pos_limits[0]
    # 시작각 자체가 2~4번 조인트의 하한이므로, 초기 상태를 margin만큼 밀어 넣지 않는다.
    default_q = clamp_joint_tensor(default_q, lims[:, 0], lims[:, 1])
    root = robot.data.default_root_state.clone()
    root[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_root_velocity_to_sim(root[:, 7:])
    robot.write_joint_state_to_sim(default_q, torch.zeros_like(default_q))
    scene.reset()

    # ---- allocation ----------------
    x, y = ROTOR_XY[:, 0].clone(), ROTOR_XY[:, 1].clone()
    spins = torch.tensor([+1.0, -1.0, -1.0, +1.0])            # lb, lf, rb, rf (flip if yaw runs away)
    A = torch.stack([torch.ones(4), y, -x, spins * args_cli.k_drag], dim=0)
    A_inv = torch.linalg.inv(A).to(device)
    f_hover = m_tot * g / 4.0
    f_max = 3.0 * f_hover

    # 참고용 비교 출력
    com = (robot.data.body_com_pos_w[0] * masses.unsqueeze(-1).to(device)).sum(0) / m_tot
    r_sim = (robot.data.body_com_pos_w[0, rotor_ids] - com)  # (4, 3), world ~ body at level

    ctrl = DronePID(N, m_tot, g, J, device)

    # ---- arm motion schedule --------------------------------------------------
    # 팔을 움직이면 CoM 이동 + 반작용 토크가 생겨 rigid-body 가정의 PID에는
    # 모델링 안 된 외란으로 작용한다. 호버 유지 성능을 보기 위한 테스트 신호.
    # 관절별 사인: q_arm = default + amp * smoothstep * sin(2*pi*t/period + phase)
    # 오른팔(dof_r*)에만 --arm_phase 만큼 위상차를 줘서 동위상/미러를 전환한다.
    T_RAMP = 4.0
    arm_t0 = T_RAMP + args_cli.arm_start_delay
    arm_ids_t = torch.tensor(arm_ids, device=device, dtype=torch.long)
    arm_phase = torch.zeros(len(arm_ids), device=device)
    for k, name in enumerate(arm_names):
        if name.startswith("dof_r"):
            arm_phase[k] = math.radians(args_cli.arm_phase)
    # 관절별 최종 command limit을 계산한다.
    # 핵심: dof_[lr][2-4]는 USD limit이 무엇이든 시작각 기준 한쪽 방향으로만 움직이게 한다.
    # 시작각 0 deg 기준 [0, +180] deg hard limit이 되도록 +방향으로 고정한다.
    cmd_lo, cmd_hi, relative_arm_mask, constrained_arm_ids_t = build_joint_command_limits(
        default_q[0], lims, arm_ids, arm_names,
        math.radians(args_cli.arm_limit_span_deg), device,
        direction=args_cli.arm_direction)

    # 사인 궤적은 hard limit 바로 위/아래를 찍지 않도록 margin 안쪽에서 만든다.
    # 단, 시작 자세는 실제 하한일 수 있으므로 q_arm 계산은 arm_center0(시작각)에서 부드럽게 출발한다.
    arm_lo = cmd_lo[arm_ids_t]
    arm_hi = cmd_hi[arm_ids_t]
    arm_inner_lo, arm_inner_hi = make_inner_limits(arm_lo, arm_hi, args_cli.arm_limit_margin)
    arm_center0 = default_q[0, arm_ids_t].clone()

    finite_arm_limit = torch.isfinite(arm_lo) & torch.isfinite(arm_hi) & ((arm_hi - arm_lo) < 1.0e3)
    rel_center = 0.5 * (arm_inner_lo + arm_inner_hi)
    rel_amp_max = (0.5 * (arm_inner_hi - arm_inner_lo)).clamp(min=0.0)
    nonrel_amp_max = torch.minimum(
        arm_center0 - arm_inner_lo, arm_inner_hi - arm_center0).clamp(min=0.0)

    arm_center = torch.where(relative_arm_mask, rel_center, arm_center0)

    # 요청 반영: dof1도 움직이고, dof2는 더 크게 움직인다.
    # dof2~4는 시작각 기준 한쪽 방향 limit(cmd_lo/cmd_hi)을 유지한다.
    # dof1은 continuous일 수 있으므로 별도 진폭을 주고 기존 USD/soft limit 안에서만 움직인다.
    arm_amp_request = torch.full_like(arm_lo, args_cli.arm_amp)
    for k, name in enumerate(arm_names):
        if name.endswith("1"):
            arm_amp_request[k] = args_cli.arm_j1_amp
        elif name.endswith("2"):
            arm_amp_request[k] = args_cli.arm_amp * args_cli.arm_j2_scale
    arm_amp_max = torch.where(relative_arm_mask, rel_amp_max, nonrel_amp_max)
    arm_amp_j = torch.minimum(arm_amp_request, arm_amp_max).clamp(min=0.0)

    # 좌우 EE 대칭 보정:
    # 2~4번 joint 명령은 그대로 두고, 1번 joint만 좌우 각각 반대 부호로 보낸다.
    # dof_l1=+, dof_r1=- 로 설정해 좌우 1번 관절이 미러 동작을 유지하도록 한다.
    # dof_[lr][2-4]는 모두 +방향, 즉 [0,+180] deg 리밋 안에서만 움직인다.
    arm_motion_sign = torch.ones_like(arm_lo)
    for k, name in enumerate(arm_names):
        if name == "dof_r1":
            arm_motion_sign[k] *= -1.0

    if args_cli.arm_mode == "sine":
        print(f"[INFO]: arm motion test: profile={args_cli.arm_profile}, "
              f"direction={args_cli.arm_direction:+.0f}, "
              f"j1_amp={math.degrees(args_cli.arm_j1_amp):.1f} deg, "
              f"j2_scale={args_cli.arm_j2_scale:.2f}, "
              f"period={args_cli.arm_period:.1f} s, "
              f"r-phase={args_cli.arm_phase:.0f} deg, start at t={arm_t0:.1f} s")
        for k, name in enumerate(arm_names):
            if relative_arm_mask[k]:
                tag = " relative-limit[start,+span]"
            elif finite_arm_limit[k]:
                tag = " usd-limited"
            else:
                tag = " continuous"
            print(f"[INFO]:   {name}: start={math.degrees(arm_center0[k].item()):+.1f} deg, "
                  f"cmd_limit=[{math.degrees(arm_lo[k].item()):+.1f}, "
                  f"{math.degrees(arm_hi[k].item()):+.1f}] deg, "
                  f"center={math.degrees(arm_center[k].item()):+.1f} deg, "
                  f"amp={math.degrees(arm_amp_j[k].item()):.1f} deg, "
                  f"sign={arm_motion_sign[k].item():+.0f} ({tag})")

    # smooth takeoff: blend from the spawn position to the reference over T_RAMP
    p0 = robot.data.root_pos_w.clone()

    print(f"[INFO]: mass={m_tot:.3f} kg, hover/rotor={f_hover:.2f} N")
    print(f"[INFO]: J_base diag = {torch.diagonal(J).tolist()} kg*m^2")
    print(f"[INFO]: rotors={rotor_names}")
    print(f"[INFO]: rotor xy (CAD, visual/log allocation only):\n{ROTOR_XY.numpy().round(3)}")
    print(f"[INFO]: rotor offsets from CoM (sim CoM, ref only):\n{r_sim.cpu().numpy().round(3)}")
    print("[INFO]: control mode = direct body +z thrust and body torque on root body")

    sim_dt = sim.get_physics_dt()
    t, count = 0.0, 0
    ros_pub = setup_ros2_debug_publisher(robot, ee_body_ids, ee_names)
    if ros_pub is not None:
        zero_thrusts = torch.zeros(N, 4, device=device)
        zero_forces = torch.zeros(N, 1, 3, device=device)
        zero_torques = torch.zeros_like(zero_forces)
        ros_pub.publish(robot, zero_thrusts, zero_forces, zero_torques)
        ros_pub.print_local_topics()

    # running diagnostics: 팔 동작 "전"(baseline)과 "중"을 분리해서 비교
    diag_err2 = torch.zeros(3, device=device)   # baseline (팔 고정 구간)
    diag_n = 0
    arm_err2 = torch.zeros(3, device=device)    # 팔 동작 구간
    arm_n = 0
    max_dev = 0.0    # 팔 동작 중 최대 위치 이탈 [m]
    max_tilt = 0.0   # 팔 동작 중 최대 기울기 [deg]
    joint_limit_clamp_count = 0

    try:
        while simulation_app.is_running():
            # ---- ARM / TILT loop ---------------------------------------------------
            # 틸트는 0 고정(쿼드로터 모드), 팔은 arm_mode에 따라 고정 또는 사인 스윕.
            q_target = default_q.clone()
            arm_active = (args_cli.arm_mode == "sine") and (t > arm_t0)
            if arm_active:
                ta = t - arm_t0
                sa = min(ta / 2.0, 1.0)
                sa = sa * sa * (3.0 - 2.0 * sa)          # 팔 동작도 smoothstep으로 시작
                wa = 2.0 * math.pi / args_cli.arm_period
                phase = torch.tensor(wa * ta, device=device) + arm_phase
                if args_cli.arm_profile == "center_sine":
                    osc = arm_amp_j * torch.sin(phase)
                    # 기존 방식: 시작각 -> 중앙으로 블렌드 후 중앙 기준 사인.
                    # 리밋은 지키지만, 반주기 뒤에는 관절이 시작각 쪽으로 되돌아간다.
                    q_arm = arm_center0 + sa * (arm_center - arm_center0) + sa * osc
                else:
                    # 안전 방식: 시작각 기준 한쪽 방향으로만 움직인다.
                    # 0.5*(1-cos)는 항상 [0,1]이고, 팔 2~4번 관절은 +방향으로만 움직인다.
                    # dof1도 같은 one-sided 프로파일로 움직인다.
                    # dof2는 위에서 arm_j2_scale이 적용되어 더 크게 움직인다.
                    one_sided = 0.5 * (1.0 - torch.cos(phase))
                    q_arm = arm_center0 + arm_motion_sign * sa * arm_amp_j * one_sided
                q_target[:, arm_ids_t] = q_arm
            # 명령 안전망: arm 2~4는 cmd_lo/cmd_hi가 시작각 기준 상대 제약으로 덮여 있다.
            q_target = clamp_joint_tensor(q_target, cmd_lo, cmd_hi)
            robot.set_joint_position_target(q_target)

            # ---- DRONE loop: cascade PID -> per-rotor thrusts --------------------
            t_eff = max(t - T_RAMP, 0.0)
            p_d, v_d, a_d, yaw_d = reference(t_eff, args_cli.mode, args_cli.radius,
                                             args_cli.period, args_cli.ref_height, N, device)
            s = min(t / T_RAMP, 1.0)
            s = s * s * (3.0 - 2.0 * s)                 # smoothstep
            p_d = (1.0 - s) * p0 + s * p_d
            v_d, a_d = s * v_d, s * a_d
            T, tau = ctrl.compute(
                robot.data.root_pos_w, robot.data.root_quat_w,
                robot.data.root_link_lin_vel_w, robot.data.root_ang_vel_b,
                p_d, v_d, a_d, yaw_d, sim_dt)

            # 실제 dynamics에는 controller의 body-frame +z thrust와 body torque를 그대로 적용한다.
            forces = torch.zeros(N, 1, 3, device=device)
            torques = torch.zeros_like(forces)
            forces[:, 0, 2] = T
            torques[:, 0, :] = tau
            robot.set_external_force_and_torque(forces, torques, body_ids=wrench_body_ids)

            # rotor thrust는 프로펠러 시각화와 로그용 등가 분배값이다.
            u = torch.cat([T.unsqueeze(-1), tau], dim=-1)               # (N, 4)
            f = torch.einsum("ij,nj->ni", A_inv, u).clamp(0.0, f_max)   # (N, 4)

            # ---- visual prop spin (추력에 비례, 물리 영향 없음) -------------------
            spinner.update(f, spins, f_hover, sim_dt)

            scene.write_data_to_sim()
            sim.step()
            t += sim_dt
            count += 1
            scene.update(sim_dt)

            # 실제 joint state 안전망: USD hard limit이 적용되지 않았거나 관성 overshoot가 생긴 경우 즉시 복구한다.
            joint_limit_clamp_count += clamp_actual_joint_state_to_limits(
                robot, cmd_lo, cmd_hi, constrained_arm_ids_t)

            if ros_pub is not None:
                ros_pub.publish(robot, f, forces, torques)

            # accumulate diagnostics (after the takeoff ramp settles)
            if t > T_RAMP + 2.0:
                e = (p_d - robot.data.root_pos_w)[0]
                if arm_active and t > arm_t0 + 2.0:      # 팔 동작 정착 후
                    arm_err2 += e ** 2
                    arm_n += 1
                    dev = torch.linalg.norm(e).item()
                    max_dev = max(max_dev, dev)
                    R0 = matrix_from_quat(robot.data.root_quat_w)[0]
                    tilt = math.degrees(math.acos(max(-1.0, min(1.0, R0[2, 2].item()))))
                    max_tilt = max(max_tilt, tilt)
                elif not arm_active:                      # 팔 고정 baseline 구간
                    diag_err2 += e ** 2
                    diag_n += 1

            if count % 200 == 0:
                pe = (p_d - robot.data.root_pos_w)[0]
                fr = f[0].tolist()
                q = robot.data.root_quat_w[0]
                w_, x_, y_, z_ = q.tolist()
                roll = math.degrees(math.atan2(2 * (w_ * x_ + y_ * z_), 1 - 2 * (x_ * x_ + y_ * y_)))
                pitch = math.degrees(math.asin(max(-1, min(1, 2 * (w_ * y_ - z_ * x_)))))
                print(f"[t={t:6.2f}]{' [ARM]' if arm_active else '      '} "
                      f"pos err=({pe[0]:+.3f},{pe[1]:+.3f},{pe[2]:+.3f}) m | "
                      f"rp=({roll:+.1f},{pitch:+.1f}) deg | "
                      f"f[N]={fr[0]:.2f} {fr[1]:.2f} {fr[2]:.2f} {fr[3]:.2f} | "
                      f"joint_limit_clamps={joint_limit_clamp_count}")

            if args_cli.max_time > 0.0 and t >= args_cli.max_time:
                if diag_n > 0:
                    rmse = torch.sqrt(diag_err2 / diag_n)
                    print(f"[SUMMARY] baseline RMSE (x,y,z) = "
                          f"({rmse[0]:.4f},{rmse[1]:.4f},{rmse[2]:.4f}) m | n = {diag_n}")
                if arm_n > 0:
                    rmse_a = torch.sqrt(arm_err2 / arm_n)
                    print(f"[SUMMARY] arm-motion RMSE (x,y,z) = "
                          f"({rmse_a[0]:.4f},{rmse_a[1]:.4f},{rmse_a[2]:.4f}) m | "
                          f"max |dev| = {max_dev:.4f} m | max tilt = {max_tilt:.2f} deg | "
                          f"n = {arm_n}")
                z = robot.data.root_pos_w[0, 2].item()
                print(f"[SUMMARY] final z = {z:.3f} m")
                print(f"[SUMMARY] joint limit software clamp count = {joint_limit_clamp_count}")
                if diag_n == 0 and arm_n == 0:
                    print("[SUMMARY] sim ended before steady-state window")
                break
    finally:
        if ros_pub is not None:
            ros_pub.shutdown()


if __name__ == "__main__":
    main()
    simulation_app.close()

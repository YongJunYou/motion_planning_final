import argparse
import math
from dataclasses import dataclass

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="DAAM flight demo (quadrotor mode).")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--mode", type=str, default="hover", choices=["hover", "circle"])
parser.add_argument("--radius", type=float, default=0.8, help="Circle radius [m].")
parser.add_argument("--period", type=float, default=8.0, help="Circle period [s].")
parser.add_argument("--ref_height", type=float, default=1.8, help="Hover altitude [m].")
parser.add_argument("--k_drag", type=float, default=0.02,
                    help="Rotor drag torque / thrust ratio [m] (yaw model).")
parser.add_argument("--max_time", type=float, default=0.0,
                    help="If >0, stop the sim after this many seconds (for tuning).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

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

DAAM_CFG = ArticulationCfg(
    # 스폰방식 지정
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/yyj/motion_planning_final/dual_arm.usd",   
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
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)))
    robot: ArticulationCfg = DAAM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# ----------------------------------------------------------------------------
# math helpers
# ----------------------------------------------------------------------------
def vee(s):
    return torch.stack([s[..., 2, 1], s[..., 0, 2], s[..., 1, 0]], dim=-1)


# ----------------------------------------------------------------------------
# Visual propeller spin (physics에 영향 없음)
# ----------------------------------------------------------------------------
PROP_DUCTS = ["lb", "lf", "rb", "rf"]            # thrusts의 열 인덱스 순서
PROP_LEAVES = ["_prop_up", "_prop_dn"]           # 덕트당 동축 프롭 2개 (USD prim 이름 suffix)
PROP_SIGN = [+1, -1]                             # 동축 쌍(up/dn)은 서로 반대 회전
VIS_RATE = 1500.0                                # 호버 추력 기준 회전속도 [deg/s], 취향껏


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
# Drone cascade PID (position -> attitude) producing [thrust_total, tau]
# ----------------------------------------------------------------------------
class DronePID:
    def __init__(self, num_envs, mass, gravity, inertia, device):
        self.N, self.m, self.g = num_envs, float(mass), float(gravity)
        self.J = inertia
        self.device = device

        # 위치제어 PID Gains 
        self.Kp_pos = torch.tensor([5.0, 5.0, 12.0], device=device)
        self.Ki_pos = torch.tensor([0.8, 0.8, 3.0], device=device)
        self.Kd_pos = torch.tensor([4.0, 4.0, 7.0], device=device)

        # 자세제어 PID Gains 
        self.Kp_att = torch.tensor([90.0, 90.0, 25.0], device=device)
        self.Kd_att = torch.tensor([14.0, 14.0, 6.0], device=device)

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


# ----------------------------------------------------------------------------
# control loop state
# ----------------------------------------------------------------------------
@dataclass
class ControlLoopState:
    t: float
    count: int
    diag_err2: torch.Tensor
    diag_n: int
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
    robot.find_bodies("base_link_01")
    rotor_ids, rotor_names = robot.find_bodies(
        ["lb_link2_01", "lf_link2_01", "rb_link2_01", "rf_link2_01"])
    robot.find_joints(["dof_[lr][1-4]"])
    robot.find_joints(["dof_[lr][bf][12]"])
    return rotor_ids, rotor_names


def compute_whole_body_inertia(robot, device):
    """링크 CoM 기준으로 전체 질량/관성행렬을 계산한다."""
    masses = robot.root_physx_view.get_masses()[0]
    inertias = robot.root_physx_view.get_inertias()[0]
    m_tot = masses.sum().item()
    g = 9.81

    body_com = robot.data.body_com_pos_w[0].cpu()
    masses_cpu = masses.cpu()
    com0 = (body_com * masses_cpu.unsqueeze(-1)).sum(0) / m_tot

    J = torch.zeros(3, 3)
    eye = torch.eye(3)
    for b in range(len(masses)):
        Ib = inertias[b].reshape(3, 3).cpu()
        d = body_com[b] - com0
        J += Ib + masses_cpu[b] * ((d @ d) * eye - torch.outer(d, d))

    return masses, m_tot, g, J.to(device).float()


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


def build_quadrotor_allocator(m_tot, g, device):
    """[thrust_total, tau_x, tau_y, tau_z] -> rotor thrust 4개로 바꾸는 allocation matrix를 만든다."""
    x, y = ROTOR_XY[:, 0].clone(), ROTOR_XY[:, 1].clone()
    spins = torch.tensor([+1.0, -1.0, -1.0, +1.0])  # lb, lf, rb, rf
    A = torch.stack([torch.ones(4), y, -x, spins * args_cli.k_drag], dim=0)
    A_inv = torch.linalg.inv(A).to(device)
    thrust_hover = m_tot * g / 4.0
    thrust_max = 3.0 * thrust_hover
    return spins, A_inv, thrust_hover, thrust_max


def get_rotor_offsets_for_log(robot, masses, rotor_ids, m_tot, device):
    """할당에는 쓰지 않고 로그 비교용으로만 시뮬 CoM 기준 rotor offset을 계산한다."""
    com = (robot.data.body_com_pos_w[0] * masses.unsqueeze(-1).to(device)).sum(0) / m_tot
    return robot.data.body_com_pos_w[0, rotor_ids] - com


def print_startup_log(m_tot, thrust_hover, J, rotor_names, r_sim):
    print(f"[INFO]: mass={m_tot:.3f} kg, hover/rotor={thrust_hover:.2f} N")
    print(f"[INFO]: J diag = {torch.diagonal(J).tolist()}")
    print(f"[INFO]: rotors={rotor_names}")
    print(f"[INFO]: rotor xy (CAD, used):\n{ROTOR_XY.numpy().round(3)}")
    print(f"[INFO]: rotor offsets from CoM (sim CoM, ref only):\n{r_sim.cpu().numpy().round(3)}")


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
    return p_d, v_d, a_d, yaw_d


def compute_rotor_thrusts(ctrl, robot, p_d, v_d, a_d, yaw_d, A_inv, thrust_max, sim_dt):
    """cascade PID 출력 [thrust_total, tau]를 rotor별 thrusts로 변환한다."""
    thrust_total, tau = ctrl.compute(
        robot.data.root_pos_w, robot.data.root_quat_w,
        robot.data.root_lin_vel_w, robot.data.root_ang_vel_b,
        p_d, v_d, a_d, yaw_d, sim_dt)

    u = torch.cat([thrust_total.unsqueeze(-1), tau], dim=-1)    # (N, 4)
    thrusts = torch.einsum("ij,nj->ni", A_inv, u).clamp(0.0, thrust_max)
    return thrusts                                                   # (N, 4) / num_envs=1이면 (1, 4)


def apply_rotor_forces(robot, thrusts, spins, rotor_ids, k_drag, device):
    """rotor별 thrusts와 drag yaw torque를 articulation에 적용한다."""
    num_envs = robot.num_instances
    forces = torch.zeros(num_envs, 4, 3, device=device)
    torques = torch.zeros_like(forces)
    forces[..., 2] = thrusts
    torques[..., 2] = spins.to(device) * k_drag * thrusts
    robot.set_external_force_and_torque(forces, torques, body_ids=rotor_ids)


def step_simulation(sim, scene, sim_dt):
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim_dt)


def accumulate_steady_state_error(t, ramp_time, p_d, robot, diag_err2, diag_n):
    if t > ramp_time + 2.0:
        diag_err2 += ((p_d - robot.data.root_pos_w)[0]) ** 2
        diag_n += 1
    return diag_err2, diag_n


def quat_to_roll_pitch_deg(quat):
    w_, x_, y_, z_ = quat.tolist()
    roll = math.degrees(math.atan2(2 * (w_ * x_ + y_ * z_), 1 - 2 * (x_ * x_ + y_ * y_)))
    pitch = math.degrees(math.asin(max(-1, min(1, 2 * (w_ * y_ - z_ * x_)))))
    return roll, pitch


def print_periodic_status(t, p_d, robot, thrusts):
    pe = (p_d - robot.data.root_pos_w)[0]
    thrusts_row = thrusts[0].tolist()
    roll, pitch = quat_to_roll_pitch_deg(robot.data.root_quat_w[0])
    print(f"[t={t:6.2f}] pos err=({pe[0]:+.3f},{pe[1]:+.3f},{pe[2]:+.3f}) m | "
          f"rp=({roll:+.1f},{pitch:+.1f}) deg | "
          f"thrusts[N]={thrusts_row[0]:.2f} {thrusts_row[1]:.2f} "
          f"{thrusts_row[2]:.2f} {thrusts_row[3]:.2f}")


def should_stop_and_print_summary(t, diag_err2, diag_n, robot):
    if args_cli.max_time <= 0.0 or t < args_cli.max_time:
        return False

    if diag_n > 0:
        rmse = torch.sqrt(diag_err2 / diag_n)
        z = robot.data.root_pos_w[0, 2].item()
        print(f"[SUMMARY] steady-state RMSE (x,y,z) = "
              f"({rmse[0]:.4f},{rmse[1]:.4f},{rmse[2]:.4f}) m | "
              f"final z = {z:.3f} m | n = {diag_n}")
    else:
        print("[SUMMARY] sim ended before steady-state window")
    return True


def run_control_step(robot, ctrl, rotor_ids, spinner, state):
    """while 루프의 한 스텝을 기능 단위로 묶은 함수."""
    robot.set_joint_position_target(state.default_q)

    p_d, v_d, a_d, yaw_d = make_smooth_reference(
        state.t, state.p0, state.N, state.device, state.ramp_time)

    thrusts = compute_rotor_thrusts(
        ctrl, robot, p_d, v_d, a_d, yaw_d,
        state.A_inv, state.thrust_max, state.sim_dt)

    apply_rotor_forces(
        robot, thrusts, state.spins, rotor_ids, args_cli.k_drag, state.device)

    spinner.update(thrusts, state.spins, state.thrust_hover, state.sim_dt)
    return p_d, thrusts


def main():
    # 시작시 설정
    sim, scene, spinner, robot = setup_sim_scene_and_robot()
    device = robot.device #계산 디바이스 (CPU or GPU)
    robot_num = robot.num_instances # 로봇 대수, 1

    rotor_ids, rotor_names = find_robot_handles(robot)
    masses, m_tot, g, J = compute_whole_body_inertia(robot, device)
    default_q = reset_robot_to_default(robot, scene)

    spins, A_inv, thrust_hover, thrust_max = build_quadrotor_allocator(m_tot, g, device)
    r_sim = get_rotor_offsets_for_log(robot, masses, rotor_ids, m_tot, device)

    ctrl = DronePID(robot_num, m_tot, g, J, device)

    ramp_time = 4.0
    p0 = robot.data.root_pos_w.clone()
    sim_dt = sim.get_physics_dt()

    print_startup_log(m_tot, thrust_hover, J, rotor_names, r_sim)

    state = ControlLoopState(
        t=0.0,
        count=0,
        diag_err2=torch.zeros(3, device=device),
        diag_n=0,
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

    # 루프
    while simulation_app.is_running():
        p_d, thrusts = run_control_step(robot, ctrl, rotor_ids, spinner, state)

        step_simulation(sim, scene, sim_dt)
        state.t += sim_dt
        state.count += 1

        state.diag_err2, state.diag_n = accumulate_steady_state_error(
            state.t, ramp_time, p_d, robot, state.diag_err2, state.diag_n)

        if state.count % 200 == 0:
            print_periodic_status(state.t, p_d, robot, thrusts)

        if should_stop_and_print_summary(state.t, state.diag_err2, state.diag_n, robot):
            break


if __name__ == "__main__":
    main()
    simulation_app.close()
import argparse
import math

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
        usd_path="/home/yyj/motion_planning_final/dual_arm.usd",   # 수정본(질량/충돌/드라이브 보강)으로 교체할 것
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


# ----------------------------------------------------------------------------
# Visual propeller spin (physics에 영향 없음)
# ----------------------------------------------------------------------------
# 프롭 링크 prim 자체의 transform은 PhysX가 매 스텝 덮어쓰므로,
# 링크 밑 비주얼 스코프 안의 "프롭 prim"에 xformOp를 추가해 돌린다.
# 구조: Robot/main_v10/main_v10/{duct}_link2_01/{duct}_link2_01/{duct}_prop_{up,dn}
#       (물리 링크)        (비주얼 스코프)     (프롭 Xform, 이름으로 식별)
#
# - USD에서 프롭을 {duct}_prop_up / {duct}_prop_dn 으로 명시적으로 이름 붙여 둠
#   (덕트 4개 x 동축 2개 = 총 8개). 이름이 유일하므로 이름 매칭이 곧 식별.
# - 추가로 부모 경로에 해당 덕트의 물리 링크가 있는지 확인해서, 향후 다른
#   서브트리에 동명 prim이 생겨도 같이 도는 것을 방지한다.
# - 프롭 prim은 instanceable Xform이고 실제 Mesh는 프로토타입 참조(인스턴스
#   프록시)로 그 아래에 있다. 따라서 정점은 TraverseInstanceProxies로 하위
#   Mesh를 찾아 읽고, 프롭 prim 로컬 프레임으로 변환해 사용한다.
# - 회전축/피벗은 메시 정점에서 직접 계산한다: 프롭은 얇은 디스크라
#   정점 공분산의 최소 고유벡터(SVD 최소 특이방향) = 디스크 법선 = 회전축,
#   centroid = 피벗. 메시 로컬 프레임이 CAD에서 틀어져 있어도 자동으로 맞아
#   세차운동처럼 삐딱하게 도는 문제가 없다.
PROP_DUCTS = ["lb", "lf", "rb", "rf"]            # f의 열 인덱스 순서
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

        # prim 이름 -> (duct 인덱스, sign) 매핑. 예: "lb_prop_dn" -> (0, -1)
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
                # 부모 경로에 해당 덕트의 물리 링크가 있어야 함 -> 다른 서브트리의
                # 동명 prim(돌면 안 되는 지오메트리)을 매칭에서 제외.
                if f"/{PROP_DUCTS[duct]}_link2_01/" not in str(prim.GetPath()):
                    print(f"[SKIP] name matched but outside duct subtree: {prim.GetPath()}")
                    continue
                # 이 prim은 비주얼 겸 충돌체로 변환된 prim이라, transform을 돌리면
                # PhysX shape 재생성 -> tensor view invalidation이 발생한다.
                # 프롭 충돌체는 불필요하므로(덕트 내부, 접촉은 팔/박스) 떼어내서
                # 순수 비주얼 prim으로 만든다. sim.reset() 이전이라 안전.
                # (applied schema는 instance prim 자신의 메타데이터라 인스턴스여도 편집 가능)
                for schema in list(prim.GetAppliedSchemas()):
                    if "Collision" in schema or "Physx" in schema:
                        prim.RemoveAppliedSchema(schema)
                # 회전축/피벗 계산: 하위 메시 정점 SVD (실패 시 bbox 중심 + 로컬 z로 폴백)
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
        """프롭 prim 하위 메시 정점에서 (피벗, 회전축)을 계산. 디스크 법선 = 최소 분산 방향.

        프롭 prim이 Mesh 자체일 수도 있고, instanceable Xform 아래 인스턴스 프록시
        Mesh일 수도 있다. TraverseInstanceProxies로 둘 다 커버하고, 정점은 각 메시의
        로컬 프레임 -> 프롭 prim 프레임으로 변환해 합산한다 (xformOp는 prim에 걸리므로).
        """
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
            # SVD 축은 부호가 임의 -> "월드" 기준으로 정규화한다.
            # 로컬 지배성분 기준으로 맞추면, 비주얼 프레임이 미러/180도 뒤집힌
            # 덕트(이 자산에선 rf)만 월드에서 반대로 돌게 된다.
            # 스폰 시점은 수평 + 틸트 0이므로, 월드 z 성분이 +가 되도록 맞추면
            # 네 덕트 모두 회전 방향 부호가 일관된다.
            Rw = np.array([[L2W[i][j] for j in range(3)] for i in range(3)])
            if (axis @ Rw)[2] < 0:
                axis = -axis
            return Gf.Vec3d(*ctr.tolist()), Gf.Vec3d(*axis.tolist())
        # 폴백: 하위에서 메시를 못 찾으면 bbox 중심 + 로컬 z (이 경우 경고 출력)
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
# Drone cascade PID (position -> attitude) producing [T, tau]
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


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / 200.0)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.5, 2.5, 2.0], target=[0.0, 0.0, 1.0])

    scene = InteractiveScene(FlightSceneCfg(num_envs=args_cli.num_envs, env_spacing=4.0))

    # ---- visual prop spinner --------------------------------------------------
    # AddTransformOp / collision 제거는 USD 구조 변경이라 sim.reset() "이전"에 해야 함.
    # (sim 시작 후 하면 PhysX tensor view가 invalidate됨)
    stage = omni.usd.get_context().get_stage()
    spinner = PropSpinner(stage, args_cli.num_envs)

    sim.reset()

    robot: Articulation = scene["robot"]
    device = robot.device
    N = robot.num_instances

    # ---- ids ---------------------------------------------------------------
    base_ids, _ = robot.find_bodies("base_link_01")
    # inner rotor assemblies (motor+prop) = tilt link2 bodies
    rotor_ids, rotor_names = robot.find_bodies(
        ["lb_link2_01", "lf_link2_01", "rb_link2_01", "rf_link2_01"])
    arm_ids, _ = robot.find_joints(["dof_[lr][1-4]"])
    tilt_ids, _ = robot.find_joints(["dof_[lr][bf][12]"])

    # ---- model parameters ----------------------------------------------------
    masses = robot.root_physx_view.get_masses()[0]
    m_tot = masses.sum().item()
    g = 9.81
    inertias = robot.root_physx_view.get_inertias()[0]

    # whole-body inertia about the vehicle CoM (parallel axis).
    # NOTE: 링크 "프레임 원점"(body_pos_w)이 아니라 링크 "CoM"(body_com_pos_w)을
    # 기준으로 합산해야 함 — 검사 결과 원점 기반 J는 최대 57% 틀렸음 (Ixx 2.4배 차이).
    body_com = robot.data.body_com_pos_w[0].cpu()
    masses_cpu = masses.cpu()
    com0 = (body_com * masses_cpu.unsqueeze(-1)).sum(0) / m_tot
    J = torch.zeros(3, 3)
    eye = torch.eye(3)
    for b in range(len(masses)):
        Ib = inertias[b].reshape(3, 3).cpu()
        d = body_com[b] - com0
        J += Ib + masses_cpu[b] * ((d @ d) * eye - torch.outer(d, d))
    J = J.to(device).float()

    # ---- initial state -------------------------------------------------------
    default_q = robot.data.default_joint_pos.clone()
    lims = robot.data.soft_joint_pos_limits[0]
    default_q[0] = default_q[0].clamp(lims[:, 0] + 0.02, lims[:, 1] - 0.02)
    root = robot.data.default_root_state.clone()
    root[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_root_velocity_to_sim(root[:, 7:])
    robot.write_joint_state_to_sim(default_q, torch.zeros_like(default_q))
    scene.reset()

    # ---- allocation from CAD geometry (hardcoded, level pose) ----------------
    # 모멘트암 (x_i, y_i)는 시뮬에서 읽지 않고 CAD 실측값(ROTOR_XY)을 사용한다.
    # -> 링크 원점 vs CoM, 프레임 정렬 등 시뮬 의존 불확실성을 할당 모델에서 제거.
    x, y = ROTOR_XY[:, 0].clone(), ROTOR_XY[:, 1].clone()
    spins = torch.tensor([+1.0, -1.0, -1.0, +1.0])            # lb, lf, rb, rf (flip if yaw runs away)
    A = torch.stack([torch.ones(4), y, -x, spins * args_cli.k_drag], dim=0)
    A_inv = torch.linalg.inv(A).to(device)
    f_hover = m_tot * g / 4.0
    f_max = 3.0 * f_hover

    # 참고용 비교 출력: 시뮬에서 읽은 로터 링크 CoM 기준 오프셋 (할당에는 사용 안 함).
    # NOTE: 이 자산은 링크 프레임 원점이 전부 루트에 몰려 있어 원점은 비교 의미가 없음.
    #       CoM(body_com_pos_w)이 실제 질량분포 위치라 이것과 CAD 값을 대조한다.
    com = (robot.data.body_com_pos_w[0] * masses.unsqueeze(-1).to(device)).sum(0) / m_tot
    r_sim = (robot.data.body_com_pos_w[0, rotor_ids] - com)  # (4, 3), world ~ body at level

    ctrl = DronePID(N, m_tot, g, J, device)

    # smooth takeoff: blend from the spawn position to the reference over T_RAMP
    T_RAMP = 4.0
    p0 = robot.data.root_pos_w.clone()

    print(f"[INFO]: mass={m_tot:.3f} kg, hover/rotor={f_hover:.2f} N")
    print(f"[INFO]: J diag = {torch.diagonal(J).tolist()}")
    print(f"[INFO]: rotors={rotor_names}")
    print(f"[INFO]: rotor xy (CAD, used):\n{ROTOR_XY.numpy().round(3)}")
    print(f"[INFO]: rotor offsets from CoM (sim CoM, ref only):\n{r_sim.cpu().numpy().round(3)}")

    sim_dt = sim.get_physics_dt()
    t, count = 0.0, 0

    # running diagnostics for the tuning summary (steady-state window)
    diag_err2 = torch.zeros(3, device=device)
    diag_n = 0

    while simulation_app.is_running():
        # ---- ARM / TILT loop: hold targets (separate channel) ---------------- 쿼드로터 처럼 동작시키려고 고정한듯.
        robot.set_joint_position_target(default_q) 

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
            robot.data.root_lin_vel_w, robot.data.root_ang_vel_b,
            p_d, v_d, a_d, yaw_d, sim_dt)

        u = torch.cat([T.unsqueeze(-1), tau], dim=-1)               # (N, 4)
        f = torch.einsum("ij,nj->ni", A_inv, u).clamp(0.0, f_max)   # (N, 4)

        forces = torch.zeros(N, 4, 3, device=device)
        torques = torch.zeros_like(forces)
        forces[..., 2] = f                                          # thrust along rotor local +z
        torques[..., 2] = spins.to(device) * args_cli.k_drag * f    # drag yaw model
        robot.set_external_force_and_torque(forces, torques, body_ids=rotor_ids)

        # ---- visual prop spin (추력에 비례, 물리 영향 없음) -------------------
        spinner.update(f, spins, f_hover, sim_dt)

        scene.write_data_to_sim()
        sim.step()
        t += sim_dt
        count += 1
        scene.update(sim_dt)

        # accumulate steady-state error (after the takeoff ramp settles)
        if t > T_RAMP + 2.0:
            diag_err2 += ((p_d - robot.data.root_pos_w)[0]) ** 2
            diag_n += 1

        if count % 200 == 0:
            pe = (p_d - robot.data.root_pos_w)[0]
            fr = f[0].tolist()
            q = robot.data.root_quat_w[0]
            w_, x_, y_, z_ = q.tolist()
            roll = math.degrees(math.atan2(2 * (w_ * x_ + y_ * z_), 1 - 2 * (x_ * x_ + y_ * y_)))
            pitch = math.degrees(math.asin(max(-1, min(1, 2 * (w_ * y_ - z_ * x_)))))
            print(f"[t={t:6.2f}] pos err=({pe[0]:+.3f},{pe[1]:+.3f},{pe[2]:+.3f}) m | "
                  f"rp=({roll:+.1f},{pitch:+.1f}) deg | "
                  f"f[N]={fr[0]:.2f} {fr[1]:.2f} {fr[2]:.2f} {fr[3]:.2f}")

        if args_cli.max_time > 0.0 and t >= args_cli.max_time:
            if diag_n > 0:
                rmse = torch.sqrt(diag_err2 / diag_n)
                z = robot.data.root_pos_w[0, 2].item()
                print(f"[SUMMARY] steady-state RMSE (x,y,z) = "
                      f"({rmse[0]:.4f},{rmse[1]:.4f},{rmse[2]:.4f}) m | "
                      f"final z = {z:.3f} m | n = {diag_n}")
            else:
                print("[SUMMARY] sim ended before steady-state window")
            break


if __name__ == "__main__":
    main()
    simulation_app.close()
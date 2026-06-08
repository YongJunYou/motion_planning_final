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

import torch

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
_D = math.radians
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
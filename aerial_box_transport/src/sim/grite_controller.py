"""gRITE base controller: geometric SE(3) tracking + robust integral + whole-body
gravity-moment compensation. Produces a body-frame 6-DoF wrench [f; tau] that is
applied directly to the base (allocation skipped, D5 abstraction).

Implemented from scratch (no source available). The control law mirrors the gRITE
structure: a geometric controller on SE(3) (position + SO(3) attitude error via
the vee map), a robust integral term (a practical stand-in for the exact RISE
term), and compensation of the gravity moment produced by the arms, computed from
the whole-body center of mass (so no Pinocchio is needed at runtime; this runs in
the am_isaac env). Numpy only.
"""
import numpy as np


def vee(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


class GRITEController:
    def __init__(self, m_total, J_base, g=9.81, dt=1.0 / 200.0):
        self.m = float(m_total)
        self.J = np.asarray(J_base, float)
        self.g = float(g)
        self.dt = float(dt)
        # position loop (original, verified-stable values; the box is held by the gripper directly in
        # the sim so tight base tracking is not needed -- stability matters more than a few cm).
        self.Kp = np.array([7.5, 7.5, 18.0])
        self.Kd = np.array([5.2, 5.2, 9.0])
        self.Ki = np.array([1.2, 1.2, 4.5])
        # attitude loop (original, verified-stable; pushing Kw to ~55 amplified sim angular-rate noise
        # and flipped the drone).
        self.KR = np.array([300.0, 300.0, 90.0])
        self.Kw = np.array([35.0, 35.0, 18.0])
        self.KRi = np.array([20.0, 20.0, 8.0])
        self.int_ep = np.zeros(3)
        self.int_eR = np.zeros(3)
        self.e3 = np.array([0.0, 0.0, 1.0])

    def reset(self):
        self.int_ep[:] = 0.0
        self.int_eR[:] = 0.0

    def compute(self, p, R, v, omega_b, com_offset_b, p_d, v_d, a_d, R_d, omega_d):
        """All world-frame except omega_b (body) and the returned wrench (body).

        com_offset_b: whole-body CoM offset from the base origin, in the base frame.
        Returns (f_body[3], tau_body[3]).
        """
        # translational (geometric PID with acceleration feedforward)
        e_p = p - p_d
        e_v = v - v_d
        self.int_ep = np.clip(self.int_ep + e_p * self.dt, -1.0, 1.0)
        a_cmd = a_d - self.Kp * e_p - self.Kd * e_v - self.Ki * self.int_ep
        F_des_w = self.m * a_cmd + self.m * self.g * self.e3      # world force (hover + track)
        f_body = R.T @ F_des_w

        # rotational (geometric SO(3) + robust integral)
        e_R = vee(0.5 * (R_d.T @ R - R.T @ R_d))
        e_w = omega_b - R.T @ R_d @ omega_d
        self.int_eR = np.clip(self.int_eR + e_R * self.dt, -0.3, 0.3)
        a_ang = -(self.KR * e_R + self.Kw * e_w + self.KRi * self.int_eR)
        tau_gyro = np.cross(omega_b, self.J @ omega_b)
        # gravity moment about the base from the (arm-shifted) whole-body CoM
        tau_grav = self.m * self.g * np.cross(com_offset_b, R.T @ self.e3)
        tau_body = self.J @ a_ang + tau_gyro + tau_grav
        return f_body, tau_body

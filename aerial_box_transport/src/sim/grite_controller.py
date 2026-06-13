"""gRITE base controller: geometric SE(3) tracking + RISE robust term + whole-body
gravity-moment / inertia compensation. Produces a body-frame 6-DoF wrench [f; tau]
applied directly to the base (allocation skipped, D5 abstraction).

Ported from the flight implementation
(tiltrotor_coax_grite_controller/src/tiltrotor_coax_grite_controller.cpp) and the
gRITE paper. Structure (NOT a plain geometric PID):

  errors        e_p, e_v (world);  Psi = R^T R_d;  e_R = vee(0.5(Psi^T - Psi));
                e_w = omega - Psi omega_d
  RISE filtered e_t1 = e_v + Lam_t1 e_p ;   e_r1 = e_w + Lam_r1 e_R
  nominal       f_n   = m R^T(a_d - K_tp e_p - K_td e_v + g e3)            (ENU gravity +g e3)
                tau_n = omega x (J omega) - J(omega x Psi omega_d - Psi omega_d_dot)
                        - J K_rp e_R - J K_rd e_w + tau_grav_comp
  RISE robust   f_ri  += dt[(K_ti+rho_t) e_t1 + Lam_t2 tanh(Theta_t e_t1)]
                f_r    = -R^T[(K_ti+rho_t) e_t1 + f_ri]
                tau_ri += dt[(K_ri+rho_r) e_r1 + Lam_r2 tanh(Theta_r e_r1)]
                tau_r  = -(K_ri+rho_r) e_r1 - tau_ri
  wrench        f = f_n + f_r ,  tau = tau_n + tau_r

The RISE term (a continuous integral of the error plus a smooth `sign`, tanh) is the
robust core: unlike a clamped PID integral it gives ASYMPTOTIC rejection of bounded
unmodeled disturbances -- here the dominant one is the carried box riding ~0.5 m off
the base (an uncompensated gravity/inertia moment). Numpy only (runs in am_isaac).
"""
import numpy as np


def vee(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def hat(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


class GRITEController:
    def __init__(self, m_total, J_base, g=9.81, dt=1.0 / 200.0):
        self.m = float(m_total)
        self.J = np.asarray(J_base, float)        # 3x3 (diagonal body inertia)
        self.g = float(g)
        self.dt = float(dt)

        # --- translational gRITE gains (per-axis world [x,y,z]) ---
        # Tuned up from the proven desk->rack baseline to cut the window-threading peak + RMSE: higher
        # P/D bandwidth (lateral x,y especially, where the carried box swings off), stronger RISE.
        self.K_tp = np.array([22.0, 22.0, 36.0])  # nominal P
        self.K_td = np.array([10.0, 10.0, 14.0])  # nominal D (kept >= overdamped vs K_tp)
        self.K_ti = np.array([8.0, 8.0, 14.0])    # RISE proportional-on-filtered-error gain
        self.rho_t = np.array([2.5, 2.5, 2.0])    # RISE robust gain
        self.Lam_t1 = np.array([1.5, 1.5, 1.5])   # filtered-error blend (e_v + Lam1 e_p)
        self.Lam_t2 = np.array([4.0, 4.0, 4.0])   # RISE sign-term amplitude
        self.Theta_t = np.array([4.0, 4.0, 4.0])  # RISE tanh sharpness

        # --- rotational gRITE gains (per-axis body [roll,pitch,yaw]); K_rp/K_rd are J-scaled ---
        self.K_rp = np.array([520.0, 520.0, 150.0])
        self.K_rd = np.array([52.0, 52.0, 26.0])
        self.K_ri = np.array([18.0, 18.0, 8.0])   # RISE (direct torque, NOT J-scaled)
        self.rho_r = np.array([3.5, 3.5, 1.5])
        self.Lam_r1 = np.array([2.0, 2.0, 2.0])
        self.Lam_r2 = np.array([0.7, 0.7, 0.3])
        self.Theta_r = np.array([5.0, 5.0, 5.0])

        self.f_ri = np.zeros(3)                   # RISE integral states
        self.tau_ri = np.zeros(3)
        self.e3 = np.array([0.0, 0.0, 1.0])

    def reset(self):
        self.f_ri[:] = 0.0
        self.tau_ri[:] = 0.0

    def compute(self, p, R, v, omega_b, com_offset_b, p_d, v_d, a_d, R_d, omega_d, omega_d_dot=None):
        """All world-frame except omega_b (body) and the returned wrench (body).
        com_offset_b: whole-body CoM offset from the base origin, in the base frame.
        omega_d_dot: feedforward angular acceleration (body of R_d); zeros if None.
        Returns (f_body[3], tau_body[3])."""
        if omega_d_dot is None:
            omega_d_dot = np.zeros(3)

        # errors
        e_p = p - p_d
        e_v = v - v_d
        Psi = R.T @ R_d
        e_R = vee(0.5 * (Psi.T - Psi))
        e_w = omega_b - Psi @ omega_d

        # RISE filtered (sliding) errors
        e_t1 = e_v + self.Lam_t1 * e_p
        e_r1 = e_w + self.Lam_r1 * e_R

        # --- translational: nominal (PD + accel ff + gravity, ENU) + RISE robust ---
        f_n = self.m * (R.T @ (a_d - self.K_tp * e_p - self.K_td * e_v + self.g * self.e3))
        self.f_ri += self.dt * ((self.K_ti + self.rho_t) * e_t1
                                + self.Lam_t2 * np.tanh(self.Theta_t * e_t1))
        f_r = -R.T @ ((self.K_ti + self.rho_t) * e_t1 + self.f_ri)
        f_body = f_n + f_r

        # --- rotational: nominal (gyroscopic + attitude/rate feedforward + PD + gravity moment) ---
        tau_ff = hat(omega_b) @ (self.J @ omega_b) \
            - self.J @ (hat(omega_b) @ (Psi @ omega_d) - Psi @ omega_d_dot)
        tau_pd = -self.J @ (self.K_rp * e_R) - self.J @ (self.K_rd * e_w)
        # gravity moment about the base from the (arm-shifted) whole-body CoM (ENU: +m g r x R^T e3)
        tau_grav_comp = self.m * self.g * np.cross(com_offset_b, R.T @ self.e3)
        tau_n = tau_ff + tau_pd + tau_grav_comp

        # --- rotational RISE robust ---
        self.tau_ri += self.dt * ((self.K_ri + self.rho_r) * e_r1
                                  + self.Lam_r2 * np.tanh(self.Theta_r * e_r1))
        tau_r = -(self.K_ri + self.rho_r) * e_r1 - self.tau_ri
        tau_body = tau_n + tau_r
        return f_body, tau_body

"""M3: reduced free box body and friction cone (vertical-channel MVP).

The box is squeezed horizontally by two end-effectors on opposite faces and is
held against gravity by vertical friction only, with no form closure (spec D6).
For the MVP we keep the vertical linear channel, which is enough to reproduce
slip (spec section 9, Phase 0). Box rotation is added later.

Vertical free body:        m_o * a_box = f_fric_total - m_o * g
Friction cone per contact: |F_t,i| <= mu * lambda_n,i

Sign convention: friction f_fric_total is the upward force holding or lifting
the box. To track an upward acceleration a the box needs f = m_o * (g + a).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class BoxParams:
    m_o: float          # box mass [kg]
    mu: float           # friction coefficient
    g: float = 9.81     # gravity [m/s^2]


def required_friction_total(m_o, a_z, g=9.81):
    """Total vertical friction needed to carry the box at upward accel a_z."""
    return m_o * (g + a_z)


def required_normal_force(m_o, a_z, mu, g=9.81, n_contacts=2):
    """Per-contact squeeze force so friction can carry the box at accel a_z.

    F_n >= m_o * (g + a_z) / (n_contacts * mu). This is the core slip law.
    """
    return m_o * (g + a_z) / (n_contacts * mu)


def max_friction_total(F_n, mu, n_contacts=2):
    """Total friction available from n contacts each squeezing with F_n."""
    return n_contacts * mu * F_n


def will_slip(F_n, m_o, a_z, mu, g=9.81, n_contacts=2):
    """True if available friction cannot hold the box at upward accel a_z."""
    return max_friction_total(F_n, mu, n_contacts) < required_friction_total(m_o, a_z, g)


def friction_cone_residual(F_t, lambda_n, mu):
    """Cone residual mu*lambda_n - |F_t|. Feasible iff >= 0."""
    return mu * lambda_n - np.abs(F_t)


def box_vertical_accel(F_n, m_o, a_ee, mu, g=9.81, n_contacts=2):
    """One-step vertical acceleration of the box given squeeze F_n and EE accel.

    If friction suffices to track a_ee the box accelerates at a_ee; otherwise
    friction saturates at the kinetic limit (mu_k = mu here) and the box cannot
    keep up. Returns (a_box, f_fric_total, slipping).
    """
    f_needed = required_friction_total(m_o, a_ee, g)
    f_max = max_friction_total(F_n, mu, n_contacts)
    if f_needed <= f_max + 1e-9:
        return a_ee, f_needed, False
    return f_max / m_o - g, f_max, True


def simulate_vertical_transport(box, F_n, a_z, t_accel, dt, n_contacts=2):
    """Constant upward acceleration burst, measuring how far the box slips.

    The gripper accelerates upward at constant a_z from rest for t_accel seconds
    while squeezing with a constant per-contact force F_n. While the friction
    needed to track the gripper does not exceed the available friction the box
    sticks; otherwise it slips and friction saturates.

    Returns a dict of time series plus 'max_slip' (max of z_ee - z_box) and the
    cached 'f_max' / 'f_needed'.
    """
    g, mu, m_o = box.g, box.mu, box.m_o
    f_max = max_friction_total(F_n, mu, n_contacts)
    f_needed = required_friction_total(m_o, a_z, g)

    n = int(round(t_accel / dt))
    t = np.zeros(n + 1)
    z_ee = np.zeros(n + 1)
    v_ee = np.zeros(n + 1)
    z_box = np.zeros(n + 1)
    v_box = np.zeros(n + 1)
    f_fric = np.zeros(n + 1)
    slipping = np.zeros(n + 1, dtype=bool)

    stuck = f_needed <= f_max + 1e-9
    for i in range(n):
        a_ee = a_z
        v_ee[i + 1] = v_ee[i] + a_ee * dt
        z_ee[i + 1] = z_ee[i] + v_ee[i] * dt + 0.5 * a_ee * dt * dt

        if stuck:
            a_box, f, slip_now = a_ee, f_needed, False
        else:
            f = f_max
            a_box = f / m_o - g
            slip_now = True
            # re-grab if the box has caught up to the gripper and friction suffices
            if v_box[i] >= v_ee[i] and f_needed <= f_max + 1e-9:
                stuck = True
                a_box, f, slip_now = a_ee, f_needed, False

        v_box[i + 1] = v_box[i] + a_box * dt
        z_box[i + 1] = z_box[i] + v_box[i] * dt + 0.5 * a_box * dt * dt
        f_fric[i + 1] = f
        slipping[i + 1] = slip_now
        t[i + 1] = t[i] + dt

    slip = z_ee - z_box
    return {
        "t": t, "z_ee": z_ee, "v_ee": v_ee, "z_box": z_box, "v_box": v_box,
        "f_fric": f_fric, "slipping": slipping, "slip": slip,
        "max_slip": float(np.max(slip)), "f_max": f_max, "f_needed": f_needed,
    }

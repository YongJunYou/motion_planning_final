"""M5: slip-aware squeeze-force regulation (spec D7, outside the OCP).

Given the desired transport acceleration a_z, compute the per-contact squeeze
normal-force setpoint that keeps the friction cone satisfied, plus a safety
margin and a low floor that keeps contact alive at low acceleration. Map the
setpoint to a penetration command via the M2 inverse contact model and feed it
into the OCP cost as the squeeze reference.
"""
import numpy as np

from model.box import required_normal_force
from model.contact import penetration_for_force, penetration_for_force_approx


def slip_aware_normal_force(m_o, a_z, mu, margin=0.0, floor=0.0, g=9.81, n_contacts=2):
    """Per-contact squeeze setpoint with a safety margin and a low floor.

    F_n = max(m_o*(g+a_z)/(n_contacts*mu) + margin, floor).
    """
    F_req = required_normal_force(m_o, a_z, mu, g, n_contacts) + margin
    return np.maximum(F_req, floor)


def slip_aware_penetration(m_o, a_z, mu, k, eps, margin=0.0, floor=0.0,
                           g=9.81, n_contacts=2, exact=True):
    """Penetration command realizing the slip-aware squeeze setpoint."""
    F_n = slip_aware_normal_force(m_o, a_z, mu, margin, floor, g, n_contacts)
    if exact:
        return penetration_for_force(F_n, k, eps)
    return penetration_for_force_approx(F_n, k)

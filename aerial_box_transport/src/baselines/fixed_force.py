"""Baseline 1: constant squeeze force, independent of transport acceleration.

This is the foil for the slip-aware policy. It slips once a_z passes the
critical acceleration where the constant squeeze can no longer supply the
friction needed to carry the box (spec section 8).
"""
import numpy as np


def fixed_normal_force(F_fixed, a_z):
    """Constant squeeze setpoint, broadcast over a_z."""
    a_z = np.asarray(a_z, dtype=float)
    return np.full_like(a_z, float(F_fixed))


def critical_acceleration(F_fixed, m_o, mu, g=9.81, n_contacts=2):
    """a_crit where the fixed force exactly meets the friction requirement.

    n_contacts*mu*F_fixed = m_o*(g + a_crit)  =>  a_crit = n_contacts*mu*F_fixed/m_o - g.
    Above a_crit the fixed-force grasp slips.
    """
    return n_contacts * mu * F_fixed / m_o - g

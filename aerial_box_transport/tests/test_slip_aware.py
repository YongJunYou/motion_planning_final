import numpy as np

from baselines.fixed_force import critical_acceleration
from model.box import required_normal_force, will_slip
from model.contact import smooth_normal_force
from planner.slip_aware import slip_aware_normal_force, slip_aware_penetration

G = 9.81


def test_slip_aware_meets_or_exceeds_requirement():
    m_o, mu = 0.3, 0.6
    a_z = np.linspace(0, 20, 101)
    F_slip = slip_aware_normal_force(m_o, a_z, mu, margin=0.5, floor=2.0)
    F_req = required_normal_force(m_o, a_z, mu, G, 2)
    assert np.all(F_slip >= F_req - 1e-9)


def test_slip_aware_never_slips_across_sweep():
    m_o, mu = 0.3, 0.6
    for a in np.linspace(0, 20, 41):
        F = float(slip_aware_normal_force(m_o, a, mu, margin=0.5, floor=2.0))
        assert not will_slip(F, m_o, float(a), mu, G, 2)


def test_fixed_force_crosses_requirement_at_a_crit():
    m_o, mu, F_fixed = 0.3, 0.6, 5.0
    a_crit = critical_acceleration(F_fixed, m_o, mu, G, 2)
    assert abs(required_normal_force(m_o, a_crit, mu, G, 2) - F_fixed) < 1e-9
    assert not will_slip(F_fixed, m_o, a_crit - 0.5, mu, G, 2)
    assert will_slip(F_fixed, m_o, a_crit + 0.5, mu, G, 2)


def test_slip_aware_penetration_maps_back_to_force():
    m_o, mu, k, eps = 0.3, 0.6, 1700.0, 1.0e-4
    a_z = 8.0
    F = float(slip_aware_normal_force(m_o, a_z, mu, margin=0.5, floor=2.0))
    phi = float(slip_aware_penetration(m_o, a_z, mu, k, eps,
                                       margin=0.5, floor=2.0, exact=True))
    assert abs(smooth_normal_force(phi, k, eps) - F) < 1e-6

import numpy as np

from model.box import (
    BoxParams,
    box_vertical_accel,
    friction_cone_residual,
    required_normal_force,
    simulate_vertical_transport,
    will_slip,
)

G = 9.81


def test_required_normal_force_formula():
    m_o, mu, a_z = 0.3, 0.6, 5.0
    expected = m_o * (G + a_z) / (2 * mu)
    assert abs(required_normal_force(m_o, a_z, mu, G, 2) - expected) < 1e-12


def test_free_fall_with_zero_squeeze():
    a_box, f, slipping = box_vertical_accel(0.0, 0.3, 0.0, 0.6, G, 2)
    assert slipping
    assert abs(f) < 1e-12
    assert abs(a_box - (-G)) < 1e-9          # free fall


def test_static_hold_with_strong_squeeze():
    a_box, f, slipping = box_vertical_accel(50.0, 0.3, 0.0, 0.6, G, 2)
    assert not slipping
    assert abs(a_box) < 1e-9                 # holds still
    assert abs(f - 0.3 * G) < 1e-9           # friction balances weight


def test_will_slip_threshold():
    m_o, mu, a_z = 0.3, 0.6, 5.0
    F_req = required_normal_force(m_o, a_z, mu, G, 2)
    assert not will_slip(F_req + 1e-6, m_o, a_z, mu, G, 2)
    assert will_slip(F_req - 1e-3, m_o, a_z, mu, G, 2)


def test_friction_cone_residual_sign():
    assert friction_cone_residual(3.0, 10.0, 0.6) > 0    # 3 <= 6
    assert friction_cone_residual(7.0, 10.0, 0.6) < 0    # 7 > 6


def test_transport_sticks_when_squeeze_sufficient():
    box = BoxParams(m_o=0.3, mu=0.6)
    a_z = 4.0
    F_req = required_normal_force(box.m_o, a_z, box.mu, box.g, 2)
    out = simulate_vertical_transport(box, F_req + 1.0, a_z, t_accel=0.5, dt=1e-3)
    assert out["max_slip"] < 1e-6                        # no slip
    assert out["z_box"][-1] > 0.0                        # box was lifted


def test_transport_slips_when_squeeze_insufficient():
    box = BoxParams(m_o=0.3, mu=0.6)
    a_z = 12.0
    F_req = required_normal_force(box.m_o, a_z, box.mu, box.g, 2)
    out = simulate_vertical_transport(box, 0.5 * F_req, a_z, t_accel=0.5, dt=1e-3)
    assert out["max_slip"] > 1e-3                        # measurable slip
    assert bool(np.any(out["slipping"]))

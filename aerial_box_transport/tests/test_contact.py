import casadi as ca
import numpy as np

from model.contact import (
    cone_radius,
    penetration_for_force,
    penetration_for_force_approx,
    smooth_normal_force,
)

K = 1700.0
EPS = 1.0e-4


def test_value_at_zero():
    assert abs(smooth_normal_force(0.0, K, EPS) - 0.5 * K * EPS) < 1e-12


def test_positive_everywhere_and_decreasing():
    phis = np.linspace(-0.01, 0.01, 401)
    vals = smooth_normal_force(phis, K, EPS)
    assert np.all(vals > 0.0)
    assert np.all(np.diff(vals) <= 0.0)
    assert vals[0] > vals[-1]


def test_separation_and_penetration_asymptotes():
    assert 0.0 < smooth_normal_force(0.01, K, EPS) < 1e-3        # 1 cm gap -> ~0
    phi = -0.005                                                 # 5 mm penetration
    rel_err = abs(smooth_normal_force(phi, K, EPS) - (-K * phi)) / (-K * phi)
    assert rel_err < 1e-2                                        # -> -k*phi


def test_derivative_matches_analytic_through_zero():
    phis = np.linspace(-0.005, 0.005, 1001)
    h = 1e-8
    d_num = (smooth_normal_force(phis + h, K, EPS)
             - smooth_normal_force(phis - h, K, EPS)) / (2 * h)
    d_analytic = 0.5 * K * (phis / np.sqrt(phis ** 2 + EPS ** 2) - 1.0)
    assert np.all(np.isfinite(d_analytic))
    assert np.max(np.abs(d_num - d_analytic)) < 1e-1


def test_cone_radius():
    assert abs(cone_radius(10.0, 0.6) - 6.0) < 1e-12


def test_exact_inverse_roundtrip():
    for F in [2.0, 5.0, 10.0, 25.0]:
        phi = float(penetration_for_force(F, K, EPS))
        assert abs(smooth_normal_force(phi, K, EPS) - F) < 1e-6


def test_approx_inverse_close_for_large_force():
    F = 25.0
    phi_exact = float(penetration_for_force(F, K, EPS))
    phi_approx = float(penetration_for_force_approx(F, K))
    assert abs(phi_exact - phi_approx) < 1e-4


def test_casadi_backend_matches_and_differentiates():
    phi = ca.MX.sym("phi")
    expr = smooth_normal_force(phi, K, EPS, backend=ca)
    f = ca.Function("f", [phi], [expr])
    dlam = ca.Function("dlam_dphi", [phi], [ca.jacobian(expr, phi)])
    for p in [-0.003, 0.0, 0.002]:
        assert abs(float(f(p)) - smooth_normal_force(p, K, EPS)) < 1e-9
        fd = (smooth_normal_force(p + 1e-7, K, EPS)
              - smooth_normal_force(p - 1e-7, K, EPS)) / 2e-7
        assert abs(float(dlam(p)) - fd) < 1e-2

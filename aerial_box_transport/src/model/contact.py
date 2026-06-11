"""M2: smooth penetration-based contact model (lecture strategy 3).

The normal contact force is a smooth (C1) function of a virtual penetration
depth phi. There is no complementarity and no active set, and the normal force
is a DERIVED quantity, not a decision variable (spec D2, D3):

    lambda_n = 0.5 * k * (sqrt(phi^2 + eps^2) - phi)

Sign convention: phi > 0 is separation, phi < 0 is penetration.

    phi > 0  (separation):  lambda_n -> 0
    phi < 0  (penetration): lambda_n -> -k * phi   (linear spring, stiffness k)
    phi = 0:                lambda_n  = k * eps / 2, derivative continuous

The same expression is reused as an OCP constraint later, so it accepts either
numpy (default) or casadi as the math backend.
"""
import numpy as np


def smooth_normal_force(phi, k, eps, backend=np):
    """Smooth normal contact force from penetration phi.

    Pass backend=casadi to build the symbolic version for the OCP.
    """
    return 0.5 * k * (backend.sqrt(phi * phi + eps * eps) - phi)


def cone_radius(lambda_n, mu):
    """Friction-cone radius: the tangential force magnitude limit, mu * lambda_n."""
    return mu * lambda_n


def penetration_for_force(F_n_req, k, eps):
    """Exact inverse of smooth_normal_force: phi such that lambda_n(phi) = F_n_req.

    With y = 2 * F_n_req / k and sqrt(phi^2 + eps^2) = y + phi, squaring gives
    phi = (eps^2 - y^2) / (2 * y), valid for F_n_req > 0. For F_n_req <= 0 the
    requested force is unreachable in contact, so phi is +inf (separation).
    """
    F_n_req = np.asarray(F_n_req, dtype=float)
    y = 2.0 * F_n_req / k
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = (eps * eps - y * y) / (2.0 * y)
    return np.where(y > 0.0, phi, np.inf)


def penetration_for_force_approx(F_n_req, k):
    """Deep-penetration approximation phi ~= -F_n_req / k (spec section 4)."""
    return -np.asarray(F_n_req, dtype=float) / k

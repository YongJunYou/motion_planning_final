"""M1 unit tests (planner-side, no Isaac at runtime; uses config/*.json)."""
import numpy as np
import pinocchio as pin

from model.whole_body import WholeBody, build_planning_model


def test_planning_model_structure():
    m = build_planning_model()
    assert m.nq == 15 and m.nv == 14            # 7 free-flyer + 8 arm joints
    names = [m.names[i] for i in range(m.njoints)]
    for jn in ["dof_l1", "dof_l4", "dof_r1", "dof_r4"]:
        assert jn in names
    assert "dof_lb1" not in names               # tilts folded into the base


def test_total_mass_matches_robot():
    m = build_planning_model()
    total = sum(I.mass for I in m.inertias)
    assert abs(total - 4.8626) < 1e-2


def test_ee_fk_finite_and_jacobian_consistent():
    wb = WholeBody()
    q = pin.neutral(wb.model)
    q[7:] = 0.4
    p_l, p_r = wb.fk_ee(q)
    assert np.all(np.isfinite(np.array(p_l))) and np.all(np.isfinite(np.array(p_r)))
    J = np.array(wb.J_ee(q)[0])[:3, :]
    p0 = np.array(wb.fk_ee(q)[0]).ravel()
    fd = np.zeros((3, wb.nv))
    eps = 1e-6
    for i in range(wb.nv):
        dv = np.zeros(wb.nv)
        dv[i] = eps
        fd[:, i] = (np.array(wb.fk_ee(pin.integrate(wb.model, q, dv))[0]).ravel() - p0) / eps
    assert np.max(np.abs(fd - J)) < 1e-4


def test_mass_matrix_spd():
    wb = WholeBody()
    q = pin.neutral(wb.model)
    M = np.array(wb.M(q))
    assert M.shape == (14, 14)
    assert np.allclose(M, M.T, atol=1e-6)
    assert np.all(np.linalg.eigvalsh(M) > 0.0)      # symmetric positive definite

"""Acceleration sweep: fixed-force baseline versus slip-aware policy.

Reproduces the headline result. For a range of transport accelerations a_z that
crosses the critical value, it compares the required squeeze force, the fixed
baseline, and the slip-aware setpoint, and measures how far the box slips under
each policy during a constant-acceleration transport burst.

Run:  conda run -n am_dualarm python src/experiments/accel_sweep.py
Outputs: results/accel_sweep.npz, results/summary.txt, results/*.png
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np  # noqa: E402

import experiments.plots as plots  # noqa: E402
from baselines.fixed_force import critical_acceleration, fixed_normal_force  # noqa: E402
from config_io import load_config  # noqa: E402
from model.box import (  # noqa: E402
    BoxParams,
    required_normal_force,
    simulate_vertical_transport,
)
from planner.slip_aware import slip_aware_normal_force  # noqa: E402


def run_sweep(robot, task):
    c, sq = robot["contact"], robot["squeeze"]
    bx, sw, tr = task["box"], task["sweep"], task["transport"]
    g = task["gravity"]
    mu, n_c = c["mu"], c["n_contacts"]
    m_o = bx["m_o"]
    box = BoxParams(m_o=m_o, mu=mu, g=g)

    a_z = np.linspace(sw["a_z_min"], sw["a_z_max"], sw["n_points"])
    F_req = required_normal_force(m_o, a_z, mu, g, n_c)
    F_slip = slip_aware_normal_force(m_o, a_z, mu, sq["margin"], sq["floor"], g, n_c)
    F_fix = fixed_normal_force(sq["fixed_force"], a_z)
    a_crit = critical_acceleration(sq["fixed_force"], m_o, mu, g, n_c)

    slip_fixed = np.zeros_like(a_z)
    slip_slip = np.zeros_like(a_z)
    for i, a in enumerate(a_z):
        of = simulate_vertical_transport(box, float(F_fix[i]), float(a),
                                         tr["t_accel"], tr["dt"], n_c)
        os_ = simulate_vertical_transport(box, float(F_slip[i]), float(a),
                                          tr["t_accel"], tr["dt"], n_c)
        slip_fixed[i] = of["max_slip"]
        slip_slip[i] = os_["max_slip"]

    drop = sw["drop_threshold"]
    return {
        "a_z": a_z, "F_req": F_req, "F_slip": F_slip, "F_fix": F_fix,
        "a_crit": float(a_crit), "slip_fixed": slip_fixed, "slip_slip": slip_slip,
        "drop_threshold": float(drop), "F_fixed_value": float(sq["fixed_force"]),
        "dropped_fixed": slip_fixed > drop, "dropped_slip": slip_slip > drop,
    }


def write_summary(res, path):
    az = res["a_z"]
    fixed_drop = az[res["dropped_fixed"]]
    slip_drop = az[res["dropped_slip"]]
    fixed_drop_str = (f"{fixed_drop[0]:.3f} m/s^2" if fixed_drop.size else "none in range")
    slip_drop_str = (f"{slip_drop[0]:.3f} m/s^2" if slip_drop.size else "none in range")
    lines = [
        "Acceleration sweep summary",
        "==========================",
        f"fixed squeeze force          : {res['F_fixed_value']:.3f} N",
        f"critical acceleration a_crit : {res['a_crit']:.3f} m/s^2",
        f"drop threshold               : {res['drop_threshold'] * 100:.1f} cm",
        f"fixed-force first drop at    : {fixed_drop_str}",
        f"slip-aware first drop at     : {slip_drop_str}",
        f"max fixed-force slip         : {res['slip_fixed'].max() * 100:.2f} cm",
        f"max slip-aware slip          : {res['slip_slip'].max() * 100:.2f} cm",
    ]
    text = "\n".join(lines)
    with open(path, "w") as f:
        f.write(text + "\n")
    return text


def main():
    robot, task = load_config()
    res = run_sweep(robot, task)
    results_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "results"))
    os.makedirs(results_dir, exist_ok=True)
    np.savez(os.path.join(results_dir, "accel_sweep.npz"), **res)
    summary = write_summary(res, os.path.join(results_dir, "summary.txt"))
    plots.plot_force_vs_accel(res, os.path.join(results_dir, "force_vs_accel.png"))
    plots.plot_slip_vs_accel(res, os.path.join(results_dir, "slip_vs_accel.png"))
    print(summary)
    print(f"\nFigures and summary written to {results_dir}")


if __name__ == "__main__":
    main()

"""Headline plots for the acceleration sweep (spec section 8).

Style: no em-dashes. Uses a headless Agg backend so figures are written to disk
without a display.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_force_vs_accel(res, path):
    """Required F_n, fixed-force line, and slip-aware setpoint vs a_z."""
    a_z = res["a_z"]
    F_req = res["F_req"]
    F_fix = res["F_fix"]
    a_crit = float(res["a_crit"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(a_z, F_fix, F_req, where=(F_req > F_fix), color="C3",
                    alpha=0.12, label="slip: fixed force fails")
    ax.plot(a_z, F_req, "k--", lw=2, label="required F_n (friction limit)")
    ax.plot(a_z, res["F_slip"], "C0-", lw=2, label="slip-aware setpoint")
    ax.plot(a_z, F_fix, "C3-", lw=2, label="fixed force")
    ax.axvline(a_crit, color="0.4", ls=":", lw=1.5)
    ymax = ax.get_ylim()[1]
    ax.annotate(f"a_crit = {a_crit:.1f} m/s^2", xy=(a_crit, 0.9 * ymax),
                xytext=(a_crit + 0.4, 0.9 * ymax), color="0.3", fontsize=9)
    ax.set_xlabel("transport acceleration a_z [m/s^2]")
    ax.set_ylabel("per-contact squeeze force F_n [N]")
    ax.set_title("Squeeze force vs transport acceleration")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_slip_vs_accel(res, path):
    """Peak slip during the transport burst vs a_z, for both policies."""
    a_z = res["a_z"]
    a_crit = float(res["a_crit"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(a_z, res["slip_fixed"] * 100.0, "C3-", lw=2, label="fixed force")
    ax.plot(a_z, res["slip_slip"] * 100.0, "C0-", lw=2, label="slip-aware")
    ax.axhline(float(res["drop_threshold"]) * 100.0, color="0.4", ls=":",
               lw=1.5, label="drop threshold")
    ax.axvline(a_crit, color="0.4", ls=":", lw=1.0)
    ax.set_xlabel("transport acceleration a_z [m/s^2]")
    ax.set_ylabel("peak slip during transport [cm]")
    ax.set_title("Box slip vs transport acceleration")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)

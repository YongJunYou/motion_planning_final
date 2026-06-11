"""Phase schedule and reference helpers for the M4 multi-phase OCP."""
import numpy as np

PHASES = ("approach", "grasp", "transport", "release")


def phase_schedule(durations, dt):
    """Return (phase_of, N, times) for fixed per-phase durations [s].

    phase_of[k] is the phase name of control interval k; times has N+1 entries.
    """
    phase_of, times, t = [], [], 0.0
    for ph in PHASES:
        for _ in range(int(round(durations[ph] / dt))):
            phase_of.append(ph)
            times.append(t)
            t += dt
    times.append(t)
    return phase_of, len(phase_of), np.asarray(times)


def smoothstep(s):
    s = float(np.clip(s, 0.0, 1.0))
    return s * s * (3.0 - 2.0 * s)

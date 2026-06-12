"""Time the OCP solve, for the planning-time comparison against the sampling-based planner.

The OCP produces a dynamically feasible, contact / slip-aware whole-body TRAJECTORY for the FULL
sequence (approach + grasp + transport + release) in a single nonlinear solve. The sampling-based
baseline (src/planner/sampling_compare.py, run in the am_sampling env) instead finds a collision-free
GEOMETRIC path on the same bounding-box geometry. This script reports the OCP wall-clock so the two
can be put side by side.

We report:
  - model build (Pinocchio WholeBody) : one-time setup, amortized (the sampler amortizes its model too)
  - solve_ocp total                   : end-to-end (build + transcription + IPOPT)
  - IPOPT-only (total - build)         : the optimization itself, ~ the analogue of planner.solve()

Run: conda run -n am_dualarm python src/planner/time_ocp.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from model.whole_body import WholeBody  # noqa: E402
from planner.ocp import solve_ocp  # noqa: E402


def main():
    t0 = time.time()
    WholeBody()
    t_build = time.time() - t0

    times = []
    status = None
    for i in range(3):                       # a few solves; report median (IPOPT time is stable)
        t0 = time.time()
        res = solve_ocp(verbose=False)
        dt = time.time() - t0
        status = res["status"]
        times.append(dt)
        print(f"  solve {i + 1}: total={dt:.2f}s  status={status}")

    times.sort()
    t_total = times[len(times) // 2]
    print("\n=== OCP planning time (full approach+grasp+transport+release in one solve) ===")
    print(f"WholeBody build (one-time) : {t_build:5.2f} s")
    print(f"solve_ocp total (median)   : {t_total:5.2f} s   status={status}")
    print(f"IPOPT-only (total - build) : {t_total - t_build:5.2f} s")


if __name__ == "__main__":
    main()

"""Grasp-constrained window-passage feasibility test with the CAPSULE/BOX (coal GJK) body model.

Plans the carried-box transport q_grasp -> q_place on the grasp manifold, where the rack sits BEHIND
the awning window (place box at x=-2, wall at x=-1.05), so the whole body must thread the tilted-sash
opening. The arm links are capsules (tight for thin pipes), base + box are boxes. Reports whether a
collision-free grasp-constrained path exists and, if so, the base tilt it needed.

Run: conda run -n am_sampling python src/planner/window_test.py [timeout]
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np  # noqa: E402

from planner.sampling_compare import Geometry  # noqa: E402
from planner.grasp_constrained import GraspManifold, plan_constrained, _offsets  # noqa: E402


def _tilt_deg(theta):
    """Base tilt (angle of the rotated z-axis from world z) for a rotation-vector attitude."""
    q = theta
    ang = float(np.linalg.norm(q))
    if ang < 1e-9:
        return 0.0
    k = q / ang
    # z-axis of exp(theta) . dot with world z = cos component
    R22 = np.cos(ang) + k[2] * k[2] * (1 - np.cos(ang))
    return float(np.degrees(np.arccos(np.clip(R22, -1, 1))))


def main():
    timeout = float(sys.argv[1]) if len(sys.argv) > 1 else 240.0
    geom = Geometry()
    _, _, q_grasp, q_place = geom.configs()
    pl, pr = geom.fk14(q_grasp)
    sep = float(pr[0] - pl[0])
    man = GraspManifold(geom, sep)

    print(f"WINDOW capsule/box (coal GJK) grasp-constrained passage test, timeout={timeout:.0f}s")
    print(f"  r_link(capsule)={geom.r_link} base_box={geom.base_box} box={geom.box_size.round(3).tolist()}")
    print(f"  grasp box x={0.5*(pl[0]+pr[0]):+.2f}  place behind wall (x=-1.05)")
    print(f"  start valid={man.valid(man.project(q_grasp))}  goal valid={man.valid(man.project(q_place))}")

    t0 = time.time()
    path, info = plan_constrained(man, q_grasp, q_place, timeout=timeout, delta=0.2)
    dt = time.time() - t0
    solved = path is not None
    print(f"\nSOLVED={solved} in {dt:.1f}s  info="
          f"{ {k: (round(v,3) if isinstance(v,float) else v) for k,v in info.items()} }")
    if not solved:
        return
    tilts = [_tilt_deg(q[3:6]) for q in path]
    clears = all(geom.window_clear_body(q) for q in path)
    m0, x0 = _offsets(geom, path)
    print(f"  path {len(path)} states; all knots window-clear (coal)={clears}")
    print(f"  base tilt over path: max {max(tilts):.1f} deg, mean {np.mean(tilts):.1f} deg")
    print(f"  box<->base offset: mean {m0*100:.1f} cm, max {x0*100:.1f} cm")
    np.save("/tmp/window_capsule_path.npy", path)
    print("  saved path -> /tmp/window_capsule_path.npy")


if __name__ == "__main__":
    main()

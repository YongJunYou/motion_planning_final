"""Export a planned 14-DoF config path to a kinematic-playback file the IsaacSim viewer can render.

For VERIFYING a planned path (does the body thread the window?), a kinematic playback is clearer than
closed-loop tracking: it poses the robot + box EXACTLY at each planned config, with no controller
tracking error and no friction-grip slip to confound what we are checking. We precompute every world
pose here (this env has the Pinocchio FK) and write them out; the viewer (am_isaac) just renders them.

Densifies the path (linear interp in config space) for smooth motion and runs FK at each frame to get
the carried box pose (pad midpoint, oriented with the base since the grip is rigid in the base frame).

Run: conda run -n am_sampling python src/planner/export_play.py [path.npy] [out.npz]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np  # noqa: E402

from planner.sampling_compare import Geometry  # noqa: E402

SPAWN = np.array([0.0, 0.0, 1.5])             # sim world <- home frame offset (matches track_reference)
BOX_BASE_TO_CENTER = 0.079                    # box prim origin at its base; center is this far up


def rotvec_to_quat_wxyz(th):
    ang = float(np.linalg.norm(th))
    if ang < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    ax = np.asarray(th) / ang
    s = np.sin(0.5 * ang)
    return np.array([np.cos(0.5 * ang), ax[0] * s, ax[1] * s, ax[2] * s])


def quat_to_R(q):                              # wxyz -> 3x3
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def R_to_quat_wxyz(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    if i == 0:
        s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                         (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if i == 1:
        s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                         0.25 * s, (R[1, 2] + R[2, 1]) / s])
    s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                     (R[1, 2] + R[2, 1]) / s, 0.25 * s])


def window_proxy(geom):
    """The planner's coal window obstacles (4 wall-border boxes + tilted sash) as (full_size, world
    center, quat_wxyz), so the viewer can spawn them as the SOLID, VISIBLE collision volumes."""
    full, cen, quat = [], [], []
    for g, T in geom._win_obs:
        full.append(2.0 * np.asarray(g.halfSide, float))
        cen.append(np.asarray(T.getTranslation(), float) + SPAWN)   # home -> sim world (z + 1.5)
        quat.append(R_to_quat_wxyz(np.asarray(T.getRotation(), float)))
    return np.array(full), np.array(cen), np.array(quat)


def densify(path, sub=8):
    """Linear interp each segment into `sub` steps (visual smoothing; no reprojection needed)."""
    out = []
    for a, b in zip(path[:-1], path[1:]):
        for t in np.linspace(0.0, 1.0, sub, endpoint=False):
            out.append(a + (b - a) * t)
    out.append(path[-1])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="/tmp/window_capsule_path.npy")
    ap.add_argument("out", nargs="?", default="/tmp/window_play.npz")
    ap.add_argument("--box", default="", help="npz with a 'box' array (OCP box PB per knot): render the "
                    "box at its TRUE trajectory (on the desk during approach, gripped during transport) "
                    "instead of the FK pad-midpoint, which floats with the open gripper before grasp")
    args = ap.parse_args()
    geom = Geometry()
    path = np.load(args.path)
    F = densify(path, sub=8)
    n = len(F)
    box_ref = None
    if args.box and os.path.exists(args.box):
        bref = np.load(args.box)["box"]
        if len(bref) == len(path):
            box_ref = densify(bref, sub=8)
    base_pos = np.zeros((n, 3))
    base_quat = np.zeros((n, 4))
    arm = np.zeros((n, 8))
    box_pos = np.zeros((n, 3))
    box_quat = np.zeros((n, 4))
    clear = np.ones(n, dtype=bool)
    for k, q in enumerate(F):
        base_pos[k] = SPAWN + q[0:3]
        bq = rotvec_to_quat_wxyz(q[3:6])
        base_quat[k] = bq
        arm[k] = q[6:14]
        pl, pr = geom.fk14(q)
        fk_mid = SPAWN + 0.5 * (pl + pr)
        if box_ref is not None:
            ctr = SPAWN + box_ref[k]                       # the OCP box position (PB)
            gripped = np.linalg.norm(fk_mid - ctr) < 0.06  # gripped => box rides the grip midpoint
            R = quat_to_R(bq) if gripped else np.eye(3)    # box stays LEVEL when resting on desk/rack
            box_quat[k] = bq if gripped else np.array([1.0, 0.0, 0.0, 0.0])
            box_pos[k] = ctr - R @ np.array([0.0, 0.0, BOX_BASE_TO_CENTER])
        else:
            box_pos[k] = fk_mid - quat_to_R(bq) @ np.array([0.0, 0.0, BOX_BASE_TO_CENTER])
            box_quat[k] = bq                               # rigid grip: box turns with the base
        clear[k] = geom.window_clear_body(q)
    proxy_full, proxy_cen, proxy_quat = window_proxy(geom)
    np.savez(args.out, base_pos=base_pos, base_quat=base_quat, arm=arm,
             box_pos=box_pos, box_quat=box_quat, clear=clear,
             proxy_full=proxy_full, proxy_cen=proxy_cen, proxy_quat=proxy_quat)
    tilt = np.degrees([np.arccos(np.clip(quat_to_R(base_quat[k])[2, 2], -1, 1)) for k in range(n)])
    print(f"wrote {args.out}: {n} frames (from {len(path)} knots)"
          + (f"  [box from {args.box}]" if box_ref is not None else ""))
    print(f"  window-clear frames: {int(clear.sum())}/{n}   base tilt max {tilt.max():.1f} deg")
    print(f"  window collision proxy: {len(proxy_full)} boxes (4 wall borders + sash)")
    print(f"  base world z: {base_pos[:,2].min():.2f}..{base_pos[:,2].max():.2f}  "
          f"box world x: {box_pos[:,0].min():.2f}..{box_pos[:,0].max():.2f}")


if __name__ == "__main__":
    main()

"""WINDOW hybrid: warm-start the whole-body OCP with the sampler's grasp-constrained window passage.

The sampler (am_sampling, coal capsule/box collision) found a collision-free route through the awning
window but only as a kinematic path (no dynamics, 8 inter-knot clips). The OCP (am_dualarm, IPOPT)
refines it into a dynamically feasible, slip-aware, TILT-MINIMIZED trajectory with a differentiable
window keep-out, removing the clips. Two stages (different conda envs):

  stage 1 (am_sampling):  build /tmp/window_seed.npz = the sampler route (q + box) on the grasp manifold
  stage 2 (am_dualarm):   solve_ocp(window=True, seed=route, use_cylinders=False) -> window_reference.npz

Run:
  conda run -n am_sampling python src/planner/hybrid_window.py --build
  conda run -n am_dualarm  python src/planner/hybrid_window.py --solve
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np  # noqa: E402

SEED = "/tmp/window_seed.npz"
PATH = "/tmp/window_capsule_path.npy"
TRANSPORT_DUR = 9.0    # 9 s reliably clears the box at the window (oriented-box penalty); the longer
#                        13 s horizon drifted to a low-tilt local min where the box clipped. The
#                        post-window flight lag at 9 s is open-space (no wall there), box still reaches rack.


def build():
    from planner.sampling_compare import Geometry
    geom = Geometry()
    path = np.load(PATH)                                    # (M,14) grasp -> place, home frame
    box = np.array([0.5 * np.add(*geom.fk14(q)) for q in path])   # box = pad midpoint (home frame)
    np.savez(SEED, q_route=path, box_route=box)
    print(f"built {SEED}: route {path.shape}, box x {box[:,0].min():.2f}..{box[:,0].max():.2f}, "
          f"base tilt max {np.degrees(np.abs(path[:,3:6]).sum(1).max()):.1f} deg (rough)")


def _ref_path():
    rdir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "results"))
    os.makedirs(rdir, exist_ok=True)
    mode = os.environ.get("WINDOW_MODE", "soft")
    name = "window_reference.npz" if mode == "soft" else f"window_reference_{mode}.npz"
    return os.path.join(rdir, name)


def solve():
    from planner.ocp import solve_ocp
    mode = os.environ.get("WINDOW_MODE", "soft")
    d = np.load(SEED)
    res = solve_ocp(window=True, use_cylinders=False, verbose=True, transport_dur=TRANSPORT_DUR,
                    window_mode=mode, seed={"q_route": d["q_route"], "box_route": d["box_route"]})
    out = _ref_path()
    np.savez(out, times=res["times"], base=res["base"], arm=res["arm"], box=res["box"],
             theta=res["theta"], phase_bounds=res["phase_bounds"], lam=res["lam"],
             fn_set=res["fn_set"], box_ref_z=res["box_ref_z"], box_ref=res["box_ref"],
             grite_ref=res["grite_ref"])
    print(f"\n[WINDOW-OCP] status: {res['status']}")
    print(f"[WINDOW-OCP] base x {res['base'][:,0].min():.2f}..{res['base'][:,0].max():.2f}, "
          f"z {res['base'][:,2].min():.2f}..{res['base'][:,2].max():.2f} m")
    print(f"[WINDOW-OCP] max base tilt: {res['max_tilt_deg']:.2f} deg")
    print(f"[WINDOW-OCP] box x {res['box'][:,0].min():.2f}..{res['box'][:,0].max():.2f} m")
    print(f"[WINDOW-OCP] wrote {out}")


def verify():
    """Stage 3 (am_sampling): check the OCP trajectory against the TRUE coal capsule/box collision
    model (the OCP only saw a smooth point-penalty surrogate), at knots and densely between them."""
    import numpy as np
    from planner.sampling_compare import Geometry
    geom = Geometry()
    d = np.load(_ref_path())
    base, theta, arm = d["base"], d["theta"], d["arm"]
    Q = np.concatenate([base, theta, arm], axis=1)               # (N+1,14) full configs
    N = len(Q)
    knot_clear = sum(int(geom.window_clear_body(Q[k])) for k in range(N))
    # dense inter-knot check (the executed motion interpolates between knots)
    bad = tot = 0
    for a, b in zip(Q[:-1], Q[1:]):
        for t in np.linspace(0, 1, 6, endpoint=False)[1:]:
            tot += 1
            bad += int(not geom.window_clear_body(a + (b - a) * t))
    tilt = np.degrees([np.linalg.norm(th) for th in theta])
    box = np.array([0.5 * np.add(*geom.fk14(q)) for q in Q])
    off = np.linalg.norm(box[:, :2] - base[:, :2], axis=1)
    print(f"coal verify: knots window-clear {knot_clear}/{N};  dense inter-knot clear {tot - bad}/{tot}")
    print(f"  max base tilt {tilt.max():.1f} deg;  box<->base offset mean {off.mean()*100:.1f} max {off.max()*100:.1f} cm")
    np.save("/tmp/window_ocp_path.npy", Q)
    print("  saved OCP path -> /tmp/window_ocp_path.npy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="stage 1 (am_sampling): build the seed")
    ap.add_argument("--solve", action="store_true", help="stage 2 (am_dualarm): solve the OCP")
    ap.add_argument("--verify", action="store_true", help="stage 3 (am_sampling): coal collision check")
    args = ap.parse_args()
    if args.build:
        build()
    if args.solve:
        solve()
    if args.verify:
        verify()
    if not (args.build or args.solve or args.verify):
        print("specify --build (am_sampling) | --solve (am_dualarm) | --verify (am_sampling)")


if __name__ == "__main__":
    main()

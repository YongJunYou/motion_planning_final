# Keyframe-guided window passage: results

Date: 2026-06-15 (recomputed from the canonical `_g2` references). Scene: dual-arm aerial manipulator
carries a cubebox_a01 (10.6 cm) box through a partially-open tilted awning window (wall solid except the
opening, sash tilted 32.4 deg) from the desk (x = +2) to the rack (x = -4). Home frame = world minus
(0, 0, 1.5).

The comparison is 2-way. Both references solve the SAME whole-body OCP (`solve_ocp`, soft_box window
mode, `use_cylinders=False`, 9 s transport); they differ only in what guides the window passage:

| reference | guidance |
|---|---|
| `window_reference_sampler_g2.npz` | sampler (CBiRRT) route seed + tracking (the competitor) |
| `window_reference_keyframe_g2.npz` | keyframe-interpolation seed + a soft keyframe waypoint (ours) |

(The earlier `wedge_only` ablation is dropped: the `win_under` wedge keep-out is too task-specific to
feature, and the keyframe waypoint alone selects the good homotopy.)

The keyframe is the teammate-flown pose, grip closed to the a01 box: base at world (-0.74, 0, 2.15),
base pitch +60 deg (no yaw), arm joints (0, 154.2, 92.6, 61.6) deg on both sides.

## Verification (coal GJK signed distance, identical method for both)

Per knot we pose the body (base box + 8 arm-link capsules + payload box) and measure the signed minimum
distance to the 5 window obstacles (4 wall borders + tilted sash). Negative = penetration.
`src/planner/verify_window.py` (am_sampling; defaults to the two g2 references).

| metric (149 knots) | sampler_g2 | **keyframe_g2 (ours)** |
|---|---|---|
| base min clearance | -2.7 cm (grazes) | **+2.8 cm (clear)** |
| arm min clearance | +4.3 cm | **+4.8 cm** |
| box min clearance | +9.1 cm | +7.5 cm |
| body-window penetrations | 2 / 149 | **0 / 149** |
| max base attitude (rotvec norm) | 84.1 deg | **53.0 deg** |
| base rotational path length | 327.8 deg | **251.2 deg** |
| base translational path length | 11.54 m | **10.13 m** |
| arm joint path length | 627.5 deg | **499.3 deg** |

At the wall crossing (box x in [-1.40, -0.70]), broken into roll / pitch / yaw (deg):

| component at crossing | sampler_g2 | keyframe_g2 |
|---|---|---|
| roll  (x) | -39 .. -7 | -3 .. +0 |
| pitch (y) | +12 .. +19 | +3 .. +50 |
| yaw   (z) | -80 .. -68 | +0 .. +3 |

(`src/planner/crossing_rpy.py` prints these; `src/planner/fig_crossing_rpy.py` plots them ->
`results/fig_crossing_rpy.png`. Path lengths: `src/planner/path_lengths.py`.)

## The result

The large attitude of the sampler solution is almost entirely YAW, not pitch. The sampler-seeded OCP
threads the window by yawing the base roughly 70 to 80 deg sideways (yaw -68..-80, pitch only +12..+19),
turning the long axis of the body to slip it edge-on through the opening. That maneuver fits
geometrically but grazes the frame (2 penetrating knots, base at -2.7 cm) and travels more (327.8 deg of
base rotation vs 251.2).

A single human keyframe (60 deg pitch, no yaw) instead guides a pitch-forward diagonal passage
(pitch +3..+50, yaw ~0): the drone leans into the tilted awning the way its geometry invites. This is
the only collision-free trajectory (0 / 149), with the best base and arm clearance and a much more
moderate total attitude (53 vs 84 deg). So the keyframe does not merely tweak the unguided path; it
selects a qualitatively different, physically intuitive, and strictly better homotopy class.

## Convergence (initialization study)

Naive (no seed) and linear-interp (TrajOpt-style straight grasp->place) initializations do NOT converge
on this non-convex narrow-gap problem (0/10 random restarts each; primal infeasibility plateaus 3 to 7
orders above the 1e-6 criterion, in IPOPT restoration). Both the sampler seed and the keyframe seed do
converge to a collision-free passage. Full convergence table: `docs/paper_tables_TODO.md` (Table I),
produced by `src/planner/table1_sweep.py`.

A nuance worth noting: the cold PASSAGE solves are collision-free for BOTH methods (sampler +0.6 cm,
keyframe +6.8 cm at the wall, 0 penetrations). It is the downstream placement-refinement continuation
(w_place / w_level / w_padlevel / w_rise, which produces the g2 references) that tightens the sampler's
yaw-squeeze into the -2.7 cm graze while the keyframe pitch passage stays clear (+2.8). The sampler
homotopy is the more fragile one.

## Keyframe-pitch robustness (sanity check)

Re-solving the keyframe passage at pitch 50 / 60 / 70 deg all converge to the same pitch-forward
homotopy (yaw within +-1 deg; pitch leads and scales 49 / 53 / 63 deg) and stay collision-free (base min
clearance +5.4 / +6.8 / +4.3 cm). `src/planner/table3_sweep.py`; Table III in the paper doc.

## Reproduce

```
cd /home/jaewoo/Research/aerial_box_transport

# the g2 references already exist in results/. To rebuild the keyframe one (am_sampling, then am_dualarm):
conda run -n am_sampling python src/planner/build_kf_seed.py
W_KF=40 W_TRK_TH=0 conda run -n am_dualarm python src/planner/keyframe_window.py

# coal verification of both g2 references (am_sampling)
conda run -n am_sampling python src/planner/verify_window.py
# path lengths + crossing roll/pitch/yaw figure (am_sampling)
conda run -n am_sampling python src/planner/path_lengths.py
conda run -n am_sampling python src/planner/fig_crossing_rpy.py

# GUI: alternate the two passages back-to-back in ONE env
#   kinematic (exact plan, no tracking error):
conda run -n am_isaac python src/sim/play_alternate.py \
    --playA /tmp/play_sampler_g2.npz --playB /tmp/play_keyframe_g2.npz --proxy
#   physics (gRITE closed-loop, friction grip, real colliders):
conda run -n am_isaac python src/sim/track_alternate.py \
    --refA results/window_reference_sampler_g2.npz --refB results/window_reference_keyframe_g2.npz
```

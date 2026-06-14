# Keyframe-guided window passage: results

Date: 2026-06-14. Scene: dual-arm aerial manipulator carries a cubebox_a01 (10.6 cm) box through a
partially-open tilted awning window (wall solid except the opening, sash tilted 32.4 deg) on the way
from the desk (x = +2) to the rack (x = -4). Home frame = world minus (0, 0, 1.5).

All three references solve the SAME whole-body OCP (`solve_ocp`, soft_box window mode,
`use_cylinders=False`, 9 s transport). They differ only in what guides the window passage:

| reference | guidance |
|---|---|
| `soft_box` | sampler route seed only (no keyframe, no wedge) |
| `wedge_only` | sampler seed + the `win_under` wedge keep-out (forbid the body above the sash) |
| `keyframe` | keyframe-interpolation seed + a soft keyframe waypoint + the `win_under` wedge |

The keyframe is the teammate-flown pose, grip closed to the a01 box: base at world (-0.74, 0, 2.15),
base pitch +60 deg (no yaw), arm joints (0, 154.2, 92.6, 61.6) deg on both sides.

## Verification (coal GJK signed distance, identical method for all three)

Per knot we pose the body (base box + 8 arm-link capsules + payload box) and measure the signed
minimum distance to the 5 window obstacles (4 wall borders + tilted sash). Negative = penetration.
`src/planner/verify_window.py` (run in am_sampling). Its tilt agrees with the OCP's own reported
value (keyframe 53.2 deg), so the metric is trustworthy.

| metric (149 knots) | soft_box | wedge_only | **keyframe** |
|---|---|---|---|
| base min clearance | -3.8 cm | -0.9 cm | **+2.8 cm** |
| arm min clearance | +0.2 cm | +3.3 cm | **+5.7 cm** |
| box min clearance | +2.6 cm | +3.8 cm | **+4.1 cm** |
| body-window penetrations | 2 / 149 | 2 / 149 | **0 / 149** |
| max base attitude (rotvec norm) | 86.7 deg | 96.8 deg | **53.2 deg** |

At the wall crossing (box x in [-1.40, -0.70]), broken into roll / pitch / yaw:

| component at crossing | soft_box | wedge_only | keyframe |
|---|---|---|---|
| roll  (x) | -18.8 .. +0.6 | -26.3 .. +6.4 | -3.4 .. -0.0 |
| pitch (y) | -0.6 .. +16.3 | -8.6 .. +24.4 | +1.0 .. +47.5 |
| yaw   (z) | -85.5 .. -81.2 | -90.4 .. -89.3 | +3.4 .. +11.2 |

## The result

The large attitude of the unguided solutions is almost entirely YAW, not pitch. Without guidance the
OCP threads the window by yawing the base roughly 85 to 90 deg sideways, turning the long axis of the
body to slip it edge-on through the opening. That maneuver fits geometrically but grazes the frame
(2 penetrating knots, base at -3.8 / -0.9 cm) and is an awkward sideways turn.

A single human keyframe (60 deg pitch, no yaw) instead guides a pitch-forward diagonal passage: the
drone leans into the tilted awning the way its geometry invites, with almost no yaw. This is the only
collision-free trajectory (0 / 149), it has the best clearance on every body part (base, arm, box),
and it does so at a much more moderate total attitude (53 deg vs 87 to 97 deg).

So the keyframe does not merely tweak the unguided path. It selects a qualitatively different,
physically intuitive, and strictly better homotopy class. This is the direct evidence for the
hypothesis that a human keyframe guides the optimizer to a better window passage than the unguided
(sampler-only) and constraint-only variants.

## Solver note

The keyframe-guided solve is sensitive to the guidance weights. W_KF = 100 with attitude
route-tracking W_TRK_TH = 80 is too stiff: IPOPT reaches a near-feasible point and then the dual
infeasibility blows up (1e15, MUMPS runs out of memory). Dropping the attitude route-tracking
(W_TRK_TH = 0, it over-constrains every transport knot) and lowering the waypoint to W_KF = 40 lets
the restoration phase recover and the solve finishes (Solved To Acceptable Level, 211 iterations,
223 s). The keyframe-interpolation seed already carries the 60 deg tilt at the window knot, so a
gentle waypoint plus the wedge constraint is enough.

## Reproduce

```
# keyframe-guided solve (am_dualarm) -> results/window_reference_keyframe.npz
cd /home/jaewoo/Research/aerial_box_transport
W_KF=40 W_TRK_TH=0 conda run -n am_dualarm python src/planner/keyframe_window.py

# verification table (am_sampling)
conda run -n am_sampling python src/planner/verify_window.py

# GUI playback of the keyframe path (am_isaac)
conda run -n am_sampling python src/planner/export_play.py /tmp/window_kf_path.npy \
    /tmp/window_kf_play.npz --box results/window_reference_keyframe.npz
conda run -n am_isaac python src/sim/play_path.py --play /tmp/window_kf_play.npz --proxy --loop
```

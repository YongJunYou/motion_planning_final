# Window-passage comparison: sampling+OCP vs keyframe+OCP (ours)

| metric | sampling+OCP (baseline) | keyframe+OCP (ours) |
|---|---|---|
| SR (task success) [%] | 60.0 | 100.0 |
|   trials (success/total) | 3/5 | 1/1 (deterministic) |
|   collision-free (0 pen) | True | True |
|   window-penetrating knots | 0 | 0 |
| MCM min clearance [cm] | +1.06 | +2.30 |
|   base min [cm] | +1.06 | +2.30 |
|   arm  min [cm] | +5.01 | +4.68 |
|   box  min [cm] | +5.21 | +5.17 |
| max base attitude [deg] | 57.3 | 53.8 |
| TEp position RMSE [mm] | 41.0 | 26.9 |
| PL SE(3)+joint arc length | 20.74 | 17.37 |
|   base translation [m] | 13.44 | 10.37 |
|   base rotation [rad] | 5.65 | 5.33 |
|   joint motion [rad] | 10.90 | 8.64 |

## Sampling+OCP baseline variability over 5 independent trials
(the keyframe method is deterministic — no run-to-run variation)

- end-to-end SR (converged + collision-free + delivered): **3/5 = 60.0%**
- MCM [cm]: -3.71 ± 10.66 (min -24.67, max 5.11)
- max base attitude [deg]: 90.14 ± 30.31 (min 52.1, max 125.7)
- PL SE(3)+joint arc length: 22.64 ± 1.04 (min 20.74, max 23.55)

The headline columns above use one representative successful run (`window_reference_soft_box.npz`); the spread shows the sampler's run-to-run instability.

Sampler route-finding SR (CBiRRT, 15 seeds): 15/15 = 100% (finds *a* collision-free kinematic route; mean tilt 55.0°, highly variable → quality, not feasibility, is the gap).

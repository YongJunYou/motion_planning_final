# aerial_box_transport

Contact-aware whole-body trajectory optimization for dual-arm aerial box transport
with slip-aware force regulation.

The research design and locked decisions live in
`../motion_planning_final/IMPLEMENTATION_SPEC.md`, and the build plan in
`../motion_planning_final/ROADMAP.md`. This package is the planner half: pure Python
(Pinocchio + CasADi + IPOPT). It runs and is unit-tested WITHOUT Isaac Sim.

## Layout
- `config/robot.yaml`  contact stiffness k, smoothing eps, friction mu, squeeze margins, model source
- `config/task.yaml`   box mass and size, gravity, transport profile, acceleration sweep
- `src/model/`         M1 whole-body model (later), M2 smooth contact, M3 box and friction cone
- `src/planner/`       M4 multi-phase OCP (later), M5 slip-aware force law
- `src/baselines/`     fixed-force baseline, decoupled baseline (later)
- `src/experiments/`   acceleration sweep and headline plots
- `src/sim/`           Isaac Sim validation (later, gated on the teammate scene)
- `tests/`             pytest unit tests for the planner half
- `results/`           generated figures and logs

## Environment
Use the `am_dualarm` conda env (Pinocchio with pinocchio.casadi, CasADi, IPOPT, numpy, scipy, matplotlib, pyyaml, pytest).

## MVP (Phase 0)
A reduced vertical-channel box squeezed by two end-effectors, the smooth contact model,
the slip-aware squeeze law versus a fixed-force baseline, and an acceleration sweep that
shows the fixed force slipping once acceleration passes a critical value while the
slip-aware law does not.

Run:
```
conda run -n am_dualarm pytest -q
conda run -n am_dualarm python src/experiments/accel_sweep.py
```
Figures and a summary are written to `results/`.

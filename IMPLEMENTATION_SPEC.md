# Implementation Spec: Contact-Aware Whole-Body Trajectory Optimization for Dual-Arm Aerial Box Transport with Slip-Aware Force Regulation

This document is the implementation brief for a coding agent. It captures the full research
direction, the design decisions that are already locked, the modules to build, and the order to
build them in. Read the whole document before writing code. When a decision here conflicts with a
"more standard" approach, follow this document. The conversation that produced these decisions is
not available to you, so treat this file as the single source of truth.

Style rule for any prose you generate (comments, README, reports): do not use em-dashes.

---

## 1. One-paragraph summary

We plan and validate a coordinated dual-arm aerial manipulator that grasps a box by squeezing it
between two end-effectors and carries it through the air. The box is held only by friction (no form
closure). The headline contribution is slip-aware force regulation: the squeeze (normal) force is
scaled up with the transport acceleration so the friction cone stays satisfied, whereas a fixed
squeeze force slips once acceleration passes a critical value. Planning is done offline as a single
whole-body multi-phase trajectory optimization (OCP) and the resulting reference is tracked online.
Validation is in Isaac Sim. The contribution is demonstrated by an acceleration sweep comparing a
fixed-force baseline against the slip-aware policy.

---

## 2. Platform and task

### Platform (OAM-4D)
- Omnidirectional variable tilt-rotor base (4 rotors, each on a 2-DoF tilt mount) that can hover at
  arbitrary attitude.
- Two arms, 4-DoF each, with a deformable pad / gripper at each end-effector (EE).
- For planning we use a reduced model (see Decision D5). The rotors and tilt mounts are NOT in the
  planning model. The base is treated as a fully actuated 6-DoF floating body driven by a base
  wrench, because thrust and tilt allocation is delegated to a low-level robust controller.

### Task
- A known-size box. Two target poses at the centers of two opposite faces.
- The two EEs squeeze those faces and hold the box by friction.
- Full cycle, in order: approach, grasp, lift and transport, place (release).
- Deformable pads: how far the commanded EE pose virtually penetrates the box face sets the contact
  force (predictive admittance idea).

---

## 3. Locked design decisions (do not relitigate)

These were decided deliberately. Do not replace them with alternatives even if alternatives look
more general or more standard.

- D1. Architecture 1: offline whole-body trajectory optimization produces a reference, tracked
  online by a controller. This is NOT online receding-horizon MPC. Do not implement a per-step
  re-solving MPC loop. (An MPC extension is explicitly out of scope for now.)
- D2. Smooth contact model (lecture "strategy 3"). Contact force is an explicit smooth function of a
  virtual penetration depth. Do NOT use complementarity constraints, active-set, or contact-implicit
  formulations.
- D3. Contact normal force is a DERIVED quantity from penetration `phi(q)`, not an independent
  decision variable. (Contrast with contact-implicit methods like IDTO where force is a true
  decision variable under complementarity. We are not doing that.)
- D4. Multi-phase means the cost and reference change per phase. The dynamics and the contact model
  are identical and smooth across all phases. Phases do NOT swap the dynamics equations. A light FSM
  only switches which cost terms / references are active at each time node.
- D5. The OCP plans only the base SE(3) trajectory and the arm joint references. Thrust and tilt
  allocation is delegated to a robust controller (gRITE / DOB) and is outside the OCP. The OCP does
  not model rotors or tilt mounts.
- D6. The box is a separate free rigid body, modeled independently from the robot and coupled only
  through the contact force `lambda`. Reasons: the box is unactuated, contact can break (approach
  phase), and box parameters (mass, inertia, mu) are identified and swept independently.
- D7. The slip-aware policy (M5) lives OUTSIDE the OCP. Given the desired transport acceleration
  profile, it computes a squeeze setpoint and feeds it into the OCP cost as a reference. (It is
  acceptable later to fold it into the cost as a stretch, but the default is outside.)
- D8. If a force-feedback loop is implemented at execution time, it is a light admittance correction
  realized by a local Jacobian-based IK step, NOT an OCP re-solve and NOT a whole-body MPC. It only
  adjusts the commanded penetration depth (a few mm) on top of the fixed reference.

Terminology to use consistently: "dual-arm" (not bimanual), "transport" (not pick-and-transport),
"trajectory optimization / OCP" for the planner (not MPC/NMPC).

---

## 4. Key relations and equations (intended formulation)

Whole-body dynamics (robot):

    M(q) qddot + h(q, qdot) = tau + J^T lambda

Smooth contact normal force (per contact i), the core of strategy 3:

    lambda_n = (k/2) * ( sqrt(phi^2 + eps^2) - phi )

    phi > 0  (separation):  lambda_n -> 0
    phi < 0  (penetration): lambda_n -> -k*phi   (linear spring, stiffness k)
    phi = 0:                lambda_n = k*eps/2,  derivative continuous everywhere

Friction (box held only by friction, no form closure):

    |F_t,i| <= mu * F_n,i                      (friction cone, per contact)
    |F_t|   ~= 0.5 * m_o * (g + a_z)            (per contact, two contacts share the load)
    F_n     >= m_o * (g + a_z) / (2*mu)         (required squeeze, the slip-aware law)

Inverse contact model (squeeze setpoint -> penetration command), deep-penetration approximation:

    phi_cmd ~= - F_n_req / k     (or solve lambda_n(phi_cmd) = F_n_req exactly)

Units are SI throughout: meters, kilograms, seconds, newtons, radians. The proposal plots show
penetration in cm for readability, but internally use meters.

---

## 5. Repository layout (suggested)

```
aerial_box_transport/
  README.md
  requirements.txt
  config/
    robot.yaml          # link masses, inertias, arm DoF, EE frame names, k, eps, mu, margins
    task.yaml           # box size/mass/inertia, target poses, phase timings, a_z profile
  src/
    model/
      whole_body.py     # M1: Pinocchio + casadi dynamics, FK, contact Jacobians
      contact.py        # M2: gap phi, smooth normal force, cone radius
      box.py            # M3: free-body dynamics, friction cone
    planner/
      transcription.py  # multiple shooting / collocation helpers
      ocp.py            # M4: assemble NLP, phase cost FSM, solve with IPOPT
      slip_aware.py     # M5: F_n_req(a_z) and penetration setpoint
    sim/
      isaac_runner.py   # M6: load scene, drive reference, log
      identify_contact.py # M6: press test, fit k and eps
      sensors.py        # M6: virtual force sensor readout (wrist joint F/T)
    baselines/
      fixed_force.py    # baseline 1: constant squeeze (e.g. 5 N)
      decoupled.py      # baseline 2 (optional): plan base then arm separately
    experiments/
      accel_sweep.py    # sweep a_z, fixed vs slip-aware
      plots.py          # F_n vs a_z, slip rate vs a_z, force tracking
  results/
```

Important: the planner half (`src/model`, `src/planner`) is pure Python (Pinocchio + CasADi +
IPOPT) and must run and be testable WITHOUT Isaac Sim. The sim half (`src/sim`) needs Isaac Sim and
runs separately. Keep these decoupled so the OCP can be developed and unit-tested on its own.

---

## 6. Module specifications

Each module lists purpose, inputs, outputs, formulation notes, and acceptance criteria. Build them
to be importable and unit-testable in isolation.

### M1. Whole-body model (`src/model/whole_body.py`)

Purpose: provide symbolic (CasADi) dynamics, forward kinematics, and contact Jacobians for the
robot, suitable for use as OCP constraints.

How:
- Use `pinocchio` plus `pinocchio.casadi` (cpin). Build the model from URDF with a free-flyer root
  joint: `pin.buildModelFromUrdf(urdf, pin.JointModelFreeFlyer())`.
- Reduce out the rotor and tilt-mount joints with `pin.buildReducedModel` (fix them), so their mass
  and inertia fold into the parent bodies, OR maintain a separate planning-only URDF. The planning
  model is: floating base (6) + two 4-DoF arms (8 joints). Confirm exact joint count and EE frame
  names against the real URDF and fail loudly if they do not match.
- Cast to casadi: `cmodel = cpin.Model(model)`, `cdata = cmodel.createData()`.
- Provide functions: `M(q)` via `cpin.crba`, nonlinear terms `h(q, qdot)` via `cpin.rnea` with zero
  acceleration (or `nle`), forward kinematics for the two EE frames, and frame Jacobians via
  `cpin.computeFrameJacobian` / `getFrameJacobian`.
- Assemble forward dynamics so the OCP can write `xdot = f(x, u, lambda)`. State and control are
  defined in section 7.
- The base is fully actuated by a 6-DoF wrench (thrust/tilt are delegated). So the actuation maps
  the base wrench to the 6 base generalized forces directly, and arm torques to arm joints.

Acceptance:
- Given a numeric `q`, `M`, `h`, EE positions, and Jacobians return finite values and match a small
  finite-difference check.
- `f(x, u, lambda)` returns a state derivative of the correct dimension and is differentiable in
  CasADi (Jacobian builds without error).

### M2. Smooth contact model (`src/model/contact.py`)

Purpose: turn EE-to-box geometry into a differentiable normal force and a friction-cone radius.

How:
- Gap function per contact: take the EE position from M1 FK, transform into the box frame using the
  box pose, project onto the outward face normal, subtract the half-extent to that face.
  `phi_i = dot(n_face_b, R_box^T (p_ee_w - p_box_w)) - half_extent`. Sign convention: `phi > 0`
  separation, `phi < 0` penetration.
- Smooth normal force: `lambda_n = 0.5 * k * (sqrt(phi^2 + eps^2) - phi)`.
- Cone radius: `mu * lambda_n` for the friction constraint used by M3 / M4.
- Provide both the world-frame normal direction (`R_box @ n_face_b`) and a helper to build the
  contact force `F_i = lambda_n * n_hat + F_t,i` once the tangential part is known.
- `k` (N/m) and `eps` are parameters. Defaults: pick `k` so that a target operating force maps to a
  few mm of penetration (for example 5 N at 3 mm gives k ~= 1.7e3 N/m). Pick `eps` so `k*eps/2` is
  well below the operating force. These are later identified in sim (M6).

Acceptance:
- `lambda_n(phi)` and its CasADi derivative are continuous through `phi = 0` (test across a range).
- For large positive `phi`, `lambda_n` is near zero; for `phi << 0`, `lambda_n ~= -k*phi`.

### M3. Box free body and friction cone (`src/model/box.py`)

Purpose: close the box dynamics under gravity and contact, and enforce the friction cone.

How:
- Free rigid-body dynamics. Linear part is required: `m_o * vdot = sum_i F_i - m_o * g * e_z`.
  Rotational part is optional for the first version: `I_o * wdot + w x (I_o w) = sum_i r_i x F_i`,
  where `r_i` is the lever from box center of mass to contact point. For the MVP, the vertical linear
  channel is enough to produce slip; add rotation only if time allows.
- Friction cone constraint per contact: `norm(F_t,i) <= mu * lambda_n,i`. Use a second-order cone
  form, or a linearized pyramid if the solver prefers it.
- Coupling: the force the EE applies to the box is `F_i`. The reaction on the robot enters the
  robot generalized forces as `sum_i J_i^T (-F_i)`. Keep the sign convention explicit and consistent
  with M1.

Acceptance:
- With zero contact force the box free-falls under gravity (sanity test).
- With two symmetric squeeze forces large enough, the box can be held statically against gravity and
  the cone constraint is satisfied.

### M4. Multi-phase OCP (`src/planner/ocp.py`, `src/planner/transcription.py`)

Purpose: assemble and solve a single trajectory optimization over the full cycle.

How:
- Transcription: direct multiple shooting with RK4, or direct collocation. Either is fine. Use
  CasADi `Opti` or `nlpsol`, and solve with IPOPT.
- Decision variables over `N` time nodes: states `x_k`, controls `u_k`, and tangential contact
  forces `F_t,i,k` (the only contact quantity that is a decision variable; the normal force is
  derived from penetration per D3).
- Constraints applied at every node, identical across phases (per D4): dynamics defects (robot plus
  box, coupled through `lambda`), the smooth contact equality `lambda_n = f(phi)`, the friction cone,
  and joint / wrench / velocity limits.
- Phase schedule is prescribed by time, in order: approach, grasp, transport, release. A light FSM
  maps each node index to a phase and selects the active cost terms and references:
  - approach: drive EE to a pre-grasp pose in front of each face (no contact yet, `phi > 0`).
  - grasp: reach the target squeeze, i.e. track the penetration / normal-force setpoint.
  - transport: track the desired box pose trajectory and hold the squeeze setpoint (this is where
    M5 sets the squeeze setpoint as a function of `a_z`).
  - release: lower the box to the place pose and bring penetration back toward zero.
- Phase boundaries can be fixed in time for the first version (do not optimize phase durations).
- Output: a reference trajectory of base SE(3) and arm joints over time, plus the predicted contact
  forces and box trajectory for plotting.

Reference pseudocode for the per-node cost (illustrative, adapt as needed):

```python
for k in range(N):
    ph = phase_of[k]   # 'approach' | 'grasp' | 'transport' | 'release'
    if ph == 'approach':
        cost += w_pos * sumsqr(p_ee[k] - p_pregrasp)
    elif ph == 'grasp':
        cost += w_f * sumsqr(lam_n[k] - F_squeeze_target)
    elif ph == 'transport':
        cost += w_box * sumsqr(box_pose[k] - box_ref[k]) \
              + w_f   * sumsqr(lam_n[k] - F_squeeze_setpoint[k])   # M5 provides setpoint
    elif ph == 'release':
        cost += w_box * sumsqr(box_pose[k] - box_place) + w_rel * sumsqr(lam_n[k])
# dynamics defects, smooth-contact equality, friction cone, limits: applied at EVERY k, unchanged
```

Acceptance:
- The OCP solves to a locally optimal point on a simple instance and returns a smooth reference where
  the contact force ramps up at grasp, holds during transport, and drops at release (mirrors the
  proposal phase plot).
- No complementarity or active-set machinery anywhere.

### M5. Slip-aware force regulation (`src/planner/slip_aware.py`)

Purpose: compute the squeeze setpoint that keeps friction feasible during transport.

How (per D7, outside the OCP):
- Given the desired transport acceleration profile `a_z(t)` (vertical component dominant), compute
  the required normal force `F_n_req = m_o * (g + a_z) / (2*mu) + margin`. The margin is a safety
  buffer plus a small floor that keeps contact alive even at low acceleration.
- Map `F_n_req` to a penetration command via the inverse contact model `phi_cmd ~= -F_n_req / k`, or
  solve `lambda_n(phi_cmd) = F_n_req`. Feed this as the `F_squeeze_setpoint` reference into M4's
  grasp / transport cost.
- Baseline for comparison: fixed squeeze force (constant, e.g. 5 N), realized by a fixed penetration,
  in `src/baselines/fixed_force.py`. This baseline must be behaviorally distinct: it should slip once
  `a_z` passes the critical value where `mu * 2 * F_n_fixed < m_o (g + a_z)`.

Acceptance:
- For a swept `a_z`, the slip-aware setpoint stays at or above the required `F_n` curve, while the
  fixed-force line crosses it at the critical acceleration `a_crit`.

### M6. Isaac Sim validation (`src/sim/`)

Purpose: run the planned reference in the already-built Isaac Sim scene, identify the contact model,
and produce the metrics.

How:
- The scene (platform USD plus deformable pads plus box) is already built. Load it, then drive the
  base SE(3) and arm joint reference (through the low-level controller if available, or by directly
  commanding joints for a first pass).
- Virtual force sensor: prefer reading the wrist joint force/torque (articulation joint force /
  effort sensor). This measures the load through the wrist regardless of pad compliance, which is
  cleaner than a contact report through a deformable pad. The exact sensor API name varies by Isaac
  Sim version, so confirm against the installed version docs.
- Contact-model identification (the dashed feedback in the pipeline): script a slow press of the EE
  into a face, log penetration vs measured force, and fit `k` (and `eps`) to the M2 curve. Plug the
  fitted values back into the planner so planned squeeze and simulated force agree.
- Optional execution-time force feedback (per D8, stretch): admittance correction
  `delta_penetration = gain * (F_des - F_meas)`, realized as a local Jacobian IK step
  `delta_q = J_pinv * delta_x_ee` on the arm joints only. This is NOT an OCP re-solve. Tie the gain
  to the identified `k`.
- Metrics to log: grasp success rate, force-tracking error, maximum transport acceleration before
  slip, slip / drop rate vs acceleration, placement accuracy, computation time.

Acceptance:
- The reference runs in sim and the box is grasped and transported in the nominal (low-acceleration)
  case.
- The identification script outputs fitted `k`, `eps` and a force-vs-penetration plot.

---

## 7. State, control, and conventions

- Planning state: `x = [q_base, q_arm, p_box, q_box, v_base, v_arm, v_box]` where `q_base` is the
  floating-base pose (use a consistent SE(3) representation, e.g. position plus quaternion with the
  unit-norm constraint, or a Lie-group integrator), `q_arm` is the 8 arm joints, `p_box`/`q_box` the
  box pose, and the `v_*` the corresponding velocities / twists.
- Planning control: `u = [w_base, tau_arm]`, a 6-DoF base wrench plus 8 arm joint torques. An
  acceleration-level control is also acceptable if it simplifies the transcription; document whichever
  you choose.
- Contact decision variable: tangential force `F_t,i` per contact only. Normal force is derived.
- Frames: define and document the world frame, base frame, box frame, and EE frames once, and keep
  sign conventions for contact forces consistent between M2, M3, and M4.
- Units: SI everywhere.

---

## 8. Experiments

The headline result is an acceleration sweep.

- Sweep the commanded transport acceleration `a_z` over a range that crosses `a_crit`.
- For each `a_z`, run both the fixed-force baseline and the slip-aware policy.
- Produce these plots (`src/experiments/plots.py`):
  - Required `F_n = m_o (g + a_z) / (2 mu)` vs `a_z`, with the fixed-force horizontal line and the
    slip-aware setpoint line, shading the region above the fixed line as "slip: fixed force fails",
    and marking `a_crit`.
  - Slip / drop rate vs `a_z` for both methods.
  - Force-tracking error over a transport run.
- Hypothesis to confirm: slip-aware sustains higher acceleration without slip than the fixed force.
- Optional baseline (`decoupled.py`): plan the base trajectory first (ignoring arm and contact), fix
  it, then plan the arm and squeeze on top. This shows the value of joint whole-body planning. Mark
  it optional; do it only if time allows.

---

## 9. Implementation order (one week, MVP first)

Do these in order. Get a result early, then deepen. Do not start with the full whole-body OCP.

Phase 0, MVP (must produce a result):
1. Reduced box model. A planar box (3-DoF: vertical, horizontal, in-plane rotation, or even just the
   vertical channel) with two EE contact points whose positions set penetration into the two faces.
   Wire M2 (contact) and M3 (box plus cone) onto it. This alone reproduces slip and needs no
   full 24-DoF model.
2. Add M5 force law and the fixed-force baseline.
3. Drive a simple transport trajectory (vertical lift plus lateral move) with a prescribed `a_z`
   profile, for both policies.
4. Acceleration sweep and the headline plots (the `F_n` vs `a_z` figure and slip rate vs `a_z`).
5. Isaac Sim headline scenario plus the contact-model identification (fit `k`, `eps`).

Phase 1, stretch (technical depth):
6. Full whole-body M1 (Pinocchio reduced model) plus M4 multi-phase OCP over base SE(3) and arm
   joints, demonstrating a single smooth OCP across all four phases.
7. Decoupled baseline and ablations (vary margin, `k`, `mu`).

Phase 2, further stretch (only if everything above is solid):
8. Execution-time admittance plus Jacobian IK force-feedback loop in sim, and an ablation of
   "force feedback off vs on". This stays Architecture 1; it does not become MPC.

---

## 10. Scope guards (what NOT to do)

- Do not implement online receding-horizon MPC or a per-step OCP re-solve. Architecture 1 only.
- Do not make the contact normal force a decision variable. It is derived from penetration.
- Do not use complementarity constraints, active-set switching, or contact-implicit formulations.
- Do not swap the dynamics equations between phases. Only cost and references change per phase.
- Do not model rotors or tilt mounts in the OCP. The base is a 6-DoF wrench abstraction.
- Do not handle execution-time force feedback by re-solving the OCP. Use local admittance plus IK.
- Do not use em-dashes in any generated text.

---

## 11. Environment and tooling

- Planner: Python, `pinocchio` (with `pinocchio.casadi`), `casadi`, IPOPT (via CasADi `nlpsol`),
  `numpy`, `matplotlib`. This half must run without Isaac Sim and should have unit tests.
- Simulation: Isaac Sim (already-built scene). The USD-to-URDF and physics-property details have
  version-specific behavior; confirm sensor and importer/exporter APIs against the installed Isaac
  Sim version. Verify that the inertial parameters the planner uses match what the simulator
  actually uses (read mass / center of mass / inertia from the stage if needed, rather than trusting
  a lossy export).
- Keep `config/robot.yaml` and `config/task.yaml` as the single place for physical parameters
  (`m_o`, `I_o`, `mu`, `k`, `eps`, box dimensions, target poses, phase timings, margin), so the
  sweep and the identification can update them in one spot.

---

## 12. Open questions to confirm before or during coding

- Exact arm joint count and joint axes, and the EE frame names in the URDF (spec assumes 4-DoF per
  arm, two arms). Fail loudly if the URDF disagrees.
- Whether a usable URDF exists or only USD (if only USD, reading inertia from the stage is the
  reliable path).
- Box rotation: include the full 6-DoF box dynamics, or keep the linear channel for the MVP.
- Whether the low-level tracking controller is available in the sim, or whether the first pass
  commands joints directly.

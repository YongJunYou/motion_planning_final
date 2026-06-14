# ROADMAP: Dual-Arm Aerial Box Transport (Contact-Aware Whole-Body Trajectory Optimization)

Source of truth for the research design is `IMPLEMENTATION_SPEC.md` (locked decisions D1 to D8, modules M1 to M6). This file is the build plan and the reuse strategy. Style rule: no em-dashes in generated prose.

## 0. Status and decisions log
- 2026-06-09: Planner code lives in a NEW conda env `am_dualarm`. The sim half stays in `am_isaac`.
- 2026-06-09: The box plus deformable-pad Isaac scene will be provided by a teammate. M6 (Isaac validation) is gated on that delivery. All planner work proceeds first.
- 2026-06-09: The robot model comes ONLY from `dual_arm_final.usd` in this repo. The vlm_drone_ws hardware is a different platform, so its URDF and robot model are NOT reused.
- 2026-06-10: Reuse assessment of `/home/jaewoo/ros2_workspace/vlm_drone_ws/src` complete (section 2).
- 2026-06-11: M1 to M6 built. The whole-body OCP (approach, grasp, transport, release) solves the desk-to-rack transport, and gRITE closed-loop tracking in IsaacSim is validated (tilt under 1 deg, position error under 3.5 cm). The slip-aware force law beats the fixed-force baseline past the critical acceleration.
- 2026-06-12: Planner-comparison study added in a new env `am_sampling` (OMPL plus Pinocchio): OCP vs sampling vs hybrid on the same narrow-passage geometry (see `aerial_box_transport/README.md`). NEW research direction started for ICRA/IROS: carry the box through a partially-open tilted awning window, which requires whole-body reconfiguration.
- 2026-06-13: Window passage solved. The body is modeled with coal (GJK/EPA: arm capsules, base and box as boxes), the window keep-out is a soft penalty in the OCP, and a yaw-rate cost cleans the attitude. The rack was moved to x=-4 so the drone exits the window before placing. The carried box is an oriented box (8 corners) so it clears the frame.
- 2026-06-14: Keyframe-guided OCP (the main result). A single human keyframe guides the OCP to a pitch-forward window passage that is the only collision-free trajectory (0 of 149 knots) with the best clearance on every body part, versus the unguided yaw-sideways squeeze that grazes the frame (2 of 149). gRITE dynamic tracking with the friction-gripped box is validated through the window (about 1.9 cm position error at the 53.5 deg peak tilt). See `aerial_box_transport/docs/window_keyframe_results.md`.
- Current state: planner, window passage, and keyframe-guided study complete and pushed (jaewoo branch). The hardware experiment is the remaining long-term item. NOTE: sections 1 to 5 below are the ORIGINAL desk-to-rack design plan; the window and keyframe work above is a research extension beyond the locked M1 to M6 scope.

## 1. Architecture recap (from the spec)
- Offline whole-body multi-phase OCP produces a reference, tracked online. Architecture 1, not MPC (D1).
- Smooth contact (strategy 3): normal force is a smooth function of penetration, derived not decided (D2, D3).
- Box is a separate free rigid body, coupled to the robot only through the contact force (D6).
- In the OCP the base is a fully actuated 6-DoF wrench. Thrust and tilt allocation is delegated to a robust low-level controller, gRITE or DOB (D5).
- The slip-aware squeeze-force law lives outside the OCP and feeds a setpoint into the cost (D7).

## 2. Reuse strategy (from the vlm_drone_ws assessment)
Bottom line: reuse the gRITE/DOB low-level controller (D5) and the Isaac reference-injection pattern (M6) by adaptation. Build M1, M4, and all contact modules (M2, M3, M5) fresh.

| Need | Verdict | Source | Adaptation |
|---|---|---|---|
| M1 Pinocchio/cpin model | BUILD FRESH | none usable (their kinematics is bespoke single-arm, no Pinocchio) | write from the USD-derived model |
| M4 multi-phase OCP | BUILD FRESH (reuse CasADi/IPOPT plumbing only) | reference_generator/scripts/whole_body_planner.py (nlpsol setup, smooth SO(3) Rodrigues idiom) | their planner is single-step velocity IK with no dynamics, phases, or contact |
| D5 low-level controller | ADAPT | tiltrotor_ros2_controller/tiltrotor_coax_grite_controller (and sibling tiltrotor_coax_dob_controller) | re-derive allocation for our rotor geometry, extend gravity and inertia compensation to two arms plus box mass, retune gains |
| M6 Isaac runner | EXTEND existing harness | this repo's main.py direct-articulation harness; vlm_drone_ws isaac_bridge/coordinate.py for NED to NWU helpers if needed | replay the planned reference, add the box, log; do not adopt their Isaac Lab env.step paradigm |
| Planner env recipe | BUILD FRESH | no Pinocchio present, no pinned env to copy | conda-forge pinocchio (with casadi bindings) + casadi + ipopt |

Interface facts that shape our design:
- The gRITE controller consumes a 30-dim SE(3) reference on `/reference_trajectory`, laid out as pos(3), vel(3), acc(3), jerk(3), R column-major(9), omega(3), omega-dot(3), omega-ddot(3). DECISION: our OCP reference output adopts this exact 30-dim contract, so it can feed gRITE on hardware and in high-fidelity sim with no translation.
- gRITE already treats the base as a fully actuated 6-DoF wrench and decouples arm commands via `/q_arm_des` and `/q_arm`. This matches D5 directly.
- No contact, box, friction, or dual-arm logic exists in vlm_drone_ws, so M2, M3, M5 are fresh.
- No Pinocchio or cpin anywhere, so there is no known-good cpin stack to inherit. Build am_dualarm fresh and verify cpin on creation.

## 3. Two-track build plan
The planner track is Isaac-free, self-verified with pytest, and does not wait on the teammate. The Isaac track joins when the scene arrives.

### Track A: Planner (Isaac-free, I self-verify)
- Step 0, Foundation: create `am_dualarm` (conda-forge pinocchio with cpin, casadi, ipopt, numpy, scipy, matplotlib, pytest, pyyaml), verify cpin import and an IPOPT solve. Scaffold `aerial_box_transport/` per the spec layout, with config/robot.yaml and config/task.yaml stubs.
- Step 1, MVP (the headline result): reduced box model (vertical channel) plus M2 smooth contact plus M3 box and friction cone. Then M5 slip-aware law plus the fixed-force baseline. Then a simple transport trajectory with a prescribed a_z. Then the acceleration sweep and the headline plots (F_n vs a_z, slip rate vs a_z). All pure math, fully unit-testable.
- Step 3, Full whole-body: extract the Pinocchio model from `dual_arm_final.usd` (read inertia and joint axes from the stage; a research subagent maps the USD joint tree and EE frames first). Build M1 (cpin dynamics, FK, contact Jacobians) and M4 (multi-phase OCP, multiple shooting or collocation with IPOPT, a phase FSM over approach, grasp, transport, release). The OCP emits the 30-dim reference contract. Then the decoupled baseline and ablations.

### Track B: Isaac validation (gated on the teammate scene)
- Step 2, M6: contact-model identification (slow press, fit k and eps), virtual wrist force sensor readout, and a reference-replay runner built on this repo's existing harness. First pass commands base wrench plus arm joints directly, not gRITE-in-sim. Headline sim scenario and metrics.
- gRITE-in-the-loop and hardware-style validation is a later refinement using the adapted controller. The 30-dim reference contract keeps that path open from the start.
- Step 4, stretch: execution-time admittance plus Jacobian IK force feedback (D8), still Architecture 1.

## 4. Agent and debugging workflow
- Sequential, main-loop driven, MVP first. No parallel module-builder agents (they drift on sign and frame conventions).
- Subagents only for scoped read-only research (USD to Pinocchio extraction, Isaac 5.1 sensor API) and adversarial math verification after each module.
- A Workflow only for the embarrassingly parallel a_z sweep and ablations, and only on explicit opt-in.
- Plan and sim are decoupled by a saved reference file (results/*.npz) that the sim runner replays.
- Every sim script gets --headless and --max_steps or --max_time so headless smoke tests run to completion (I run these), while the GUI run is for visual confirmation (the user runs these). A new launcher alias like ambox mirrors amdrone.
- Each run logs its config and metrics to results/ for reproducibility.

## 5. Open items
- Teammate scene: box with explicit size, mass, and inertia; deformable pads at both EEs; a wrist joint to read force and torque. Box mass, inertia, and mu must match config/task.yaml.
- USD to Pinocchio extraction for M1 (no URDF exists, so read from the stage).
- Box rotation: linear channel for the MVP, full 6-DoF later.

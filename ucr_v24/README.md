# UCR 2026 — Jetson Nano Autonomous Car

Autonomous road-following on the physical round-3 car. Perception is a
TensorRT road-segmentation model; control is a port of the proven
`ucr_main_v20.py` PID. The control loop ticks at a fixed 10 Hz so the
sim-tuned per-step constants keep their meaning on the real robot.

---

## Current status

**Running binary:** `main` (a.k.a. `ucrv5` — same source file, renamed at
the CMake target). Lane-following only: `LaneFollowerV20` + `RoadSegmenter`.

The sign detector, sign voter, skeleton topology, and turn state machine
exist in the tree but are **not wired into the current `main.cpp`**. They
were integrated in an earlier build (Log 13) and reverted to isolate a
turn-arming problem: on the round-3 map the area-spike gate never fired,
because the road-segment model does not mask far enough ahead to expose
the junction widening. Pending re-integration on a gate that works.

| Component | File | Status |
|---|---|---|
| Hardware bring-up (camera, motor, servo, IMU, ToF, OLED) | `hardware/*` | Running |
| TensorRT session (FP16 + FP32 I/O) | `perception/TrtSession.h` | Running |
| Road segmentation (TwinLiteNet) | `perception/RoadSegmenter.h` | Running |
| v20 PID + occlusion gate + asymmetry | `control/LaneFollowerV20.h` | Running |
| Sign detection (YOLO 8-class head) | `perception/SignDetector.h` | Standalone-verified, not wired |
| Sign voter (tick-indexed) | `control/SignVoter.h` | Not wired |
| Skeleton topology (Zhang-Suen + TopoLatch) | `perception/SkeletonTopology.h` | Not wired |
| Turn state machine (area-spike gate) | `control/TurnStateMachine.h` | Tested, never fired — not wired |
| v21 opening detector + Bezier path (reference) | `control/LaneFollowerV21.h` | **NOT COMPILED** — algorithm reference only |

---

## Architecture (current running build)

```
Camera (BGR 640×384)
│
▼
RoadSegmenter          ← TensorRT engine + BGR→RGB + argmax
│  (CV_8UC1 mask at model output resolution, currently 320×184)
▼
LaneFollowerV20        ← keep_largest → centerline → near/far/curve
│                     → occlusion gate → asymmetry → PID → speed
│  (v20::LaneFollower::Output)
▼
main loop              ← fixed 10 Hz tick, soft-start ramp
│                     → real-angle → servo calibration → servo cmd
▼
Motor / Servo
```

The control loop ticks at a **fixed period**, independent of inference
rate. Every rate limit and EMA inside `LaneFollowerV20` is *per step*, so
the tick rate defines their real-world meaning. See "Fixed-rate tick".

---

## File map

```
config/
  RobotConfig.h           ← all tunable constants (see "Configuration guide")
perception/
  TrtSession.h            ← generic TensorRT engine wrapper (FP16/FP32 I/O)
  RoadSegmenter.h         ← TwinLiteNet binary segmentation (2-ch argmax)
  SignDetector.h          ← YOLO 8-class detection head (not wired)
  SkeletonTopology.h      ← Zhang-Suen thinning + L/S/R arms (not wired)
control/
  LaneFollowerV20.h       ← the v20 PID stack (header-only, cohesive, running)
  SignVoter.h             ← tick-indexed sign vote (not wired)
  TurnStateMachine.h      ← force-turn SM + area-spike gate (not wired)
  LaneFollowerV21.h       ← v21 opening + Bezier reference (NOT COMPILED)
main.cpp                  ← hardware init + control loop + UI (v20-only)
```

**Rule of thumb:** if a piece of code touches only the TensorRT API, it
belongs in `TrtSession.h`. If it also touches the model's tensor layout
(BGR→RGB, argmax, NMS, class decode), it belongs in a model wrapper
(`RoadSegmenter.h`, `SignDetector.h`).

---

## Build

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
sudo ./main
```

The CMake target is currently `main`. In Log 13 the same source was built
as `ucrv5`; either name works — they refer to the same file.

No new libraries are linked beyond what `thisinh` already required:
`nvinfer`, `cudart`, OpenCV, pthreads.

Expected boot:

```
=== UCR2026: Auto Run ===
Engine: /home/car4/Downloads/road/kga2.engine
Control period: 100 ms
[RoadSegmenter] in 320x184 out 320x184
```

Note: the engine is **320×184**, not 320×180. The model was exported at
320×184, so `near_error` / `far_error` are computed on a slightly
vertically-distorted mask relative to a clean 5:3 crop. This has been
accepted (Log 13 P6); fix at engine-export time if it ever biases
straight-line behaviour.

If you see:

- `[TrtSession] Init error: ...` → engine file missing or built against a
  different TensorRT than the one on the Jetson.
- `[RoadSegmenter] Expected 2 output channels` → the engine is not a
  TwinLiteNet binary-segmentation export.
- `[RoadSegmenter] Dynamic dim ... not supported` → static-shape export
  required; `TrtSession` does not handle dynamic bindings.

Press `q` or `Esc` to stop.

---

## Fixed-rate tick

`main.cpp` calls `LaneFollower::step()` **only** when
`steady_clock::now() >= next_control`. Inference runs as fast as the
hardware allows in between.

- `rate_limit = 10` means **10° per step**, not 10° per frame. At 10 Hz
  that is 100°/s — the value v20 was tuned at.
- Same for `derivative_alpha`, `d_far_ema_alpha`, `cov_ema_alpha`,
  `asym_ema`, and the speed EMA.

If you change `CONTROL_PERIOD_MS`, every per-step constant in
`LaneFollowerV20.h` changes meaning. Recompute alphas with
`alpha = 1 - exp(-dt / tau)`, and rescale the rate limits proportionally.

---

## Servo calibration

The v20 PID emits a steering angle in **PID convention** (positive =
right turn), in degrees, after `post_pid_gain`. `main.cpp`:

1. Negates to **real-angle convention** (positive = left turn).
2. Clamps to `±MAX_REAL_ANGLE_DEG` (30°).
3. Maps through `RealAngleToServo()` → **raw servo command** (positive =
   left).
4. Clamps to `±MAX_SERVO_CMD` (55) — a second safety independent of the
   real-angle bound.
5. Rounds and passes to `Servo::SetAngle()`.

The `RealAngleToServo` / `ServoToRealAngle` pair encodes the empirical
servo asymmetry from Log 12:

```
servo_cmd > 0 (left):   real =  0.96 * servo_cmd − 6.2
servo_cmd < 0 (right):  real =  0.71 * servo_cmd + 2.5
```

Left and right are ~35% different in gain, and the mechanical deadband is
asymmetric (~6.5° servo on the left, ~3.5° on the right). Treat the
`±MAX_REAL_ANGLE_DEG` bound as **the** meaningful steering limit; the
`±MAX_SERVO_CMD` is a belt-and-braces clamp.

**Known latent issues** (from Log 12 E52/E53; not fixed, documented here):

- **E52 — `post_pid_gain` (2.2) exceeds the real-angle saturation point.**
  PID output `13.6` already saturates `MAX_REAL_ANGLE_DEG = 30`; the upper
  ~38% of the PID's numeric range maps to a flat 30°. Either lower
  `POST_PID_GAIN` to ~1.36 or raise `MAX_REAL_ANGLE_DEG` to ~40 — decide
  empirically on-track. Current build leaves 2.2 in place.
- **E53 — servo zero is not the mechanical centre.** The two fitted lines
  cross real-zero at different servo commands. The code enforces
  `real=0 ⟺ servo=0` by convention, producing a small piecewise
  discontinuity at centre. May show as mild left/right drift near
  straight-line; verify at low speed.

---

## Sign classes (reserved for integration)

The 8-class engine emits, in order:

| id | name | meaning |
|---|---|---|
| 0 | construction_ahead | road-work warning |
| 1 | no_left | turn right required (or straight if 4-way) |
| 2 | no_right | turn left required (or straight if 4-way) |
| 3 | parking | enter parking state |
| 4 | roundabout | roundabout handling |
| 5 | traffic_light | stop on red (future) |
| 6 | turn_left | turn left required |
| 7 | turn_right | turn right required |

Precedence when several compete: **explicit turn > no-turn > default
straight**. `SignDetector.h` NMS is class-agnostic — if `no_left` and
`turn_left` co-occur on the same sign post, only the top scorer survives.
This is acceptable for now; upgrade to class-aware NMS if the class list
ever puts conflicting signs on one post.

`SIGN_DETECT_EVERY = 2` is the current setting. 4 was tested with no
observed difference at ~6 Hz loop rate; 2 is kept.

---

## Turn phases (reserved for integration)

`TurnStateMachine` uses these phases, driven by measured BNO055 heading
delta from force-start (no hardcoded timers):

| phase | condition | target speed |
|---|---|---|
| ENTRY | `|Δheading| < FORCE_TURN_PHASE_DEG (15°)` | `FORCE_APPROACH_SPEED (8)` |
| TURNING | `15° ≤ |Δheading| < 50°` | `FORCE_TURNING_SPEED (5)` |
| EXIT | `|Δheading| ≥ 50°` | `FORCE_EXIT_SPEED (12)` |
| reset | `|Δheading| ≥ 56°` | voter + force reset |

The gate that arms this is the **area-spike** detector — mask area must
exceed a slow EMA baseline by `TURN_AREA_RATIO` for `TURN_GATE_PERSIST`
consecutive ticks. On the round-3 map this **never fired**; re-integration
is pending a gate that works on a mask that does not see far ahead.

---

## Heading source (reserved for integration)

`GetStatus`-equivalent on the car is `BNO055::eulerDeg.x` (fused, drifts,
wraps 0/360). `BNO055::relativeYawDeg()` (gyro-integrated, bias-corrected)
is the alternative but is **currently always 0** — `integrateYawRate(dt)`
is never called from `SensorThread` (E38 from the Round-3 handoff). Any
turn-completion logic that depends on gyro-integrated yaw needs E38 fixed
first.

`HEADING_SIGN` / `HEADING_OFFSET_DEG` in `RobotConfig.h` are the transform
into the `TurnStateMachine`'s expected convention (positive = right).

---

## Verification checklist

Before any timed run:

- [ ] `sudo ./main` boots, prints `[RoadSegmenter] in 320x184 out 320x184`
- [ ] Camera frame shows in UI; green mask overlays the road correctly
- [ ] Red centerline follows the road centre visually
- [ ] `kf` in UI ≈ 1.00 on open road; drops visibly at junctions/obstacles
- [ ] `INV` flag off when mask is healthy; on when the mask is lost
- [ ] Motor at 0 → car does not move; `MIN_SPEED_CMS` → smooth creep
- [ ] Servo at `RealAngleToServo(+10)` → car turns right (verify empirically)
- [ ] Servo at `RealAngleToServo(−10)` → car turns left
- [ ] Emergency stop test: Ctrl-C → motor stops, servo centres
- [ ] First-tick motor speed is `min(speed, MAX_ACCEL_CMS2 × 0.1)` not a
      small fractional value like `0.615` — see "P5" below.

---

## Configuration guide

`config/RobotConfig.h` is split into two sections:

- **Section A — active** — read by the current `main.cpp`.
- **Section B — reserved** — read by nothing currently; will be wired into
  `SignDetector` / `SignVoter` / `TurnStateMachine` on re-integration.

Tune **only** Section A values for the current binary. Section B values
can be edited freely; they have no effect until integration.

### Section A — the values you'll actually touch

| Constant | Meaning | Safe range |
|---|---|---|
| `MAX_SPEED_CMS` | Top forward speed | 5 → 25 |
| `MIN_SPEED_CMS` | Floor speed in curves / occl | 2 → 12 |
| `MAX_REAL_ANGLE_DEG` | Real-angle steering bound | 20 → 40 |
| `MAX_SERVO_CMD` | Raw servo clamp | 40 → 60 |
| `CONTROL_PERIOD_MS` | PID tick period | leave at 100 unless re-tuning |
| `MAX_ACCEL_CMS2` | Speed ramp on start | 10 → 40 |
| `POST_PID_GAIN` | PID-to-real-angle scale | calibrate empirically |
| `CURVE_SCALE` | Curve slowdown aggressiveness | 80 (more) → 200 (less) |
| `MIN_ROWS`, `COVERAGE_THRESHOLD` | Mask validity gate | tighten if garbage masks slip |

### Section B — reserved

All `SIGN_*`, `TOPO_*`, `TURN_*`, `FORCE_*`, `SLOW_SPEED_CMS`,
`HEADING_*`, and `sign_class::*` constants. See "Turn phases" and "Sign
classes" above for meaning.

### Removed (dead, do not re-add)

These were failed gate iterations from Log 13 and are deleted from
`RobotConfig.h` and `TurnStateMachine::TurnConfig`:

- `TURN_CONDITION_Y_LO_RATIO`, `TURN_CONDITION_COVERAGE` — bottom-band coverage gate
- `TURN_OPENING_MARGIN`, `TURN_NEAR_MIN_COVERAGE` — far-band vs near-band gate
- `OPEN_THRESH`, `OPEN_MIN_HEIGHT`, `OPEN_SMOOTH_WIN`, `OPEN_BASE_WIN`, `TURN_OPENING_PROXIMITY` — V21 opening constants (existed only in the never-compiled draft; port them into `TurnConfig` if/when you swap the gate)

---

## What NOT to do

- **Do not** tick the PID at inference rate. Use the fixed-period gate.
- **Do not** reset `LaneFollower` on a lost-mask frame. Only a TensorRT
  *failure* (`inferOK == false`) triggers a hard reset; a lost mask just
  yields `valid == false` and holds `last_angle`.
- **Do not** put model-specific code in `TrtSession.h`.
- **Do not** put hardware logic in `RoadSegmenter` / `LaneFollower` — those
  are pure.
- **Do not** split `LaneFollowerV20.h`. It is cohesive; the near/far/
  curve/occl/asym signals feed the PID and only exist in that context.
- **Do not** `#include` `control/LaneFollowerV21.h` in any build target. It
  is reference-only and will not compile (see its header comment).
- **Do not** change `CONTROL_PERIOD_MS` without retuning every per-step
  rate limit and EMA in `LaneFollowerV20.h`.

---

## Open issues / next steps

1. **P5 — first-tick speed.** Boot log shows `[Driver -> MCU]: N1 v0.615385`
  on the first actuation. With `MAX_ACCEL_CMS2 = 20`, `CONTROL_PERIOD_MS =
  100`, the expected first applied speed from 0 is `2.0` cm/s. `0.615385
  = 200 / 325`, which is `Motor::SetSpeedCms` cm/s → wheel-rad/s with
  3.25 cm radius. Inspect `Motor::SetSpeedCms` and the first-tick call site
  in `main.cpp` to find where the division is happening (either `speed` is
  being set to `0.615` pre-motor, or `Motor` converts via a different
  factor).
2. **Turn gate.** The area-spike gate does not fire on the round-3 map.
  Candidate next gates: (a) `LaneFollowerV21`'s rolling-median opening
  detector, ported into `TurnStateMachine::CheckTurnCondition`; (b) a
  sign-bbox-based gate using `SignDetector`'s bbox bottom edge as a
  proximity proxy; (c) fusion of the two. Decided in a future session.
3. **E38 — `relativeYawDeg()` always 0.** `integrateYawRate(dt)` is never
  called from `SensorThread`. Required for turn completion if you choose
  the gyro-integrated heading source. Otherwise use `eulerDeg.x` and
  accept drift.
4. **Sign/turn integration.** Re-introduce `SignDetector` +
  `SignVoter` + `TurnStateMachine` into a new `main.cpp` **only after**
  the gate is redesigned. The current reverted state is stable — do not
  re-add the area-spike gate as-is.
5. **E52/E53 servo calibration.** Decide `POST_PID_GAIN` vs
  `MAX_REAL_ANGLE_DEG` alignment (E52), and validate E53 near
  straight-line at low speed.

---

## Logs and debugging

The only visibility currently is the on-screen overlay (`SHOW_UI = true`):

```
FPS <fps> | spd <applied>/<target> | PID <pre-gain> | out <post-clamp> | kf <kept_frac> [INV]
```

- `spd <applied>/<target>` — soft-start ramped value vs. follower's target
- `PID <pre-gain>` — raw PID output before `post_pid_gain`
- `out <post-clamp>` — final rounded servo command
- `kf <kept_frac>` — largest-component fraction; drops at junctions/obstacles
- `INV` — mask is invalid but coverage still > 2 %

When you add a CSV logger, the columns to capture:
`t_usec, valid, coverage, kept_frac, near_e, far_e, eff, curve_smooth,
asym, occl, angle_pre_gain, angle, servo_cmd, applied_speed, tof[4], imu_yaw`.

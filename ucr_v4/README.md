# UCR 2026 — Jetson Nano Autonomous Car

Autonomous road-following + sign-triggered turns on the physical round-3 car.
Road perception is a TwinLiteNet TensorRT engine; the sign detector is a
YOLO-family TensorRT engine; heading is read from a BNO055 over I²C. The
driving stack is a port of the proven `ucr_main_v20.py` PID **plus** the
v28 turn state machine. Everything runs on a fixed 10 Hz control tick so
sim-tuned constants keep their meaning on the real robot.

---

## Architecture at a glance

```
                    Camera (BGR 640×384)                 BNO055 (I²C)
                            │                                 │
                            ├──────────────┐                  │
                            ▼              ▼                  │
                    RoadSegmenter    SignDetector             │
                    (TwinLiteNet,    (YOLO 416×416,           │
                     TensorRT)        TensorRT)               │
                            │              │                  │
                            ▼              ▼                  │
                    CV_8UC1 mask     SignDetection[]          │
                            │              │                  │
                            ▼              ▼                  │
                    LaneFollowerV20  SignVoter        eulerDeg.x
                    (v20 PID:         (4-frame          (fused yaw
                     near/far/curve/   persistence,     0..360,
                     occl/asym/speed)  blacklist)        °/s rate)
                            │              │                  │
                            ▼              ▼                  ▼
                         (angle,       committed sign      heading
                          speed,             │                │
                          valid)             │                │
                            └──────────┬─────┴────────────────┘
                                       ▼
                                TurnStateMachine
                                NORMAL ⇄ APPROACH → TURNING → NORMAL
                                       │
                                       ▼
                            main-loop dispatcher
                            (TURNING → SM owns actuators,
                             else → PID owns actuators)
                                       │
                                       ▼
                              real-angle → servo → motor
```

The control loop ticks at a **fixed period**. Inference (road, sign) runs
between ticks at whatever speed the hardware allows. This is intentional:
every rate limit and EMA inside `LaneFollowerV20` and `TurnStateMachine` is
*per step*, so the tick rate defines their real behaviour.

---

## File map

```
config/
  RobotConfig.h            ← every tunable constant (one place)
perception/
  TrtSession.h             ← generic TensorRT engine wrapper (no model logic)
  RoadSegmenter.h          ← TwinLiteNet road segmentation (uses TrtSession)
  SignDetector.h / .cpp    ← 8-class sign detector (uses TrtSession)
control/
  LaneFollowerV20.h        ← the v20 PID stack (header-only, cohesive)
  SignVoter.h / .cpp       ← persistence + blacklist over SignDetections
  TurnStateMachine.h / .cpp← NORMAL/APPROACH/TURNING, heading-measured exit
sensor/
  BNO055.h / .cpp          ← I²C IMU driver (already existed)
hardware/
  Camera / Motor / Servo   ← existing drivers (unchanged)
ucrv4.cpp                  ← main: hardware init + control loop + dispatcher + UI
```

### What goes where

| Concern | File |
|---|---|
| Engine load / buffer / stream | `TrtSession.h` |
| BGR→RGB, argmax, confidence margin (road) | `RoadSegmenter.h` |
| BGR→RGB, letterbox, NMS, class decode (sign) | `SignDetector.cpp` |
| Near/far split, occlusion, PID, speed | `LaneFollowerV20.h` |
| Sign voting, blacklist, commit aging | `SignVoter.cpp` |
| Turn trigger, taper, heading completion | `TurnStateMachine.cpp` |
| I²C IMU reads, fused yaw | `BNO055.cpp` (unchanged) |
| Constants you actually tune | `RobotConfig.h` |
| Wiring, tick scheduling, dispatcher, UI | `ucrv4.cpp` |

**Rule of thumb:** TensorRT API only → `TrtSession.h`. Model tensor layout
(argmax, NMS, letterbox, class decode) → model wrapper
(`RoadSegmenter.h`, `SignDetector.cpp`). Actuator arbitration → `ucrv4.cpp`.

---

## Build

Unchanged from the `thisinh` scaffold.

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
sudo ./main
```

No new libraries. `nvinfer`, `cudart`, OpenCV, pthreads, `libi2c` are all
already required.

---

## Running

```bash
sudo ./main
```

Expected boot log:

```
=== UCR2026: Auto Run (turn-enabled=yes) ===
Road engine : /home/car4/Downloads/road/kga2.engine
Sign engine : /home/car4/Downloads/sign/huy_sign.engine
Control     : 100 ms
[SignDetector] Output dims: [12, 3549]          ← layout auto-detect
[SignDetector] Layout: CHANNELS_FIRST, nc=8 A=3549
[SignDetector] Ready: engine=... input=416x416 nc=8 anchors=3549
[main] mask resolution: 320x180  (TURN_BASELINE_FLOOR was tuned for 320x192)
```

If `turn-enabled=NO`, the BNO055 or sign engine failed to init. The car
will still follow the road (PID only) but will never turn on a sign.

Failure modes and what they mean:

- `[TrtSession] Init error: ...` → engine file missing or built for a
  different TensorRT version. Re-export against the Jetson's TensorRT.
- `[SignDetector] Cannot determine output layout` → the engine has an
  unexpected output shape. Check the source `.onnx` in Netron; the layout
  auto-detect handles `[4+nc, A]`, `[A, 4+nc]`, and `[N, 6]` (NMS-baked).
- `[SignDetector] Input elem count mismatch` → `SIGN_INPUT_SIZE` in
  `RobotConfig.h` doesn't match the engine. Fix one or the other.
- `[main] mask resolution:` not ~320×192 → `TURN_BASELINE_FLOOR` and
  `TURN_PEAK_DROP_PX` need rescaling (see "Turn state machine" below).
- `BNO055 init failed: ...` → heading unavailable, turn SM disabled. Check
  `/dev/i2c-1`, address 0x28/0x29, and the calibration file path (E43 is
  still a hardcoded `/home/car4/...` path in `BNO055.h`).

Press `q` or `Esc` to stop.

---

## The fixed-rate control tick

`ucrv4.cpp` calls `LaneFollower::step()` and `TurnStateMachine::Update()`
**only** when `steady_clock::now() >= next_control`. Inference runs as fast
as the hardware allows in between.

Why this matters:

- `rate_limit = 10` in the PID means **10° per step**, not 10° per frame.
  At 10 Hz that is 100°/s, which is what the sim was tuned at. If you tick
  the PID every inference frame and inference runs at 30 Hz, the effective
  steering rate triples.
- Same for `derivative_alpha`, `d_far_ema_alpha`, `cov_ema_alpha`,
  `asym_ema`, and the speed EMA.
- The turn state machine's `TURN_BASELINE_ALPHA = 0.97` EMA assumes 10 Hz
  input on `px_mid`. Ticking at a different rate changes the effective
  time constant.

If you change `CONTROL_PERIOD_MS`, **every per-step constant in
`LaneFollowerV20.h` and `TurnStateMachine.h` changes meaning**. Treat it
as a lockstep value: at 20 Hz, halve the rate limits and re-tune all alphas
with `alpha = 1 - exp(-dt / tau)`.

---

## Sign detection

**Model.** YOLOv8/v11-style detection head, 8 classes, letterboxed to
416×416, exported to TensorRT. Class order in `SignDetector.h::SignClass`
**must** match the training `names` dict — reordering breaks every decision
below silently.

| Index | Class | Trigger? |
|---|---|---|
| 0 | `construction_ahead` | no (not implemented — new-sign work) |
| 1 | `no_left` | no (blocked + voter reset) |
| 2 | `no_right` | no (blocked + voter reset) |
| 3 | `parking` | no (future `ParkingStateMachine`) |
| 4 | `roundabout` | no (not implemented) |
| 5 | `traffic_light` | no (not implemented) |
| 6 | `turn_left` | **yes** → APPROACH with dir = −1 |
| 7 | `turn_right` | **yes** → APPROACH with dir = +1 |

The three that would trigger in a full Round-3 implementation
(`construction_ahead`, `roundabout`, `traffic_light`) and the two
`no_left`/`no_right` T-junction precedence rules are deferred. Right now
they just get blocked by `SignVoter` after one commitment, exactly like in
v28 — the car keeps road-following.

**Frequency.** `SIGN_DETECT_EVERY = 2` → inference every other tick, i.e.
**5 Hz at the 10 Hz control rate**. Sign signs are visible for > 1 s at
competition speed, so a 4-detection commit (`SIGN_VOTER_MIN_FRAMES = 4`,
800 ms) still fires reliably. If you find fast-moving signs slipping
through, drop `SIGN_VOTER_MIN_FRAMES` to 3 — do **not** drop to 2.

**Output-layout auto-detect.** `SignDetector::DetectLayout()` reads the
output binding dims once at `Init()` and picks one of:

| Shape | Layout | Notes |
|---|---|---|
| `[12, A]` (channels-first, `A > 20`) | `CHANNELS_FIRST` | Ultralytics raw export |
| `[A, 12]` (`A > 20`) | `ANCHORS_FIRST` | rare |
| `[N, 6]` or `[6, N]` | `NMS_BAKED` | Ultralytics `nms=True` export |

If you see `[SignDetector] Layout: CHANNELS_FIRST, nc=4 ...` something is
wrong with the engine — 8 classes means `nc=8`, so `d1 = 12` channels-first
or `d2 = 12` anchors-first.

---

## Sign voter

`SignVoter` is an exact port of v28's `SignVoter` class. It provides two
things the state machine relies on:

1. **Persistence.** A class is committed only after `SIGN_VOTER_MIN_FRAMES`
   consecutive detections (with `SIGN_VOTER_MAX_GAP` tolerance). One bad
   detection frame does not arm a turn.
2. **Blacklist.** After a turn completes (or a false positive bails), the
   sign that caused it is blacklisted for `SIGN_VOTER_BLACKLIST_FRAMES`
   detection frames. Without this the same sign re-arms the state machine
   the moment it exits `TURNING`.

Committed status is read via `sign_voter.GetCommitted()` (−1 = none) and
fed to `TurnStateMachine::Update()` as `sign_committed`.

---

## Turn state machine

Ported from v28.py. Timer-free.

```
NORMAL ──(sign_committed = turn_left / turn_right)──► APPROACH
APPROACH ──(px_mid_peak - px_mid ≥ PEAK_DROP_PX)──► TURNING
APPROACH ──(px_mid back near baseline, peak never crossed)──► NORMAL (bail)
TURNING ──( |heading - entry_heading| ≥ TARGET_YAW )──► NORMAL
```

**Why peak-drop?** A turn into a wide junction makes `px_mid` peak and fall
in one or two frames. An absolute-threshold trigger (v23–v26) fires early
on those; a first-fall-from-peak trigger (v27) fires on single-frame noise
peaks. v28 requires the peak to fall by `PEAK_DROP_PX` before firing.

**Why heading?** Topology (`straight`/`left`/`right`) never drops to zero
mid-junction, and doesn't change at 10 km/h — it cannot tell you when a
turn is complete. The BNO055's `eulerDeg.x` (or a properly-integrated
`relativeYawDeg()`) does. `TURN_TARGET_YAW = 72°` = 0.8 × 90°.

**Angle convention.** `TurnStateMachine::Output::angle_deg` uses the **same
sign as the PID output** (positive = right, matching sim `AVControl`).
`ucrv4.cpp` negates it to real-left-positive before `RealAngleToServo()`,
exactly as it does for the PID. Do **not** apply `POST_PID_GAIN` to turn
angles — they are already real-degrees in the `MAX_REAL_ANGLE_DEG = 30`
bound.

**`TURN_BASELINE_FLOOR` is resolution-specific.** v28 tuned `2280.0` for
YOLO segmentation output at ~320×192. The Round-3 road model is TwinLiteNet.
If `ucrv4.cpp` reports `mask resolution: 320x180`, close enough (adjust
`TURN_BASELINE_FLOOR` by hand to match `px_mid`'s median on a straight
segment). If it reports `640x384`, `px_mid` will be ~4× larger — either
rescale `TURN_BASELINE_FLOOR` to ≈ 9000 and `TURN_PEAK_DROP_PX` to ≈ 800,
or normalise `MaskMidPixels()` in `ucrv4.cpp` by
`(mask.cols * mask.rows) / (320*192)`.

---

## Heading source (BNO055)

The turn state machine consumes a single scalar `heading` per tick. It is
produced in `ucrv4.cpp` from `BNO055::Data::eulerDeg.x` — the chip's fused
yaw (accel + gyro fusion, IMU mode `0x08`, no magnetometer, slow drift).

```cpp
heading = WrapDeg360( HEADING_SIGN * eulerDeg.x + HEADING_OFFSET_DEG );
```

- **`HEADING_SIGN`** defaults to `+1.0f`. The BNO055's native yaw increases
  counter-clockwise viewed from above. If `test_bno055` shows yaw *increasing*
  on a clockwise bench turn, leave it at `+1`. If it *decreases*, set
  `-1.0f`. **Verify on the bench before driving** — a flipped sign turns
  left when the sign says right.
- **`eulerDeg.x` vs `relativeYawDeg()`.** `relativeYawDeg()` requires
  `integrateYawRate(dt)` to be called after every `read()` — E38, not yet
  wired into `SensorThread`. The current code uses `eulerDeg.x` because it
  works today without an E38 fix. `ucrv4.cpp` also calls
  `integrateYawRate(dt)` opportunistically each tick so that if you switch
  `HeadingProvider::Source` later, `relativeYawDeg()` is already alive.

**Drift.** In IMU mode (no magnetometer) yaw drifts ~1–3°/min. For a single
turn of < 5 s this is negligible. For multi-turn navigation over a 60 s lap,
use `relativeYawDeg()` (once E38 is fixed) — it has bias correction on top
of the raw gyro and drifts less.

---

## Adding a new perception model

Copy `SignDetector.{h,cpp}` as the template (it's the more complete one —
shows letterbox, NMS, layout auto-detect). Structure:

```cpp
class NewDetector {
    TrtSession session_;
    std::vector<float> inBuf_, outBuf_;
    // model-specific dims, class table
public:
    bool Init(const std::string& enginePath);
    bool Detect(const cv::Mat& bgr, /* ... */);
};
```

All model-specific decode (argmax, NMS, class order) stays inside
`NewDetector`. Do **not** put it in `TrtSession` (breaks the abstraction
for the next model) and do **not** put it in `ucrv4.cpp` (it will get
copied and diverge).

---

## Adding a new controller / mission state

Current pattern: `ucrv4.cpp` is a dispatcher. It asks `TurnStateMachine`
"do you own the actuators this tick?" via `turn_out.in_turn_ctrl`. If yes,
turn owns; if no, PID owns. Turn speed caps are applied even when the PID
owns steering (`out.speed_cms` is `min`-combined with the PID's speed).

For the next controller (parking), add a new class with the same interface
shape:

```cpp
struct ParkOutput {
    float angle_deg   = NaN;   // PID convention (pos = right)
    float speed_cms   = NaN;   // NaN = no cap
    bool  in_park_ctrl = false;
};
```

and extend the dispatcher in `ucrv4.cpp` with a priority order
(`PARK > TURN > PID` is likely correct — parking should preempt a turn
if both somehow fire). **Do not** let a new controller mutate
`LaneFollowerV20` internals. Feed the PID output back into the dispatcher
and let it decide.

---

## Configuration guide

All system constants live in `config/RobotConfig.h`. Algorithm tuning
(`kp`, `kd`, occlusion thresholds, asymmetry caps) stays in
`control/LaneFollowerV20.h`; turn-specific ratios stay in
`control/TurnStateMachine.h`'s constructor defaults unless overridden via
`RobotConfig.h` macros. The split exists so you can change the **system**
without touching **algorithms**.

### The values you will actually touch

| Constant | Meaning | Safe range |
|---|---|---|
| `MAX_SPEED_CMS` | Top forward speed | 5 → 25 |
| `MIN_SPEED_CMS` | Floor speed during curves / occl | 2 → 8 |
| `MAX_REAL_ANGLE_DEG` | Real steering clamp (servo-independent) | 20 → 35 |
| `MAX_SERVO_CMD` | Raw servo clamp (independent of real angle) | 40 → 60 |
| `HEADING_SIGN` | +1 / −1 — must be bench-verified | ±1 |
| `CONTROL_PERIOD_MS` | PID tick period | leave at 100 |
| `MAX_ACCEL_CMS2` | Speed ramp on start | 10 → 40 |
| `CURVE_SCALE` | Curve slowdown aggressiveness | 80 → 200 |
| `SIGN_DETECT_EVERY` | Detection decimation | 1 → 3 (≥ 3 needs voter tune) |
| `SIGN_VOTER_MIN_FRAMES` | Commit after N detections | 3 → 5 |
| `TURN_BASELINE_FLOOR` | v28 px_mid baseline (resolution-specific!) | see "Turn state machine" |
| `TURN_PEAK_DROP_PX` | Drop below peak that fires TURNING | 100 → 400 (rescale with floor) |
| `TURN_TARGET_YAW` | Heading delta that ends a turn | 60 → 85 |
| `TURN_SHARP_ANGLE` | Real angle during TURNING | ≤ `MAX_REAL_ANGLE_DEG` |

### Calibrate `POST_PID_GAIN`

`2.2` is inherited from `test_run.cpp`. To replace it with a measured
value: park on flat ground, command the servo manually (5°, 10°, 15°, 20°,
30°, 40°, 50°) while rolling at constant low speed, record yaw-rate from
`BNO055::eulerDeg.x` for each, and compute the servo angle that produces
the same turn radius the sim was tuned at for PID output 1.0. That ratio
is `POST_PID_GAIN`.

**This is the only knob that the turn state machine bypasses.** Do not
`× POST_PID_GAIN` inside the dispatcher for turn output — see "Turn state
machine" above.

---

## Verification checklist

Before any timed run:

### Boot
- [ ] `sudo ./main` prints `[SignDetector] Layout: ... nc=8 A=...`
- [ ] `[main] mask resolution:` printed on tick 1 — note it, compare to 320×192
- [ ] Camera frame shows in UI, mask overlays road correctly
- [ ] Centerline (red) follows the road center visually
- [ ] `kf` in UI ≈ 1.00 on open road, drops at junctions / obstacles
- [ ] `INV` flag off when mask is healthy, on when lost

### Bench (car stationary, wheels on blocks or free)
- [ ] **Heading sign** — rotate BNO055 clockwise viewed from above →
  `hdg` in UI **increases**. If it decreases, set `HEADING_SIGN = -1.0f`.
- [ ] **Servo sign** — manual `+N°` command turns the car right (matches
  sim convention). Matches `RealAngleToServo` calibration built earlier.
- [ ] **Sign engine layout** — print detections by showing a known sign to
  the camera; verify `sign=` in UI matches the expected class name. If
  wrong, the training `names` dict does **not** match `SignClass`.
- [ ] **Sign engine boxes** — the yellow rectangle covers the sign, not
  offset. If offset, `SIGN_INPUT_SIZE` is wrong (letterbox mismatch).

### Motion
- [ ] Motor at 0 → car does not move; at `MIN_SPEED_CMS` → smooth creep
- [ ] Emergency stop: Ctrl-C → motor stops, servo centers
- [ ] Turn SM trigger on a real junction — watch for `TURN=APPR` → `TURN=TURN`
  transition in UI, `hdg` delta at exit ≈ `TURN_TARGET_YAW`
- [ ] `TURN=TURN` should not fire early (before `px_mid` peaks). If it does,
  increase `TURN_PEAK_DROP_PX`.

---

## What NOT to do

- **Do not** tick the PID or the turn SM at inference rate. Use the fixed
  period gate in `ucrv4.cpp`.
- **Do not** multiply turn angles by `POST_PID_GAIN`. Turn angles are real
  degrees already, bounded by `MAX_REAL_ANGLE_DEG`.
- **Do not** reset `LaneFollower` on a single dropped road-mask frame. A
  lost mask yields `valid=false` (handled downstream) — only a TensorRT
  *failure* (`inferOK=false`) resets the follower.
- **Do not** let `TurnStateMachine` mutate `LaneFollowerV20` internals.
  The dispatcher reads both outputs and decides. (The one exception — the
  v28 seeding of `pid.prev_output = angle` on turn entry — belongs to the
  dispatcher in `ucrv4.cpp`, not to the SM.)
- **Do not** put model-specific code in `TrtSession.h`.
- **Do not** put hardware logic in `RoadSegmenter` / `SignDetector` /
  `LaneFollower` — those must stay pure so you can replay offline.
- **Do not** split `LaneFollowerV20.h`. It's cohesive — near/far/curve/
  occl/asym signals only exist in the PID context.
- **Do not** change `CONTROL_PERIOD_MS` without re-tuning every alpha and
  rate limit in `LaneFollowerV20.h` **and** `TurnStateMachine.h`.
- **Do not** trust `SignVoter` class order without verifying the engine's
  class names. Reordered enum = every turn decision is wrong, silently.

---

## Future work (round 3 scope)

Ordered by dependency:

1. **New sign semantics** — extend `TurnStateMachine::DecideDir()` (or add
   a `SignalStateMachine`) for `construction_ahead`, `roundabout`,
   `traffic_light`, and the `no_left` / `no_right` T-junction precedence
   rule. Requires the exact Round-3 challenge spec for those four signs.
2. **Parking state machine** — `mission/ParkingStateMachine.h` with
   `PARK_SEARCH → PARK_ALIGN → PARK_MANEUVER`, uses `parking` sign trigger
   and ToF for clearance. Needs reverse motor command and a separate
   dispatcher priority slot.
3. **Bridge handling** — new state in the mission SM, probably no new
   model. Verify road segmentation on bridge railings/lighting first.
4. **Pedestrian stop** — needs a crosswalk detector and a safe-resume
   policy (persistent evidence + ToF clearance + hysteresis). Do not
   implement as a single-frame trigger.
5. **E37 / E44 / E46** (pre-existing Log 12 items) — extend
   `ControlCommand` with `stampUsec`/`seq`/`mode`, add a `maxSteeringDeg`
   clamp inside `ActuatorThread` (belt-and-suspenders with the
   `MAX_REAL_ANGLE_DEG` clamp here), add a stale-command watchdog. The
   current `ucrv4.cpp` is a monolithic loop so these are less urgent, but
   they become mandatory the moment the autonomy stack is split into
   threads.
6. **E38** — wire `integrateYawRate()` into whatever thread owns the IMU
   so `relativeYawDeg()` becomes a usable low-drift heading source. Until
   then, the turn SM relies on `eulerDeg.x` and its slow drift.
7. **E43** — externalise the BNO055 calibration file path. It is
   currently hardcoded to `/home/car4/Desktop/CAR3/...`.

Each of (1)–(4) is additive: new files under `perception/`, `control/`, or
`mission/`, one dispatcher branch in `ucrv4.cpp`.

---

## Logs and debugging

The only runtime visibility today is the on-screen overlay
(`SHOW_UI = true`). Overlay content:

```
FPS <fps> | spd <applied>/<target> | PID <pre-gain> | out <post-clamp>
| TURN=<NORM|APPR|TURN> dir=<±1> | hdg <heading> | sign=<name|->
```

- `spd <applied>/<target>` — soft-start-ramped speed vs. what the
  follower wanted
- `PID <pre-gain>` — raw PID output before `POST_PID_GAIN`
- `out <post-clamp>` — final servo command after gain / clamp / calibration
- `TURN=<state> dir=<±1>` — turn SM state, `dir` = −1 for left, +1 for right
- `hdg <heading>` — BNO055 fused yaw in degrees
- `sign=<name>` — currently committed sign (via `SignVoter`), `-` if none

Yellow rectangles on the image are sign detections, labelled with
class name + confidence.

If you add a CSV logger later (E45), the columns worth capturing:

```
t_usec, frame, valid, coverage, kept_frac,
near_e, far_e, eff, curve_smooth, asym, occl,
angle_pre_gain, angle, servo_cmd, speed, applied_speed,
turn_state, turn_dir, px_mid, px_mid_peak, drop_from_peak,
frozen_base, brake_th, turn_th,
heading, sign_cls, sign_conf, sign_committed, sign_age,
tof[4]
```

Rationale: the first row of `near_e, far_e, eff, angle_pre_gain, angle`
explains **every** PID decision; the `turn_*` + `px_mid*` group explains
every **turn** decision; `heading` closes the loop on why a turn ended when
it did. Without all three groups in one row, you cannot diagnose a bad
turn from a log alone — the two failure modes (wrong sign, wrong heading
delta) look identical from the servo command trace.


---

## What changed vs. the previous README

Just to make the diff obvious:

| Section | Change |
|---|---|
| Architecture | Second perception path (SignDetector), BNO055, SignVoter, TurnStateMachine, dispatcher |
| File map | Added `SignDetector`, `SignVoter`, `TurnStateMachine`, `sensor/BNO055`; main is `ucrv4.cpp` not `main.cpp` |
| Boot log | Added `[SignDetector]` and `[main] mask resolution:` lines |
| Failure modes | Sign engine layout, `nc=8` check, BNO055 init, mask resolution |
| Fixed tick | Now applies to `TurnStateMachine` too, not just the PID |
| **New** Sign detection | 8-class table, trigger map, `SIGN_DETECT_EVERY=2`, layout auto-detect |
| **New** Sign voter | Persistence + blacklist + commit aging |
| **New** Turn state machine | State diagram, why peak-drop, why heading, angle convention, `TURN_BASELINE_FLOOR` resolution warning |
| **New** Heading source | `eulerDeg.x` vs `relativeYawDeg()`, `HEADING_SIGN` bench-verify requirement, drift note |
| Config guide | Added `HEADING_SIGN`, `SIGN_*`, `TURN_*`, `MAX_REAL_ANGLE_DEG`; flagged `TURN_BASELINE_FLOOR` resolution |
| Verification | Split into Boot / Bench / Motion; added sign engine + heading sign checks |
| What NOT to do | Added "don't apply POST_PID_GAIN to turn angles", "don't mutate follower from turn SM", "don't change tick rate without retuning alphas", "don't trust voter class order without verifying names" |
| Future work | Reordered to lead with new-sign semantics + parking; added E37/E38/E43/E44/E46 status |
| Logs | Added sign, turn, heading columns to the future CSV schema, with the rationale for why all three groups must be in one row |

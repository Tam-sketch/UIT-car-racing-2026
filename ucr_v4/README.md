# `README.md`

```markdown
# UCR 2026 — Jetson Nano Autonomous Car

Autonomous road-following on the physical round-3 car. Perception is a
TensorRT road-segmentation model; control is a port of the proven
`ucr_main_v20.py` PID. Everything runs on a fixed 10 Hz control tick so
the sim-tuned constants keep their meaning on the real robot.

---

## Architecture at a glance

```
Camera (BGR 640×384)
│
▼
RoadSegmenter          ← TensorRT engine + BGR→RGB + argmax
│  (CV_8UC1 mask at model output resolution)
▼
LaneFollowerV20        ← keep_largest → centerline → near/far/curve
│                     → occlusion gate → asymmetry → PID → speed
│  (v20::LaneFollower::Output)
▼
main loop              ← fixed 10 Hz tick, soft-start ramp, servo/motor
│
▼
Motor / Servo
```

The control loop ticks at a **fixed period**, regardless of how fast
inference runs. This is intentional — every rate limit and EMA inside
`LaneFollowerV20` is *per step*, so the tick rate defines their real
behavior.

---

## File map

```
config/
  RobotConfig.h           ← every tunable constant (one place)
perception/
  TrtSession.h            ← generic TensorRT engine wrapper (no model logic)
  RoadSegmenter.h         ← TwinLiteNet road segmentation (uses TrtSession)
control/
  LaneFollowerV20.h       ← the v20 PID stack (header-only, cohesive)
main.cpp                  ← hardware init + control loop + UI
```

### What goes where

| Concern | File |
|---|---|
| Engine load / buffer / stream | `TrtSession.h` |
| BGR→RGB, argmax, confidence margin | `RoadSegmenter.h` |
| Near/far split, occlusion, PID, speed | `LaneFollowerV20.h` |
| Constants you actually tune | `RobotConfig.h` |
| Wiring, timing, hardware order | `main.cpp` |

**Rule of thumb:** if a piece of code touches only the TensorRT API, it
belongs in `TrtSession.h`. If it also touches the model's tensor layout
(BGR→RGB, argmax, NMS, class decode), it belongs in a model wrapper
(`RoadSegmenter.h`, future `SignDetector.h`).

---

## Build

The build system is unchanged from the `thisinh` scaffold.

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
sudo ./main
```

No new libraries are linked. `nvinfer`, `cudart`, OpenCV, and pthreads
were already required.

---

## Running

```bash
sudo ./main
```

Expected boot log:

```
=== UCR2026: Auto Run ===
Engine: /home/car4/Downloads/road/kga2.engine
Control period: 100 ms
[RoadSegmenter] in 320x180 out 320x180
```

If you see:

- `[TrtSession] Init error: ...` → engine file missing or incompatible
  with the installed TensorRT version. Re-export the engine against the
  Jetson's TensorRT.
- `[RoadSegmenter] Expected 2 output channels` → the engine is not a
  TwinLiteNet binary-segmentation model. Check the export.
- `[RoadSegmenter] Dynamic dim ... not supported` → the engine has
  dynamic shape bindings. Fix the export to use static shapes.

Press `q` or `Esc` to stop.

---

## The fixed-rate control tick

`main.cpp` calls `LaneFollower::step()` **only** when
`steady_clock::now() >= next_control`. Inference runs as fast as the
hardware allows in between.

Why this matters:

- `rate_limit = 10` in the PID means **10° per step**, not 10° per
  frame. At 10 Hz that is 100°/s, which is the value the sim was tuned
  at. If you tick the PID every inference frame and inference happens
  at 30 Hz, the effective steering rate triples.
- Same for `derivative_alpha`, `d_far_ema_alpha`, `cov_ema_alpha`,
  `asym_ema`, and the speed EMA.

If you change `CONTROL_PERIOD_MS` in `RobotConfig.h`, **every
per-step constant in `LaneFollowerV20.h` changes meaning**. Treat it as
a lockstep value: if you need 20 Hz, halve the rate limits and re-tune
the alphas with `alpha = 1 - exp(-dt / tau)`.

---

## Adding a new perception model (sign detector, crosswalk, etc.)

Copy `RoadSegmenter.h` as a template. Structure:

```cpp
class SignDetector {
    TrtSession session_;          // reuse — do NOT reimplement
    std::vector<float> inBuf_, outBuf_;
    // ... model-specific dims
public:
    bool Init(const std::string& enginePath);
    bool Predict(const cv::Mat& bgrFrame, SignResult& out);
};
```

The sign model will need more post-processing (NMS, class decode,
bbox regression). Keep all of it inside `SignDetector::Predict` — do
not leak it into `TrtSession` and do not leak it into `main.cpp`.

Then in `main.cpp`:

```cpp
ucr::perception::SignDetector sign;
sign.Init(ucr::config::SIGN_ENGINE_PATH);

// inside the loop, after road.Predict():
ucr::perception::SignResult signs;
sign.Predict(frame, signs);
// → feed signs to a future MissionStateMachine
```

---

## Adding a new controller / mission state

When the sign detector is ready you will add a state machine between
`LaneFollowerV20` and the motor/servo. Suggested layout:

```
mission/
  MissionStateMachine.h    ← reads road + sign output, decides mode
  TurnController.h         ← takes over servo during TURN_COMMITTED
  ParkController.h         ← future
```

`main.cpp` becomes a dispatcher: it asks the state machine "who owns
the actuator this tick?", and forwards the command from the right
owner. Do **not** let a turn controller mutate `LaneFollowerV20`
internals — feed its output back into the state machine, and let the
state machine decide.

---

## Configuration guide

All system-level constants live in `config/RobotConfig.h`. The
`v20::LaneFollower` internal tuning (kp, kd, occlusion thresholds,
asymmetry caps) stays in `control/LaneFollowerV20.h` because those are
algorithm details, not system knobs.

### The values you will actually touch

| Constant | Meaning | Safe range |
|---|---|---|
| `MAX_SPEED_CMS` | Top forward speed | 5 → 25 |
| `MIN_SPEED_CMS` | Floor speed during curves / occl | 2 → 8 |
| `MAX_STEER_DEG` | Servo clamp | **verify ≤20 before going to 50** |
| `INVERT_STEER` | Servo sign flip | must be empirically checked |
| `CONTROL_PERIOD_MS` | PID tick period | leave at 100 unless re-tuning |
| `MAX_ACCEL_CMS2` | Speed ramp on start | 10 → 40 |
| `POST_PID_GAIN` | PID-to-servo scale | **calibrate empirically** |
| `CURVE_SCALE` | How aggressive curve slowdown is | 80 (more) → 200 (less) |
| `MIN_ROWS`, `COVERAGE_THRESHOLD` | Mask validity gate | tighten if garbage masks slip through |

### Calibrate `POST_PID_GAIN`

The current value (2.2) is a guess inherited from `test_run.cpp`. To
replace it with a measured value:

1. Park the car on a clear, flat surface.
2. Command the servo manually (5°, 10°, 15°, 20°, 30°, 40°, 50°) and
   let it hold for ~1 s each while rolling slowly forward at constant
   speed.
3. Record the yaw-rate from `BNO055::eulerDeg.x` for each angle.
4. Compute what servo angle produces the same turn radius that the
   sim's PID output of "1.0" was tuned to. That ratio is
   `POST_PID_GAIN`.

---

## Verification checklist

Before any timed run:

- [ ] `sudo ./main` boots, prints `[RoadSegmenter] in WxH out WxH`
- [ ] Camera frame shows in UI, mask overlays road correctly
- [ ] Centerline (red) follows the road center visually
- [ ] `kf` in UI ≈ 1.00 on open road, drops at junctions / obstacles
- [ ] `INV` flag off when mask is healthy, on when lost
- [ ] Motor at 0 → car does not move; at `MIN_SPEED_CMS` → smooth creep
- [ ] Servo at `+N°` → car turns in the direction you expect
- [ ] Emergency stop test: kill `main` (Ctrl-C) → motor stops, servo centers

---

## What NOT to do

- **Do not** tick the PID at inference rate. Use the fixed-period gate.
- **Do not** reset `LaneFollower` on a single dropped road-mask frame
  (current code resets on `inferOK=false` — a TensorRT *failure*, not a
  lost mask; a lost mask yields `valid=false` but does not reset).
- **Do not** put model-specific code in `TrtSession.h`.
- **Do not** put hardware logic (camera, motor, servo) in
  `RoadSegmenter` or `LaneFollower` — those should be pure.
- **Do not** split `LaneFollowerV20.h`. It is cohesive; the near/far/
  curve/occl/asym signals feed the PID and only exist in that context.

---

## Future work (round 3 scope)

Ordered by dependency:

1. **Sign detector** — `perception/SignDetector.h` on top of `TrtSession`
2. **Mission state machine** — `mission/MissionStateMachine.h`
3. **Turn controller** — `mission/TurnController.h`, uses IMU heading for
   completion (requires fixing E38: `integrateYawRate` is never called
   in `SensorThread`)
4. **Bridge handling** — new state, possibly no new model
5. **Pedestrian stop** — needs the crosswalk detector; needs a
   safe-resume policy
6. **Parallel parking** — `mission/ParkController.h`, needs ToF
   clearance logic and reverse motion

Each of (1)–(6) is additive: new files under `perception/` and
`mission/`, one new line in `main.cpp` to wire it in, and (once more
than 2 owners exist) a small dispatcher in `main.cpp`.

---

## Logs and debugging

Currently the only visibility is the on-screen overlay (`SHOW_UI =
true`). The overlay shows:

```
FPS <fps> | spd <applied>/<target> | PID <pre-gain> | out <post-clamp> | kf <kept_frac> [INV]
```

- `spd <applied>/<target>` — soft-start ramped value vs. what the
  follower wanted
- `PID <pre-gain>` — raw PID output before `POST_PID_GAIN`
- `out <post-clamp>` — final servo command after gain, clamp, invert
- `kf <kept_frac>` — largest-component fraction; drops at junctions
- `INV` — mask is invalid but coverage is still > 2 %

If you add a CSV logger later, the columns to capture are:
`t_usec, valid, coverage, kept_frac, near_e, far_e, eff, curve_smooth,
asym, occl, angle_pre_gain, angle, speed, tof[4], imu_yaw`.
```
```

That's the whole file. Save as `README.md` at the repo root (or in the round-3 source folder if you keep round-1/2 code in the same tree).
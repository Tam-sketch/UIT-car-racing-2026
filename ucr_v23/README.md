# README.md (Cập nhật cho kiến trúc Path-based Turn Planner)

Dưới đây là nội dung `README.md` hoàn chỉnh, đã cập nhật theo kiến trúc mới (Path-based, TurnPlanner, bỏ TurnStateMachine).

```markdown
# UCR 2026 — Jetson Nano Autonomous Car

Autonomous road-following on the physical round-3 car. Perception is a
TensorRT road-segmentation model; control is a port of the proven
`ucr_main_v20.py` PID. 

**Cập nhật Round 3:** 
- Chuyển sang **Path-based Turn Planner** (port 1:1 từ `openinglogic.py`).
- Thay thế hoàn toàn `TurnStateMachine` cũ (dùng IMU heading) bằng cách 
  modify `row_centers` để PID tự bẻ lái.
- Thêm `SignVoter` để chống nhiễu sign detect đơn lẻ.

Everything runs on a fixed 10 Hz control tick so the sim-tuned constants 
keep their meaning on the real robot.

---

## Architecture at a glance

```
Camera (BGR 640×384)
│
├─→ SignDetector (YOLOv8) ──→ SignVoter ──→ setSign()
│                                              │
▼                                              ▼
RoadSegmenter (TwinLiteNet)            LaneFollowerV20
│  (CV_8UC1 mask 320×184)               │  ├─ extractCenterline
▼                                      │  ├─ TurnPlanner.update()
TurnPlanner (openinglogic.py port) ────┘  │   (modify path nếu có biển)
   (morph_close, edge_dev, bezier)         ├─ OcclusionGate
                                           ├─ Asymmetry
                                           ├─ PID (v20)
                                           └─ Speed planning
                                              │
                                              ▼
main loop (10 Hz) ──→ Motor / Servo
```

---

## File map

```
config/
  RobotConfig.h           ← every tunable constant (hardware + system)
perception/
  TrtSession.h            ← generic TensorRT engine wrapper
  RoadSegmenter.h         ← TwinLiteNet road segmentation
  SignDetector.h          ← YOLOv8 sign detection (NMS + unletterbox)
control/
  TurnPlanner.h           ← [NEW] Port 1:1 từ openinglogic.py
  LaneFollowerV20.h       ← v20 PID + tích hợp TurnPlanner
  SignVoter.h             ← Chống nhiễu sign (multi-frame voting)
  TurnStateMachine.h      ← [DEPRECATED] Không còn dùng, có thể xóa
main.cpp                  ← hardware init + control loop + UI
```

### What goes where

| Concern | File |
|---|---|
| Engine load / buffer / stream | `TrtSession.h` |
| BGR→RGB, argmax, confidence margin | `RoadSegmenter.h` |
| Sign bbox, NMS, class decode | `SignDetector.h` |
| Multi-frame sign voting | `SignVoter.h` |
| **Opening detection, Bézier path, ghost mode** | **`TurnPlanner.h`** |
| Near/far split, occlusion, PID, speed | `LaneFollowerV20.h` |
| Hardware constants, servo calibration | `RobotConfig.h` |
| Wiring, timing, hardware order | `main.cpp` |

**Rule of thumb:** 
- Nếu code chỉ chạm TensorRT API → `TrtSession.h`.
- Nếu chạm tensor layout (BGR→RGB, argmax, NMS) → model wrapper (`RoadSegmenter.h`, `SignDetector.h`).
- Nếu xử lý hình học đường đi (centerline, path) → `TurnPlanner.h` hoặc `LaneFollowerV20.h`.

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
=== UCR2026 Auto Run (Path-based, signs=yes) ===
[RoadSegmenter] in 320x184 out 320x184
```

If you see:

- `[TrtSession] Init error: ...` → engine file missing or incompatible
  with the installed TensorRT version.
- `[RoadSegmenter] Expected 2 output channels` → the engine is not a
  TwinLiteNet binary-segmentation model.
- `[SignDetector] engine has no I/O` → sign engine missing.
- `signs=NO` → sign detector không load được, turn sẽ không kích hoạt.

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

## Turn Planner Logic (Mới)

`TurnPlanner` hoạt động theo cơ chế **Path-based** (thay vì force-turn bằng IMU cũ):

1. **Detect Opening:**
   - Morph close (kernel 35x5) để làm sạch mask.
   - Tính `edge_profile` (trái/phải) của vùng đường.
   - Smooth bằng moving average (win 5), baseline bằng rolling median (win 50).
   - `deviation > 15` px liên tục ≥ 10 hàng → phát hiện opening (ngã rẽ).

2. **Trigger:**
   - Nếu `op.y_mid > 0.75 * mask.rows` (gần xe) → thực hiện rẽ.
   - Với biển `no_left`/`no_right`, kiểm tra `isStraightBlocked`. Nếu đường thẳng vẫn đi được → bỏ qua biển, đi thẳng.

3. **Modify Path:**
   - Tìm `x_turn` (điểm giữa opening và baseline).
   - Vẽ đường cong **Bézier bậc 2** từ centerline hiện tại vào opening.
   - Extend thẳng vào làn mới (tối đa 80px).

4. **Ghost Mode:**
   - Nếu mất dấu opening nhưng vẫn đang trong turn, dùng `last_y_mid` và `turn_width` để tạo "opening ảo" và tiếp tục rẽ.

**Lưu ý:** `TurnPlanner` modify `row_centers`. PID v20 sau đó sẽ tính `near_error`, `far_error` từ path mới này và tự động bẻ lái.

---

## Adding a new perception model

Copy `RoadSegmenter.h` as a template. Structure:

```cpp
class NewModel {
    TrtSession session_;
    std::vector<float> inBuf_, outBuf_;
public:
    bool Init(const std::string& enginePath);
    bool Predict(const cv::Mat& bgrFrame, Result& out);
};
```

Keep all model-specific post-processing inside `Predict`.

Then in `main.cpp`:

```cpp
ucr::perception::NewModel model;
model.Init("path/to/engine");

// inside the loop:
ucr::perception::Result result;
model.Predict(frame, result);
```

---

## Configuration guide

All system-level constants live in `config/RobotConfig.h`. The
`v20::LaneFollower` internal tuning (kp, kd, occlusion thresholds,
asymmetry caps) stays in `control/LaneFollowerV20.h` because those are
algorithm details, not system knobs.

### The values you will actually touch

| Constant | Meaning | Safe range |
|---|---|---|
| `MAX_SPEED_CMS` | Top forward speed | 5 → 20 |
| `MIN_SPEED_CMS` | Floor speed during curves / occl | 2 → 8 |
| `MAX_REAL_ANGLE_DEG` | Servo physical clamp | **verify ≤ 30** |
| `CONTROL_PERIOD_MS` | PID tick period | leave at 100 unless re-tuning |
| `MAX_ACCEL_CMS2` | Speed ramp on start | 10 → 40 |
| `POST_PID_GAIN` | PID-to-servo scale | **calibrate empirically** |
| `CURVE_SCALE` | How aggressive curve slowdown is | 80 (more) → 200 (less) |
| `MIN_ROWS`, `COVERAGE_THRESHOLD` | Mask validity gate | tighten if garbage masks slip through |

### Turn Planner Tuning (Trong `TurnPlannerConfig`)

Các thông số mặc định đã được port 1:1 từ Python (dành cho mask 640x360). Vì C++ chạy mask 320x184, **nên kiểm tra lại nếu thấy turn hoạt động không đúng**:

| Constant | Default | Meaning |
|---|---|---|
| `open_thresh` | 15 | Ngưỡng deviation để nhận là opening. Giảm → nhạy hơn. |
| `min_height` | 10 | Số hàng pixel tối thiểu của opening. Giảm → dễ trigger. |
| `turn_trigger_y_ratio` | 0.75 | Trigger khi opening gần xe (y > 0.75*H). |
| `corner_radius` | 20 | Độ cong của Bézier. Tăng → cua mượt hơn. |
| `turn_speed_cap` | 12.0 | Tốc độ tối đa khi đang modify path (cm/s). |

*Lưu ý:* `turn_speed_cap` nằm trong `v20::Config` (không phải `TurnPlannerConfig`).

---

## Verification checklist

Before any timed run:

- [ ] `sudo ./main` boots, prints `[RoadSegmenter] in 320x184 out 320x184`
- [ ] Camera frame shows in UI, mask overlays road correctly
- [ ] Centerline (red) follows the road center visually
- [ ] `kf` in UI ≈ 1.00 on open road, drops at junctions / obstacles
- [ ] `INV` flag off when mask is healthy, on when lost
- [ ] Motor at 0 → car does not move; at `MIN_SPEED_CMS` → smooth creep
- [ ] Servo at `+N°` → car turns in the direction you expect
- [ ] **Đưa biển `turn_left` trước cam → line chuyển sang MAGENTA, `turn=Y`**
- [ ] **Bỏ biển → line trở lại đỏ, `turn=N`**
- [ ] Emergency stop test: kill `main` (Ctrl-C) → motor stops, servo centers

---

## What NOT to do

- **Do not** tick the PID at inference rate. Use the fixed-period gate.
- **Do not** reset `LaneFollower` on a single dropped road-mask frame
  (current code resets on `inferOK=false` — a TensorRT *failure*, not a
  lost mask).
- **Do not** put model-specific code in `TrtSession.h`.
- **Do not** put hardware logic (camera, motor, servo) in
  `RoadSegmenter` or `LaneFollower` — those should be pure.
- **Do not** split `LaneFollowerV20.h`. It is cohesive.
- **Do not** dùng lại `TurnStateMachine.h`. Logic force-turn bằng IMU đã bị loại bỏ hoàn toàn.

---

## Future work (round 3 scope)

Ordered by dependency:

1. **Bridge handling** — new state, possibly no new model
2. **Pedestrian stop** — needs the crosswalk detector; needs a
   safe-resume policy
3. **Parallel parking** — `mission/ParkController.h`, needs ToF
   clearance logic and reverse motion

Each of (1)–(3) is additive.

---

## Logs and debugging

Currently the only visibility is the on-screen overlay (`SHOW_UI = true`). 
The overlay shows:

```
FPS <fps> | spd <applied>/<target> | PID <pre-gain> | sign=<sign_name> | turn=<Y/A/N> side=<left/right> ymid=<y> | hdg <heading>
```

- `spd <applied>/<target>` — soft-start ramped value vs. what the
  follower wanted.
- `PID <pre-gain>` — raw PID output before `POST_PID_GAIN`.
- `sign=<name>` — sign đang được voter commit (hoặc `-` nếu không có).
- `turn=Y` — TurnPlanner đang modify path (line màu MAGENTA).
- `turn=A` — Đang tiếp cận opening (Approaching), chưa modify path.
- `turn=N` — Không có turn.
- `side=<left/right>` — Hướng rẽ đang được lên kế hoạch.
- `ymid=<y>` — Vị trí y của opening gần nhất.
- `hdg <heading>` — Heading từ BNO055 (chỉ để tham khảo, không dùng cho control).
```

---

### Tóm tắt các thay đổi chính so với README cũ:
1. **Architecture**: Cập nhật sơ đồ khối (thêm SignDetector, SignVoter, TurnPlanner).
2. **File map**: Thêm `TurnPlanner.h`, `SignVoter.h`; đánh dấu deprecated `TurnStateMachine.h`.
3. **Turn Planner Logic**: Thêm section giải thích cơ chế Path-based mới.
4. **Tuning**: Thêm bảng thông số `TurnPlannerConfig` và `turn_speed_cap`.
5. **Verification**: Thêm checklist kiểm tra line MAGENTA khi có biển.
6. **UI Overlay**: Cập nhật format log mới (`turn=Y/A/N`, `side=...`, `ymid=...`).

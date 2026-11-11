# Trả lời 2 câu hỏi

## Câu 1: Có tạo nhiều centerline không? — **KHÔNG, chỉ 1 centerline**

Nhìn vào `extractCenterline()`:

```cpp
for (int y = h - 1; y >= 0; y--) {
    int s, e;
    if (largestRun(row, w, s, e)) row_centers.push_back({(s + e) / 2, y});
}
```

`largestRun()` **chỉ lấy đoạn liên tục dài nhất** trên mỗi hàng. Vậy:

- Mỗi hàng → **1 điểm center** duy nhất
- Cả mask → **1 centerline** duy nhất (bám theo "largest run" per row)

**Kể cả khi tắt `keep_largest` (trong V6.1 patch khi turn active)**: `largestRun` vẫn chỉ chọn **1 run mỗi hàng**. Đây chính là điểm mà tài liệu "New Lane Association" chỉ ra:

> Nếu lane cũ (thẳng, rộng) và nhánh rẽ trái (hẹp), `largestRun` sẽ chọn lane cũ → centerline vẫn bám lane cũ.

**Hệ quả:** V6.1 tắt `keep_largest` chỉ giúp `detectOpenings` nhìn thấy các nhánh (vì nó scan edge profiles), nhưng `extractCenterline` **vẫn chỉ cho 1 path**. Đây là lý do cần thêm `turn_committed_` state: khi đã commit, PID dùng Bézier path (từ memory), không dùng centerline nữa.

## Câu 2: Chỉ vào opening khi có biển báo? — **ĐÚNG cho ENTRY, SAI cho CONTINUATION**

```cpp
std::vector<Opening> detectOpenings(const cv::Mat& mask, const SignState& sign) {
    if (!sign.active || sign.side == 0) return openings;  // ← CHẶN khi không có sign
    ...
}
```

| Giai đoạn | Cần sign? |
|---|---|
| **Detect opening** (tìm nhánh rẽ) | ✅ Có — `sign.active` phải true |
| **Commit turn** (bắt đầu Bézier) | ✅ Có — cần cả sign + opening |
| **Continuation** (Bézier chạy) | ❌ Không — `turn_committed_` persist |
| **Ghost mode** (opening biến mất) | ❌ Không — synthesize từ `commit_turn_point_` |
| **Release** (về PID) | ❌ Không — dùng mask/heading/max_ticks |

**Tóm tắt:** Biển báo cần cho **kích hoạt** turn, nhưng **không cần** để **duy trì** turn đã commit.

---

# README.md (cập nhật cho V6.1)

```markdown
# UCR 2026 — Jetson Nano Autonomous Car (V6.1)

Autonomous road-following + sign-triggered Bezier turns. TensorRT
road-segmentation + YOLO sign detection + sign voter + opening detector +
Bezier path planner + PID. Fixed 10 Hz control tick.

**V6.1 status:** all components wired. Sign detection, voter, opening
detector, and Bezier turn are ACTIVE in `main.cpp`. Traffic-light (red/green)
and pedestrian stop are active.

---

## Current status — V6.1 (fully wired)

| Component | File | Status |
|---|---|---|
| Hardware bring-up | `hardware/*` | Running |
| TensorRT session | `perception/TrtSession.h` | Running |
| Road segmentation | `perception/RoadSegmenter.h` | Running |
| v20 PID + occlusion + asymmetry | `control/LaneFollowerV21.h` (V20 core) | Running |
| Sign detection (YOLO 8-class) | `perception/SignDetector.h` | Running |
| Sign voter | `control/SignVoter.h` | Running |
| Opening detection (V21) | `control/LaneFollowerV21.h` | Running |
| Bezier path + TurnPlan | `control/LaneFollowerV21.h` | Running |
| Traffic light (HSV) + pedestrian stop | `main.cpp` | Running |
| Turn commit + completion | `control/LaneFollowerV21.h` | Running (V6.1) |
| Old `TurnStateMachine.h` (area-spike) | `control/TurnStateMachine.h` | Deprecated, unused |
| Old `SkeletonTopology.h` | `perception/SkeletonTopology.h` | Deprecated, unused |

---

## Architecture (V6.1)

```
Camera (BGR 640×384)
│
├─→ SignDetector (YOLO) ──→ SignVoter ──→ SignState
│                            (temporal filter)
│
▼
RoadSegmenter (TwinLiteNet 320×184)
│
▼
LaneFollowerV21 (V6.1)
  ├─ extractCenterline  ← ONLY 1 centerline per frame
  │                        (largest-run per row; keep_largest
  │                         disabled during turn for openings)
  ├─ detectOpenings     ← ONLY when sign.active
  │                        (edge-profile deviation)
  ├─ TurnPlan commit    ← sign + opening + tangent check
  ├─ Bezier path        ← cached; survives sign timeout
  ├─ TurnPlan release   ← mask-at-turn-point / heading / max_ticks
  └─ PID → angle, speed
│
▼
main.cpp ──→ Motor + Servo
  └─ stop overrides: person > red light > normal
```

**Key architectural notes:**

1. **Single centerline:** `extractCenterline()` runs `largestRun()` per
   row → exactly ONE path per frame. During a turn, this path still
   tracks the OLD road (because old road is wider than the new branch).
   The Bezier path is the actual turn target — PID uses `final_path`
   which is `current_plan_.path` while committed, not the raw centerline.

2. **Sign gates opening detection:** `detectOpenings()` returns empty
   when `sign.active == false`. No sign → no opening → no turn. This is
   intentional to avoid false positives from any road widening.

3. **Sign is NOT required to finish a turn:** once `turn_committed_` is
   set (sign + opening + tangent check), the turn completes even if the
   sign voter times out, because completion is driven by:
   - mask coverage at `commit_turn_point_`, OR
   - heading delta ≥ 60° (optional IMU), OR
   - `turn_max_ticks` fallback (3 s hard cap).

---

## V6.1 patches (relative to earlier V6)

Five targeted fixes for the "Bezier dies mid-turn" bug:

| # | Patch | Effect |
|---|---|---|
| 1 | Skip `keep_largest` when `turn_committed_` | Opening detection sees both old road + new branch |
| 2 | Tangent sanity check before commit | Rejects commit when centerline still bends the wrong way |
| 3 | Release via mask-at-`commit_turn_point_` | Replaces unreliable `cl.near_error` check |
| 4 | Optional heading release (BNO055) | If IMU present, releases at 60° turn |
| 5 | `turn_max_ticks` fallback | Hard cap at 30 ticks (~3 s) to prevent permanent stuck |

Plus two earlier fixes carried over:

- `turn_memory_.last_y_mid = best.y_mid` (was `std::min` — bug locked to
  furthest opening instead of nearest).
- Skip `OcclusionGate` while `turn_committed_` (avoids false emergency
  during turn when mask suddenly loses road).

Traffic-light classifier upgraded from per-pixel loop to `cv::inRange`
+ `cv::countNonZero` (5–10× faster). Green light clears `lastRedSeenTick`
so car resumes immediately when light turns green — UNLESS a pedestrian
is also in frame, in which case `stopForPerson` wins.

---

## File map

```
config/
  RobotConfig.h           ← all tunables (Section A + Section B)
perception/
  TrtSession.h            ← TensorRT wrapper (FP16/FP32)
  RoadSegmenter.h         ← TwinLiteNet (2-ch argmax)
  SignDetector.h          ← YOLO 8-class head
  SkeletonTopology.h      ← deprecated (unused)
control/
  LaneFollowerV21.h       ← V20 PID + V21 opening + V6.1 turn plan
  SignVoter.h             ← tick-indexed sign vote
  TurnStateMachine.h      ← deprecated (unused)
main.cpp                  ← hardware + control loop + UI
```

---

## Build

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
sudo ./main
```

Expected boot:
```
=== UCR2026 V6.1 (turns=yes) ===
Road: /home/car4/Downloads/road/kga2.engine
Sign: /home/car4/Downloads/sign/huy_sign.engine
Control: 100 ms
[RoadSegmenter] in 320x184 out 320x184
[SignDetector] in 416x416 out (12,3549) classes=8
```

---

## Fixed-rate tick

`main.cpp` calls `LaneFollower::step()` only when
`steady_clock::now() >= next_control`. Inference runs in between.

Every rate limit and EMA inside `LaneFollowerV21` is **per-step**. If you
change `CONTROL_PERIOD_MS`, recompute alphas with `1 - exp(-dt/tau)` and
rescale rate limits proportionally.

---

## Sign classes (8-class engine)

| id | name | action |
|---|---|---|
| 0 | construction_ahead | turn left (map-specific) |
| 1 | no_left | turn right (or straight if 4-way) |
| 2 | no_right | turn left (or straight if 4-way) |
| 3 | parking | not yet wired |
| 4 | roundabout | not yet wired |
| 5 | traffic_light | red = stop, green = resume |
| 6 | turn_left | turn left |
| 7 | turn_right | turn right |
| 8 | walking_person | stop immediately |

**Pedestrian overrides everything** (including green light).

---

## Turn completion — 3 signals

While `turn_committed_ == true`, turn persists regardless of sign state.
Release happens when **any** of:

| Signal | Trigger |
|---|---|
| A | Mask coverage ≥ 65% in a 24×24 window at `commit_turn_point_` |
| B | `|Δheading| ≥ 60°` from `commit_heading_` (only if IMU passed) |
| C | `turn_tick_ ≥ 30` (~3 s) — hard safety cap |

Requires `turn_tick_ ≥ 8` (0.8 s minimum) before A or B can fire.

---

## Configuration guide

### Section A — active

| Constant | Meaning | Safe range |
|---|---|---|
| `MAX_SPEED_CMS` | Top forward speed | 5 → 20 |
| `MIN_SPEED_CMS` | Floor speed (curves, lost mask) | 2 → 12 |
| `MAX_REAL_ANGLE_DEG` | Real-angle bound | 20 → 40 |
| `MAX_SERVO_CMD` | Raw servo clamp | 40 → 60 |
| `CONTROL_PERIOD_MS` | PID tick period | leave at 100 |
| `MAX_ACCEL_CMS2` | Speed ramp | 10 → 40 |
| `POST_PID_GAIN` | PID → real-angle scale | 1.4 → 2.2 (see E52) |
| `CURVE_SCALE` | Curve slowdown | 80 (more) → 200 (less) |
| `MIN_ROWS`, `COVERAGE_THRESHOLD` | Mask validity | tighten if garbage slips |

### Turn completion tuning

| Constant | Default | Meaning |
|---|---|---|
| `TURN_MIN_TICKS` | 8 | Min ticks before release allowed |
| `TURN_MAX_TICKS` | 30 | Force release (safety) |
| `TURN_RELEASE_ROAD_FRAC` | 0.65 | Coverage threshold at turn point |
| `TURN_RELEASE_WINDOW_PX` | 12 | Half-window for coverage check |
| `TURN_COMPLETE_DEG` | 60 | Heading delta for release (if IMU) |
| `TURN_COMMIT_PROXIMITY` | 0.50 | Commit only when opening y > 0.5·H |
| `TURN_SYNTH_WIDTH_PX` | 60 | Fallback Bezier width when ghost |

### Opening detection

| Constant | Default | Meaning |
|---|---|---|
| `OPEN_THRESH` | 15 | Pixel deviation threshold |
| `OPEN_MIN_HEIGHT` | 10 | Min vertical extent |
| `OPEN_SMOOTH_WIN` | 5 | Moving average for edges |
| `OPEN_BASE_WIN` | 50 | Rolling median for baseline |

### Bezier path

| Constant | Default | Meaning |
|---|---|---|
| `BEZIER_CORNER_RADIUS` | 20 | Corner curvature |
| `BEZIER_MIN_TURN_DIST` | 4 | Min turn distance |
| `BEZIER_EXTEND_LEN` | 80 | Extension into new lane |

---

## Known issues

1. **E52 — `post_pid_gain` saturation:** with `POST_PID_GAIN=2.2` and
   `MAX_REAL_ANGLE_DEG=30`, PID output `>13.6` saturates. Upper 38% of
   PID range is flat. Consider lowering to 1.4 or raising the angle bound.

2. **Single centerline limitation:** `extractCenterline` uses
   `largestRun` per row → only 1 path. During turn, if new branch is
   narrower than old road, raw centerline still tracks old road. This
   is why V6.1 uses `turn_committed_` state and Bezier path instead of
   re-extracting centerline during turn.

3. **Ghost turn uses fixed width:** when opening disappears mid-turn,
   synthesized opening uses `turn_memory_.turn_width` (from last seen
   opening) or `TURN_SYNTH_WIDTH_PX` (60) as fallback. If actual lane
   width is very different, Bezier endpoint may be off by ±20 px.

---

## What NOT to do

- Do not tick PID at inference rate — use `next_control` gate.
- Do not reset `LaneFollower` on a lost-mask frame. Only TensorRT
  failure (`inferOK == false`) triggers hard reset.
- Do not `#include` `TurnStateMachine.h` or `SkeletonTopology.h` — they
  are deprecated and unused.
- Do not re-enable `keep_largest` during turn — V6.1 depends on it being
  disabled so opening detection sees both branches.
- Do not change `CONTROL_PERIOD_MS` without retuning all per-step
  rate limits and EMA alphas.

---

## Verification checklist

Boot:
- [ ] `sudo ./main` prints `[RoadSegmenter] in 320x184 out 320x184`
- [ ] `[SignDetector] in 416x416 out (12,3549) classes=8`
- [ ] `turns=yes` in boot header

Straight-line:
- [ ] Green mask overlays road correctly
- [ ] Red centerline follows road center
- [ ] `kf ≈ 1.00` on open road
- [ ] `INV` off when mask healthy, on when lost

Turn:
- [ ] Đưa biển `turn_left` trước cam → state chuyển `TURN APPROACHING`
- [ ] Tiến gần opening → state chuyển `TURN LEFT` (path magenta)
- [ ] Che opening giữa cua → state chuyển `GHOST TURN LEFT` (path yellow, vẫn sống)
- [ ] Xe hoàn thành cua → state về `CENTERLINE` (không sớm, không muộn)
- [ ] Nếu kẹt giữa cua → sau ~3s (30 tick) tự release

Traffic light:
- [ ] Đưa ảnh đèn đỏ trước cam → `spd` về 0, state hiển thị `STOP:red`
- [ ] Đưa ảnh đèn xanh → `spd` trở lại (trừ khi có người)

Pedestrian:
- [ ] Đưa ảnh người đi bộ → `spd` về 0, state `STOP:person`
- [ ] Người + đèn xanh → vẫn dừng (person wins)
- [ ] Người biến mất sau 2s → xe tự đi tiếp

Emergency:
- [ ] Ctrl-C → motor stops, servo centers

---

## Debug UI

```
FPS 10.2 | spd 12.0/12.0 | PID -3.2 | out -15 | kf 0.98
sign turn_left | TURN LEFT | near -2.1 far 5.4 cov 0.82
```

**State strings:**
- `CENTERLINE` — PID bám đường bình thường
- `TURN APPROACHING` — sign active, opening thấy nhưng chưa đủ gần để commit
- `TURN PENDING (tangent)` — opening gần nhưng centerline vẫn thẳng (chờ xe tiến)
- `TURN LEFT/RIGHT` — Bezier active, path magenta
- `GHOST TURN LEFT/RIGHT` — opening biến mất, path yellow
- `INVALID MASK` — mask rỗng, xe dừng khẩn cấp

**Stop strings in log:**
- `-` — không dừng
- `STOP:red` — đèn đỏ
- `STOP:person` — người đi bộ

---

## Logs

On-screen overlay only (`SHOW_UI = true`). To add CSV logger, columns:
`t_usec, valid, coverage, kept_frac, near_e, far_e, eff, curve_smooth,
asym, occl, angle_pre_gain, angle, servo_cmd, applied_speed, state_str,
turn_committed, turn_tick, sign_id, person_age, red_age`.
```

---

## Tóm tắt 2 câu trả lời

1. **Chỉ 1 centerline:** `extractCenterline` dùng `largestRun` per row → **1 path duy nhất**. Trong turn, path này vẫn bám đường cũ (rộng hơn) → đó là lý do cần `turn_committed_` state + Bezier path cache để không phụ thuộc centerline nữa.

2. **Chỉ vào opening khi có biển:** `detectOpenings` return empty nếu `sign.active == false`. Nhưng **sau khi commit**, turn vẫn tiếp tục dù biển có mất — vì completion dựa vào mask/heading/max_ticks, không phụ thuộc sign.

Nếu sau này muốn multi-branch extraction (nhiều centerline thật sự), cần viết `extractMultiBranchCenterlines` — đây là scope ~300-500 dòng code, không nằm trong V6.1.

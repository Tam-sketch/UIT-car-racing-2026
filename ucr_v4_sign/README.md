Dưới đây là file `README.md` hoàn chỉnh dành cho phiên bản V3+ (đã tích hợp Mission State Machine và Sign Detection). Bạn lưu file này ở thư mục gốc của `ucr_v3`.

```markdown
# UCR 2026 — Jetson Nano Autonomous Car (V3+)

Autonomous road-following on the physical round-3 car. Perception combines a
TensorRT road-segmentation model and a YOLO-based sign detector. Control uses
a port of the proven `ucr_main_v20.py` PID, gated by a deterministic Mission
State Machine that handles intersection turns via sign commands. 

Everything runs on a fixed 10 Hz control tick so the sim-tuned constants keep
their meaning on the real robot.

---

## Architecture at a glance

```
Camera (BGR 640×384) + IMU (BNO055)
│
├─► RoadSegmenter (TensorRT) ──► CV_8UC1 road mask
│
└─► SignDetector (TensorRT)  ──► vector<Detection> (class_id, conf, bbox)
│
▼
MissionStateMachine ──► Quyết định trạng thái:
│                        - LANE_FOLLOW (dùng LaneFollowerV20)
│                        - FORCE_TURN  (dùng TurnController + IMU heading)
│
▼
main loop ──► Fixed 10 Hz tick, hiệu chuẩn góc lái (Regression), motor/servo
│
▼
Motor (UART) / Servo (I2C PWM)
```

### Luồng hoạt động (Data Flow)
1. **10 Hz Tick**: Cứ mỗi 100ms, lấy ảnh từ Camera và góc Heading từ IMU.
2. **Perception**: Chạy song song `RoadSegmenter` (lấy mask đường) và `SignDetector` (lấy biển báo).
3. **Sign Voting**: `SignVoter` tích lũy biển báo qua 5 frame. Nếu đủ tin cậy, nó "Commit" (chốt) quyết định.
4. **Mission State Machine**: 
   - Nếu đang `LANE_FOLLOW`: Đưa mask cho `LaneFollowerV20`.
   - Nếu có biển báo rẽ VÀ đường đủ rộng (Bottom Coverage > 78%): Chuyển sang `FORCE_TURN`.
   - Nếu đang `FORCE_TURN`: Giao quyền cho `TurnController` bẻ lái cứng dựa trên góc Heading thực tế của IMU.
5. **Steering Calibration**: Góc PID thô (hoặc góc Force Turn) được đưa qua **Phương trình hồi quy** để bù trừ hao hụt cơ khí trước khi gửi xuống Servo.

---

## File map

```
config/
  RobotConfig.h           ← Mọi hằng số hệ thống, đường dẫn model, phương trình hồi quy
perception/
  TrtSession.h            ← Generic TensorRT engine wrapper
  RoadSegmenter.h         ← TwinLiteNet road segmentation
  SignDetector.h          ← YOLO sign detection (decode bounding box + NMS)
control/
  LaneFollowerV20.h       ← The v20 PID stack (near/far/occl/asym)
  SignVoter.h             ← Vote & Commit logic (chống nhiễu biển báo)
  TurnController.h        ← Force turn phases (ENTRY → TURNING → EXIT)
mission/
  MissionStateMachine.h   ← Bộ não trung tâm (FOLLOW vs FORCE_TURN)
main.cpp / ucrv3.cpp      ← Khởi tạo hardware, chạy vòng lặp 10Hz, áp dụng Regression
```

---

## Steering Calibration (Thay thế `POST_PID_GAIN`)

Trong mô phỏng cũ, `POST_PID_GAIN = 2.2` được dùng để bù hao hụt. Trên xe thật, hao hụt là **phi tuyến tính** và **bất đối xứng**. Do đó, V3+ sử dụng phương trình hồi quy đo đạc thực tế:

- **Bánh trái:** `y = 0.96x - 6.2`  ⇒  `x = (y + 6.2) / 0.96`
- **Bánh phải:** `y = 0.71x + 2.5`  ⇒  `x = (y - 2.5) / 0.71`

*(Trong đó: `y` là góc PID yêu cầu, `x` là góc thực tế cần gửi xuống servo).*

**Lưu ý quan trọng:** Trong `ucrv3.cpp`, bạn **phải** khởi tạo `lane_cfg.post_pid_gain = 1.0f;` để tắt hệ số nhân 2.2 cũ. Hàm `ApplySteeringCalibration` sẽ đảm nhận việc chuyển đổi góc.

---

## Sign & Mission Configuration

### Các loại biển báo được hỗ trợ (Class ID từ model `huy_sign.engine`)
| ID | Class name | Hành động |
|---|---|---|
| 6 | `turn_left` | Rẽ trái |
| 7 | `turn_right` | Rẽ phải |
| 0 | `construction_ahead` | **Xử lý như `turn_left`** (theo yêu cầu bài toán) |
| 1 | `no_left` | Không rẽ trái (mặc định rẽ phải hoặc đi thẳng) |
| 2 | `no_right` | Không rẽ phải (mặc định rẽ trái hoặc đi thẳng) |
| 5 | `traffic_light` | *Chưa xử lý* |
| 3 | `parking` | *Chưa xử lý* (sẽ phát triển sau) |
| 4 | `roundabout` | *Chưa xử lý* (sẽ phát triển sau) |

### Các thông số cấu hình trong `RobotConfig.h`
- `VOTER_ACTIVATE_FRAMES = 5`: Số frame liên tiếp phải thấy biển báo để xác nhận.
- `VOTER_TIMEOUT_FRAMES = 100`: Nếu 100 frame không thấy lại, reset bộ vote.
- `FORCE_ANGLE = 25.0`: Góc bẻ lái cứng khi Force Turn.
- `HEADING_RESET_DEG = 56.0`: Góc IMU chênh lệch để xác nhận đã rẽ xong.
- `TURN_CONDITION_COVERAGE = 0.78`: Vùng đường phía dưới phải phủ 78% diện tích mới cho phép kích hoạt Force Turn.

---

## Build

```bash
mkdir -p build && cd build
cmake ..
make ucrv3 -j$(nproc)
sudo ./ucrv3
```

---

## The Fixed-Rate Control Tick

`main.cpp` chỉ gọi `MissionStateMachine::Update()` khi `steady_clock::now() >= next_control`. 
Mọi hằng số rate-limit (ví dụ `rate_limit = 10` trong PID) được hiểu là **10°/step**. Ở 10Hz, nó tương đương 100°/s. Không được thay đổi `CONTROL_PERIOD_MS` mà không re-tune toàn bộ alpha và rate limit.

---

## Verification Checklist (Trước khi chạy timed-run)

- [ ] `sudo ./ucrv3` boot thành công, in ra `[SignDetector] in 640x640 out ...`.
- [ ] Camera frame hiển thị, mask đường overlay đúng.
- [ ] Biển báo được vẽ bounding box (debug UI) khi đưa vào camera.
- [ ] Servo bẻ đúng hướng: PID ra âm → bánh quay trái, PID ra dương → bánh quay phải (kiểm tra hàm `ApplySteeringCalibration`).
- [ ] Khi có biển `turn_left`, xe giảm tốc, bẻ lái cứng 25°, đi đúng 56° (IMU) rồi trả lái về bám làn.
- [ ] Emergency stop: Ctrl-C dừng xe, motor = 0, servo = 0.

---

## Future Work (Round 3 Scope)

1. **Traffic Light**: Dừng xe khi đèn đỏ (cần nhận diện màu).
2. **Pedestrian Crossing**: Dừng xe nhường đường (cần model crosswalk hoặc dùng ToF).
3. **Parallel Parking**: `ParkController.h` sử dụng ToF và góc lùi.
4. **Roundabout**: Cần state machine riêng để nhường đường và đi vòng.
5. **Bridge Handling**: Xử lý khi mask đường bị mất ở hai bên (lan can cầu).
```
```

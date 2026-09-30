# 🧠 KNOWLEDGE BASE & TROUBLESHOOTING LOG
*Cập nhật: Sep 2026 — UCR 2026 Self-Driving Car Project*

Tài liệu lưu trữ các kinh nghiệm xương máu, lỗi hệ thống đã được khắc phục và các thông số kỹ thuật đã được kiểm chứng.

---

## 1. DOCKER & HỆ ĐIỀU HÀNH (LINUX HOST)

**1.1. Lỗi X11 GUI Authorization (Unable to open display :1)**
- **Triệu chứng:** Chạy code có `cv2.imshow()` hoặc `cv2.waitKey()` báo lỗi Qt plugin.
- **Nguyên nhân:** Docker container (root) bị Linux Host từ chối vẽ lên X server.
- **Cách khắc phục:** Mở terminal trên **máy Host thật** và chạy `xhost +local:root`. KHÔNG THỂ chạy từ trong Docker.
- **Tắt GUI hoàn toàn:** `export ENABLE_GUI=0` — tất cả hàm `imshow` sẽ bị bypass, ảnh debug sẽ được ghi ra thư mục `debug_frames/` thay thế.

**1.2. Lỗi File Permission (Bị khoá quyền)**
- **Triệu chứng:** File do Python tạo ra (dataset, ZIP) bị khoá — không thể copy hay xoá từ Host.
- **Nguyên nhân:** Docker chạy root (UID 0), file bên ngoài Host dùng UID 1000.
- **Cách khắc phục:** Chèn vào cuối mọi script sinh file:
  ```python
  import subprocess
  subprocess.run(["chmod", "-R", "777", "/workspace/my_code/dataset"], check=False)
  ```

**1.3. Clone / Sao lưu Docker Container**
- **Lưu container đang chạy thành Image mới:**
  ```bash
  docker ps                                           # Lấy container_id
  docker commit <container_id> <tên_image>:<tag>     # Tạo snapshot
  ```
- **Export Image sang file để chuyển máy (Offline):**
  ```bash
  docker save -o backup.tar <tên_image>   # Máy A: xuất ra file
  docker load -i backup.tar               # Máy B: nạp file vào
  ```
- **Đẩy lên Docker Hub để dùng từ xa:**
  ```bash
  docker tag <image_cũ> <username>/<image>:<tag>
  docker push <username>/<image>:<tag>
  ```

---

## 2. THƯ VIỆN ĐIỀU KHIỂN (ucr_lib.so)

**2.1. AVControl() — Chỉ nhận positional arguments**
- **Triệu chứng:** `AVControl(speed=x, angle=y)` → `TypeError: incompatible function arguments`
- **Nguyên nhân:** Binding C++ chỉ hỗ trợ positional float, không nhận keyword args.
- **Cách đúng:** `AVControl(float(speed), float(angle))`

**2.2. Tranh chấp điều khiển khi lái thủ công**
- **Vấn đề:** Nếu code vẫn gọi `GetStatus()` và `AVControl(0, 0)` khi lái Manual, game Unity bị giật.
- **Giải pháp:** Khi `--drive manual`, vòng lặp chỉ chứa `GetRaw()` và `time.sleep()`. Bỏ qua hoàn toàn `AVControl`.

---

## 3. PERCEPTION — YOLO SEGMENTATION

**3.1. Lựa chọn Model & Tradeoff Tốc độ/Độ chính xác**
- **Model hiện tại đang dùng:** `best7.pt` (tốt nhất, finetune nhiều vòng nhất)
- **Thứ tự ưu tiên weights:** `best7.pt` > `best6.pt` > ... > `best1.pt`
- **KHÔNG dùng `half=True` hay `quantize`:** Đã kiểm nghiệm thực tế — quantize làm xe chỉ qua được 5-7 checkpoint thay vì 10/10. Độ chính xác mask ở góc cua hẹp giảm quá nhiều.
- **Độ phân giải:** Để mặc định `imgsz=640`. Giảm xuống 320 làm mask kém chính xác ở ngã rẽ.

**3.2. Dynamic Frame-Skip**
- **Cơ chế:** Trên đường thẳng (`blended_error < 20`), bỏ qua 1 frame (`YOLO_SKIP=2`) để tăng gấp đôi FPS. Trước khi vào cua (`blended_error > 12`), chạy YOLO mọi frame (`YOLO_SKIP=1`).
- **Lý do:** PID/Steering angle cần nhịp thời gian ổn định. "Mở mắt sớm" trước cua là bắt buộc.

**3.3. Xử lý mask không hợp lệ (maycayv3.py)**
- **Coverage check:** Bỏ qua và tái sử dụng frame trước nếu `coverage < 0.05` (mất tín hiệu) hoặc `coverage > 0.90` (mask quá nhiễu).
- ```python
  coverage = float((seg_mask > 0).mean())
  mask_valid = 0.05 <= coverage <= 0.90
  ```

---

## 4. THUẬT TOÁN ĐIỀU HƯỚNG (Steering Algorithm)

**4.1. Vấn đề Ngã 3 Giả (Fake Intersection)**
- **Hiện tượng:** Xe bị văng ra ngoài khi gặp điểm giao nhau (đường mở rộng đột ngột). Mask phình to về 1 phía → tâm centerline bị kéo lệch → xe ngoặt theo hướng sai.
- **Giải pháp — LaneWidthEstimator với Rolling Baseline:**
  - `base_width` không fix cứng từ đầu, mà cập nhật liên tục (`0.95*base + 0.05*current`).
  - Chỉ nhận dạng là ngã 3 khi đường phình rộng hơn **40%** (`threshold=0.4`).
  - Khi `wide_change=True`: khoá góc lái về `[-5, 5]` và ép phanh mạnh (`curve_ratio=1.0`).
  - **Ngưỡng 40% thay vì 20%:** Ngưỡng 20% bị báo động giả liên tục trên các đoạn đường thẳng dài có sự chênh lệch nhỏ về độ rộng.

**4.2. Tham số tốc độ đã được kiểm nghiệm**
- `max_speed=45`, `min_speed=22–24` — Dải vận tốc an toàn nhất để qua toàn bộ 10 checkpoint.
- `min_speed=32` trở lên → xe văng ra ngoài ở Checkpoint 7 (cua quá gắt).
- Hệ số phanh `damping=0.55` (cua nhanh) — đổi thành `0.65` nếu thường xuyên văng cua.
- Tốc độ hội tụ nhanh: `speed = 0.4 * old + 0.6 * new` (thay vì 0.7/0.3 ù lì hơn).

**4.3. PID Controller (maycayv2.py và maycayv3.py)**
- **Tham số đã kiểm nghiệm:**
  ```
  kp=0.32, ki=0.003, kd=0.75
  integral_limit=3.0, derivative_alpha=0.55
  output_limit=20.0 (tanh saturation)
  rate_limit=8.0 (deg/frame) — KHÔNG để 3.0 sẽ bị understeering ở cua gắt
  adaptive_d_boost=0.10, adaptive_d_cap=2.5
  ```
- **rate_limit=4.0:** Tốt cho đường mượt nhưng **không kịp** bẻ lái ở góc cua đột ngột. Cần ít nhất 8.0.
- **Deadband:** `if abs(blended_error) < 2.0: error = 0` — tránh xe lắc lư li ti trên đường thẳng.

**4.4. Centerline Extraction — Vectorization**
- Thay vì vòng lặp Python qua 480 dòng ảnh (~5ms/frame):
  ```python
  y_coords, x_coords = np.nonzero(gray)
  unique_y, inv = np.unique(y_coords, return_inverse=True)
  mean_x = (np.bincount(inv, weights=x_coords) / np.bincount(inv)).astype(int)
  ```
  Tốc độ ~0.1ms/frame — nhanh gấp 50 lần.

**4.5. Lap Time Benchmarks**
| Phiên bản | Thời gian | Ghi chú |
|---|---|---|
| Baseline | ~3m00s | Không tối ưu |
| Sau vectorization + Frame-skip | ~2m57s | Cải thiện nhỏ |
| Mục tiêu tiếp theo | < 2m50s | Cần PID + tốc độ cao |

---

## 5. LABELS & TRAINING DATA

**5.1. Tham số YOLO Training Quan Trọng**
- `fliplr=0.0` — **SINH TỬ.** Tắt hoàn toàn. Lật ngang làm model học sai góc cua.
- `hsv_v=0.4` — Nhiễu độ sáng. Cực kỳ quan trọng cho Night Map và Snow Map.
- `optimizer='AdamW'`, `lr0=1e-3`, `lrf=1e-5`, `patience=15`.
- **Finetune từ model cũ** luôn tốt hơn train từ đầu (huấn luyện thêm 50 epochs trên data mới).

**5.2. Sự thất bại của Point-Prompt SAM**
- SAM + 1 điểm duy nhất → segment vụn vỡ trên môi trường game 3D.
- **Giải pháp:** Rải lưới điểm (`Grid`) ở toàn bộ nửa dưới màn hình + lọc sơ bộ bằng HSV trước để không chấm lên tuyết/cỏ.

**5.3. Lỗi đường dẫn data.yaml trên Colab**
- **Nguyên tắc vàng:** `glob.glob('**/*.jpg', recursive=True)` thay vì fix cứng path cấp 1.

---

## 6. FILE & MODULE HIỆN TẠI

| File | Mô tả |
|---|---|
| `maycay.py` | Bản baseline ổn định (YOLO + Weighted Centerline) |
| `maycayv2.py` | Bản PID + YOLO, speed lanh lẹ (`rate_limit=8`) |
| `maycayv3.py` | Bản PID + YOLO + Coverage gate + CSV Log |
| `ucr_lib.so` | Thư viện kết nối game Unity (không sửa được) |
| `weights/best7.pt` | Model tốt nhất hiện tại |
| `pid_log_v3.csv` | Log PID của maycayv3 — dùng để vẽ biểu đồ |

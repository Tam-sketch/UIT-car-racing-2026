# 🧠 KNOWLEDGE BASE & TRABOULESHOOTING LOG
Tài liệu lưu trữ các kinh nghiệm xương máu và những lỗi hệ thống đã được khắc phục.

## 1. CÁC LỖI LIÊN QUAN ĐẾN DOCKER VÀ QUYỀN TRUY CẬP (LINUX HOST)

**1.1. Lỗi X11 GUI Authorization (Unable to open display :1)**
- **Triệu chứng:** Chạy các code có gọi OpenCV `cv2.imshow()` hoặc `cv2.waitKey()` báo lỗi Qt plugin không khởi tạo được.
- **Nguyên nhân:** Docker container (chạy bằng quyền root) bị Linux Host từ chối vẽ giao diện ra màn hình (X server).
- **Cách khắc phục triệt để:** Bắt buộc phải mở terminal trên **máy Linux thật (Host)** và chạy lệnh `xhost +local:root`. Hoàn toàn KHÔNG THỂ chạy lệnh này từ bên trong Docker. Nếu không cần xem hình, sử dụng biến môi trường `export ENABLE_GUI=0` để bỏ qua mọi hàm GUI.

**1.2. Lỗi File Permission (Bị khoá quyền)**
- **Triệu chứng:** Các file do script Python sinh ra (như ảnh dataset, file ZIP) bị dính biểu tượng khoá ổ khoá. Người dùng bên ngoài Host (hoặc VS Code) không thể copy, xoá, sửa được.
- **Nguyên nhân:** Docker chạy ở chế độ root, nên mọi file tạo ra mặc định owner là root (UID 0), trong khi user bên ngoài thường có UID 1000.
- **Cách khắc phục:** Trong các file script như `auto_label.py`, `prepare_dataset.py`, `collect_data.py`, luôn phải chèn thêm khối mã sau ở cuối chương trình:
  ```python
  import subprocess
  subprocess.run(["chmod", "-R", "777", "/workspace/my_code/dataset"], check=False)
  ```

## 2. NHỮNG PHÁT HIỆN TRONG QUÁ TRÌNH LABEL & XỬ LÝ ẢNH

**2.1. Vấn đề "Xe tự label vào thân chính mình" (Car Body Intrusion)**
- **Hiện tượng:** Khi chạy `auto_label.py` bằng màu sắc, vùng thân xe xanh thẫm thường xuyên bị tô xanh nhầm thành mặt đường.
- **Nguyên nhân cốt lõi:** Developer lấy ảnh screenshot từ game (Third-person view) để debug. Tuy nhiên, luồng dữ liệu thực tế đẩy qua hàm `GetRaw()` của `ucr_lib.so` lại là **First-person view** (Camera gắn ngay mũi xe, không hề thấy thân xe).
- **Bài học:** Chỉ debug và phân tích màu sắc trực tiếp trên ảnh lấy từ hàm `GetRaw()`.

**2.2. Sự thất bại của Point-Prompt SAM trên Map Game**
- **Hiện tượng:** Dùng thuật toán SAM của Meta và chấm (Point prompt) 1 điểm vào giữa đường, kết quả SAM chỉ segment ra 1 mảng vụn vỡ lổm chổm chứ không loang ra hết mặt đường.
- **Nguyên nhân:** Môi trường đồ hoạ Game 3D thường không có các đường viền texture sắc nét như đời thực.
- **Giải pháp:** Phải Prompt SAM bằng một **lưới điểm toạ độ (Grid)** rải đều ở toàn bộ nửa dưới màn hình. Khi đó SAM sẽ gộp tất cả các điểm này lại và cho ra 1 mảng đường 100% hoàn hảo, tự né các vật cản như gốc cây.

**2.3. Bất đồng bộ trong ucr_lib.so**
- **Vấn đề:** Khi bạn lái xe thủ công (Manual), nếu code Python vẫn liên tục gọi `GetStatus()` và `AVControl(0, 0)`, game Unity sẽ bị cà giật do tranh chấp luồng điều khiển (Python gửi lệnh tốc độ 0, bàn phím gửi lệnh tốc độ 20).
- **Giải pháp:** Chế độ `collect_data.py --drive manual` phải được cách ly hoàn toàn, vòng lặp chỉ chứa duy nhất hàm `GetRaw()` và `time.sleep()`. Bỏ qua hoàn toàn hàm GetStatus.

## 3. CÁC THAM SỐ VÀNG TRONG YOLO TRAINING

- `fliplr=0.0`: **Sinh tử.** Tuyệt đối tắt tính năng lật ngang ảnh khi augment data cho xe tự lái. Lật ngang sẽ khiến biển báo, mép cua bị đảo ngược, làm model học sai lệch góc cua.
- `hsv_v=0.4`: Augment nhiễu loạn về ánh sáng, cực kỳ quan trọng cho Night Map và Snow Map, giúp xe không bị mù khi đi vào góc khuất.
- Lỗi đường dẫn Data.yaml trên Colab: Do sự chênh lệch cấu trúc thư mục khi nén ZIP. **Nguyên tắc vàng:** Giải nén `unzip -d /content/raw_data/` và dùng `glob.glob('**/*.jpg', recursive=True)` để hệ thống tự tìm ảnh thay vì fix cứng đường dẫn thư mục cấp 1.

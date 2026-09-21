# 🏎️ UIT CAR RACING 2026 - AUTONOMOUS DRIVING

Dự án phát triển xe tự lái cho cuộc thi UIT Car Racing 2026. Phiên bản này đã được nâng cấp hoàn toàn để chạy mượt mà trên **Linux Native (Ubuntu/Docker)**, không còn phụ thuộc vào Windows hay WSL2, giải quyết dứt điểm các lỗi kết nối mạng.

---

## 🚀 Tính năng nổi bật của bản cập nhật (Tháng 9/2026)

- Hỗ trợ Native Linux Socket: Chạy thẳng game trên Linux không cần socat.
- X11 Forwarding: Hỗ trợ hiển thị giao diện Camera (OpenCV) trực tiếp từ trong Docker ra màn hình host.
- Auto-Label thông minh: Hỗ trợ tự động label bằng màu sắc (HSV) cho map tuyết hoặc SAM (Segment Anything Model) độ chính xác cao.
- Tự động mở khoá quyền: Khắc phục lỗi "Permission denied" do Docker root sinh ra.
- Train Model tối ưu: Script đóng gói data tự động cho Google Colab.

---

## 🛠️ Hướng dẫn cài đặt & Chạy xe

### 1. Khởi động môi trường & Game
1. Mở Terminal thật trên máy bạn, chạy lệnh cấp quyền vẽ giao diện cho Docker:
   ```bash
   xhost +local:root
   ```
2. Khởi động file thực thi của Game Unity (ví dụ: `V1_demo_Linux.x86_64`). Bật sang chế độ **Autonomous Mode**.
3. Mở Terminal trong Docker, vào thư mục code:
   ```bash
   cd /workspace/my_code
   ```

### 2. Chạy xe tự lái (Inference)
Chạy script `maycay.py`. Script này sẽ tự động tải file trọng số `best.pt` mới nhất trong thư mục weights và điều khiển xe.
```bash
python maycay.py
```
Nếu bạn gặp lỗi hiển thị đồ hoạ (GUI) do xhost chưa nhận, bạn có thể tắt GUI bằng lệnh:
```bash
export ENABLE_GUI=0
python maycay.py
```

---

## 📦 Quy trình tự tạo Model cho Map Mới (Data Pipeline)

Nếu xe chạy qua map mới (vd: map tuyết, ban đêm) và bị mù, bạn cần làm theo 4 bước sau:

**Bước 1: Thu thập ảnh (Collect Data)**
Bật game sang chế độ Manual Mode. Khởi động file thu thập để lấy ảnh tự động khi bạn lái xe bằng tay:
```bash
python collect_data.py --scene snow_map --drive manual --max 1200
```

**Bước 2: Đóng gói ảnh thô (Raw Data)**
Bỏ qua bước label thủ công tại máy tính. Do địa hình phức tạp, chúng ta sẽ đẩy thẳng toàn bộ ảnh thô lên Google Colab để dùng trí tuệ nhân tạo (SAM) label với độ chính xác tuyệt đối.
Tại terminal, chạy lệnh nén thư mục ảnh gốc:
```bash
cd /workspace/my_code/dataset/raw
zip -r Raw_Snow_Map.zip snow_map/
```

**Bước 3: Auto-Label bằng SAM & Train trên Google Colab**
1. Tải file `Raw_Snow_Map.zip` lên thư mục `Train_UCR2026` trên Google Drive.
2. Mở Colab (chọn GPU T4) và chạy script tự động rải lưới SAM để label và train (Xem file `WORKFLOW.md` để lấy đoạn code chuẩn). Quá trình này sẽ mất khoảng 20 phút.

**Bước 4: Thay thế Model**
Tải file `best.pt` trên Drive về, chép đè vào thư mục `/workspace/my_code/Road_Seg_Model/modelYolo/weights/best3.pt` và chạy lại `maycay.py`.

---
*Developed for UIT Car Racing 2026*

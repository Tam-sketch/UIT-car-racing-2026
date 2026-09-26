# 🏎️ UIT CAR RACING 2026 - AUTONOMOUS DRIVING

Dự án phát triển xe tự lái cho cuộc thi UIT Car Racing 2026. Phiên bản này đã được nâng cấp hoàn toàn để chạy mượt mà trên **Linux Native (Ubuntu/Docker)**
---

## 🚀 Tính năng

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
2. Attach the docker to VScode:
Run this line on bash / powershell to attach the docker environment to VScode.
```bash
docker run --name it-car -it -p 11000:11000 --network="host" -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --gpus all <imageid>
```
3. Khởi động file thực thi của Game Unity (ví dụ: `V1_demo_Linux.x86_64`). Bật sang chế độ **Autonomous Mode**.
4. Mở Terminal trong Docker, vào thư mục code:
   ```bash
   cd /workspace
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
python collect_data.py --scene map1 --drive manual --max 1200
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
2. Mở Colab (chọn GPU T4) và chạy script tự động rải lưới SAM để label và train (Xem file `WORKFLOW.md` để lấy đoạn code chuẩn).

**Bước 4: Thay thế Model**
Tải file `best.pt` trên Drive về, chép đè vào thư mục `/workspace/my_code/Road_Seg_Model/modelYolo/weights/best3.pt` và chạy lại `maycay.py`.

---
*Developed for UIT Car Racing 2026*

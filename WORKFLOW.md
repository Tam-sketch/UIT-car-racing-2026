# 🔄 WORKFLOW: QUY TRÌNH PHÁT TRIỂN XE TỰ LÁI UCR 2026
*Cập nhật: Sep 2026*

---

## TỔNG QUAN HỆ THỐNG

```
[Game Unity] → GetRaw() → [YOLO Segmentation] → [Steering Algorithm] → AVControl() → [Game Unity]
                                                          ↑
                                               (PID hoặc Weighted Error)
```

**3 File điều khiển xe:**
- `maycay.py` — Bản ổn định, dùng Weighted Centerline (không PID)
- `maycayv2.py` — PID + speed lanh lẹ (đề xuất thử nghiệm tốc độ)
- `maycayv3.py` — PID + Coverage Gate + CSV Log (đề xuất cho phân tích chuyên sâu)

---

## PHASE 1: CHẠY XE (Ngày thi đấu)

### Chạy nhanh
```bash
# Vào thư mục
cd /workspace/my_code

# Cho phép GUI (chạy trên máy HOST, không phải Docker)
xhost +local:root

# Chạy xe (chọn 1 trong 3)
python maycay.py         # Ổn định nhất
python maycayv2.py       # PID, nhanh hơn
python maycayv3.py       # PID + log CSV (chậm hơn một chút do ghi log)

# Tắt GUI nếu bị lỗi xhost
export ENABLE_GUI=0 && python maycay.py
```

### Thay đổi model nhanh
```bash
# Xem danh sách model hiện có
ls Road_Seg_Model/modelYolo/weights/
# best1.pt  best2.pt  ...  best7.pt  (best7 tốt nhất)
```
Model được load tự động theo thứ tự ưu tiên trong code. Nếu muốn chỉ định:
```bash
python maycayv2.py --model Road_Seg_Model/modelYolo/weights/best7.pt
```

### Đọc log PID sau khi chạy (maycayv3)
```bash
# Xem 10 dòng cuối log
tail -20 /workspace/my_code/pid_log_v3.csv

# Vẽ biểu đồ bằng Python nhanh
python3 -c "
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv('/workspace/my_code/pid_log_v3.csv')
df[['error','angle','speed']].plot(figsize=(14,4))
plt.tight_layout(); plt.savefig('pid_plot.png')
print('Saved pid_plot.png')
"
```

---

## PHASE 2: THU THẬP DỮ LIỆU MỚI

```bash
# Lái xe thủ công để thu thập ảnh
export ENABLE_GUI=0
python collect_data.py --scene <ten_map> --drive manual --max 1200

# Ảnh sẽ được lưu vào: dataset/raw/<ten_map>/
```
> ⚠️ **Quan trọng:** Ở chế độ manual, script KHÔNG gọi `AVControl()` để tránh tranh chấp với bàn phím.

---

## PHASE 3: AUTO-LABEL & TRAINING (Google Colab)

### Bước 1 — Đóng gói data
```bash
cd /workspace/my_code/dataset/raw
zip -r <TenMap>.zip <ten_map>/
```
Upload file ZIP lên Google Drive vào thư mục `Train_UCR2026/`.

### Bước 2 — Chạy Colab
Paste script dưới đây vào Colab (GPU T4). Script này tự động label bằng SAM và train YOLO.

```python
from google.colab import drive
import os, cv2, glob
import numpy as np
drive.mount('/content/drive')

!pip install -q ultralytics
from ultralytics import SAM, YOLO

# ── PHẦN 1: AUTO-LABEL VỚI SAM ──────────────────────────────────────────────
!unzip -q "/content/drive/MyDrive/Train_UCR2026/<TenMap>.zip" -d /content/raw_data/
sam_model = SAM('sam_b.pt')
os.makedirs('/content/UCR2026_Dataset/images/train', exist_ok=True)
os.makedirs('/content/UCR2026_Dataset/labels/train', exist_ok=True)

# recursive=True để tìm ảnh trong mọi folder con
img_paths = glob.glob('/content/raw_data/**/*.jpg', recursive=True)

for p in img_paths:
    img = cv2.imread(p); h, w = img.shape[:2]
    # Lọc sơ bộ bằng HSV: chỉ chấm điểm vào vùng tối+có màu (mặt đường nhựa)
    # Tránh SAM bị lừa bởi vùng tuyết trắng hoặc bầu trời
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v, s = hsv[:,:,2], hsv[:,:,1]
    points, labels = [], []
    for y in range(h*2//3, h, 20):
        for x in range(10, w-10, 40):
            if v[y,x] < 160 and s[y,x] > 20:  # Tối + có màu = đường nhựa
                points.append([x,y]); labels.append(1)
    if not points: points = [[w//2, h-5]]; labels = [1]  # Fallback

    res = sam_model(img, points=[points], labels=[labels], device=0, verbose=False)
    mask = (res[0].masks.data[0].cpu().numpy() * 255).astype(np.uint8)

    # Khử lỗ hổng nhỏ trong mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > h*w*0.05:
            c = c.reshape(-1,2).astype(np.float32)
            c[:,0] /= w; c[:,1] /= h
            name = os.path.basename(p)
            cv2.imwrite(f'/content/UCR2026_Dataset/images/train/{name}', img)
            with open(f'/content/UCR2026_Dataset/labels/train/{name[:-4]}.txt','w') as f:
                f.write("0 " + " ".join([f"{x:.4f} {y:.4f}" for x,y in c]) + "\n")

# ── PHẦN 2: CẤU HÌNH ────────────────────────────────────────────────────────
with open('/content/UCR2026_Dataset/data.yaml','w') as f:
    f.write("path: /content/UCR2026_Dataset\ntrain: images/train\nval: images/train\nnc: 1\nnames:\n  0: road")

# ── PHẦN 3: TRAIN (Finetune từ model tốt nhất hiện có) ───────────────────────
model = YOLO('/content/drive/MyDrive/Train_UCR2026/best7.pt')
model.train(
    data='/content/UCR2026_Dataset/data.yaml',
    epochs=50, imgsz=640, batch=16, workers=2, cache='disk', device=0,
    patience=15, optimizer='AdamW', lr0=1e-3, lrf=1e-5,
    fliplr=0.0,   # SINH TỬ: Tắt lật ảnh — xe học sai góc cua nếu bật
    hsv_v=0.4,    # Tăng nhiễu độ sáng — quan trọng cho Night Map / Snow Map
    project='/content/drive/MyDrive/Train_UCR2026', name='new_map_model'
)
```

### Bước 3 — Tải model về và cập nhật
1. Tải `best.pt` từ `Train_UCR2026/new_map_model/weights/best.pt` trên Drive.
2. Đặt vào `/workspace/my_code/Road_Seg_Model/modelYolo/weights/best8.pt` (đặt tên tăng dần).
3. Kiểm tra: `python maycay.py --model Road_Seg_Model/modelYolo/weights/best8.pt`

---

## PHASE 4: TINH CHỈNH THAM SỐ ĐIỀU KHIỂN

### Thông số Tốc độ
```python
# Trong maycay.py / maycayv2.py
max_speed = 45   # Tốc độ tối đa đường thẳng
min_speed = 22   # Tốc độ tối thiểu (giới hạn để xe qua cua gắt như CP7)
# ⚠️ min_speed > 32 → xe văng ra ngoài ở Checkpoint 7
```

### Thông số PID (maycayv2, maycayv3)
```python
SteeringPID(
    kp=0.32,          # Tăng → phản xạ mạnh hơn, dễ dao động
    ki=0.003,         # Giữ nhỏ để tránh drift
    kd=0.75,          # Tăng → chống overshoot ở góc cua
    rate_limit=8.0,   # ⚠️ KHÔNG để < 6.0 → xe không kịp bẻ lái vào cua gắt
    output_limit=20.0 # Góc lái tối đa (độ)
)
```

### Quy trình debug bằng CSV log
```bash
# Sau khi chạy maycayv3.py, phân tích log
python3 -c "
import pandas as pd
df = pd.read_csv('/workspace/my_code/pid_log_v3.csv')
print('Góc lái trung bình:', df['angle'].abs().mean())
print('Sai số trung bình:', df['error'].abs().mean())
print('Tốc độ trung bình:', df['speed'].mean())
print('Số frame lái cực đoan (>15 độ):', (df['angle'].abs() > 15).sum())
"
```

---

## PHASE 5: BẢO TRÌ & BACKUP

### Backup model quan trọng
```bash
# Backup toàn bộ weights
cp -r /workspace/my_code/Road_Seg_Model/modelYolo/weights/ /workspace/weights_backup_$(date +%Y%m%d)/
```

### Clone Docker container
```bash
docker ps                                          # Lấy container ID
docker commit <id> ucr2026_backup:$(date +%Y%m%d) # Tạo snapshot
docker save -o ucr2026_backup.tar ucr2026_backup:$(date +%Y%m%d) # Export ra file
```

---

## QUICK REFERENCE — LỖI THƯỜNG GẶP

| Lỗi | Nguyên nhân | Cách sửa |
|---|---|---|
| `TypeError: AVControl() got unexpected keyword argument` | Dùng `AVControl(speed=x, angle=y)` | Sửa thành `AVControl(float(speed), float(angle))` |
| `Unable to open display :1` | Thiếu xhost | Chạy `xhost +local:root` trên Host |
| `SteeringPID() got unexpected keyword argument 'd_term_clamp'` | Tham số cũ không còn trong class | Xoá `d_term_clamp`, dùng `adaptive_d_boost` |
| Xe văng cua liên tục | `min_speed` quá cao hoặc `rate_limit` quá thấp | Giảm `min_speed=22`, tăng `rate_limit=8` |
| Brake liên tục dù đường thẳng | `LaneWidthEstimator` báo động giả Ngã 3 | Tăng ngưỡng `threshold=0.4`, bật rolling baseline |
| YOLO bị `half is deprecated` | Dùng `half=True` trong predict | Bỏ tham số đó, để mặc định |

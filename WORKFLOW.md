# 🔄 WORKFLOW: QUY TRÌNH HUẤN LUYỆN MODEL YOLO CHO MAP MỚI

Tài liệu này hướng dẫn chi tiết quy trình từ A-Z để giúp chiếc xe thích nghi với một sa hình hoàn toàn mới (ví dụ: Map Tuyết, Map Đêm).

## 1. THU THẬP DỮ LIỆU GỐC (DATA COLLECTION)

Mục đích: Lấy những khung hình camera thực tế từ trong game.

- **Thao tác trong Game**: Bật chế độ Manual Mode.
- **Lệnh thực thi**:
  ```bash
  export ENABLE_GUI=0   # Tat GUI neu bi loi xhost
  python collect_data.py --scene snow_map --drive manual --max 1200
  ```
- **Lưu ý**: Script `collect_data.py` ở chế độ manual chỉ gọi hàm `GetRaw()` (chỉ lấy ảnh). Tuyệt đối không gọi `GetStatus()` và `AVControl()` để tránh hiện tượng giật lag tranh chấp quyền điều khiển với bàn phím của bạn.

## 2. AUTO-LABEL (TẠO MASK PHÂN ĐOẠN)

Bạn có 2 cách để Label dữ liệu tuỳ theo độ phức tạp của Map.

### CÁCH A: Dùng thuật toán màu sắc HSV (Nhanh - Tại Local)
Áp dụng tốt nếu mặt đường có màu sắc khá đặc trưng (như đường tuyết hơi ngả xanh tối).
```bash
# Don dep rác nếu có
rm -rf dataset/mask/snow_map/
# Chạy label
python auto_label.py --scene snow_map --mode snow
# Đóng gói Dataset ra ZIP
python prepare_dataset.py --scene snow_map
```

### CÁCH B: Dùng AI SAM (Siêu chuẩn - Khuyên dùng - Trên Colab)
Áp dụng khi thuật toán màu bị sai hoặc map quá phức tạp. Bạn nén toàn bộ thư mục `/workspace/my_code/dataset/raw/snow_map/` thành file `Raw_Snow_Map.zip` và upload lên Google Drive. Toàn bộ quá trình từ Label đến Train sẽ chạy tự động trên Colab bằng code ở Mục 3.

## 3. TRAIN YOLO TRÊN GOOGLE COLAB

Copy đoạn script Python sau dán vào Colab (yêu cầu bật GPU T4). Script này tích hợp luôn tính năng Auto-label bằng SAM cực kỳ chính xác.

```python
from google.colab import drive
import os, cv2, glob, numpy as np
drive.mount('/content/drive')

!pip install ultralytics
from ultralytics import SAM, YOLO

# --- PHẦN 1: TỰ ĐỘNG LABEL BẰNG SAM (Bỏ qua nếu đã làm Cách A) ---
!unzip -q "/content/drive/MyDrive/Train_UCR2026/Raw_Snow_Map.zip" -d /content/raw_data/
sam_model = SAM('sam_b.pt')
os.makedirs('/content/UCR2026_Dataset/images/train', exist_ok=True)
os.makedirs('/content/UCR2026_Dataset/labels/train', exist_ok=True)

# LUU Y: de recursive=True de quet duoc tat ca cac file nam trong folder con
img_paths = glob.glob('/content/raw_data/**/*.jpg', recursive=True) 

for p in img_paths:
    img = cv2.imread(p); h, w = img.shape[:2]
    # Rải lưới toạ độ ở nửa dưới màn hình để bắt mảng đường
    points, labels = [], []
    for y in range(h*2//3, h, 20):
        for x in range(w//4, 3*w//4, 40):
            points.append([x, y]); labels.append(1)
            
    res = sam_model(img, points=[points], labels=[labels], device=0, verbose=False)
    mask = (res[0].masks.data[0].cpu().numpy() * 255).astype(np.uint8)
    
    # Khử nhiễu hổng (bông tuyết)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > (h*w*0.05):
            c = c.reshape(-1, 2).astype(np.float32)
            c[:, 0] /= w; c[:, 1] /= h
            name = os.path.basename(p)
            cv2.imwrite(f'/content/UCR2026_Dataset/images/train/{name}', img)
            with open(f'/content/UCR2026_Dataset/labels/train/{name.replace(".jpg", ".txt")}', 'w') as f:
                f.write("0 " + " ".join([f"{x:.4f} {y:.4f}" for x, y in c]) + "\n")

# --- PHẦN 2: TẠO FILE CẤU HÌNH ---
with open('/content/UCR2026_Dataset/data.yaml', 'w') as f:
    f.write("path: /content/UCR2026_Dataset\ntrain: images/train\nval: images/train\nnc: 1\nnames:\n  0: road")

# --- PHẦN 3: TRAIN MÔ HÌNH ---
model = YOLO('/content/drive/MyDrive/Train_UCR2026/best3.pt') # Finetune từ model cũ
model.train(
    data='/content/UCR2026_Dataset/data.yaml',
    epochs=50, imgsz=640, batch=16, workers=2, cache='disk', device=0,
    patience=15, optimizer='AdamW', lr0=1e-3, lrf=1e-5, 
    fliplr=0.0,    # QUAN TRONG: Tắt lật ảnh để xe không học sai góc cua
    hsv_v=0.4,     # QUAN TRONG: Tăng nhiễu độ sáng, rất tốt cho map tuyết
    project='/content/drive/MyDrive/Train_UCR2026', name='snow_map_sam_model'
)
```

## 4. KIỂM THỬ THỰC TẾ

1. Tải file `best.pt` mới sinh ra từ thư mục `Train_UCR2026/snow_map_sam_model/weights/`.
2. Đổi tên thành `best3.pt` và ghi đè vào máy tính tại: `/workspace/my_code/Road_Seg_Model/modelYolo/weights/`
3. Chạy `python maycay.py` và quan sát xe bám đường.

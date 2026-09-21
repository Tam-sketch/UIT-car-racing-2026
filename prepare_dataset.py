#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prepare_dataset.py  (v2 - 2026)
---------------------------------
Dong goi dataset tu raw anh + label .txt thanh cau truc chuan YOLO,
chia 80% train / 20% val, va nen thanh UCR2026_<scene>_Dataset.zip de
upload len Google Colab / Kaggle.

Cach dung:
    python prepare_dataset.py --scene snow_map
    python prepare_dataset.py --scene snow_map --val-ratio 0.15 --out-name UCR2026_Snow
"""

import os
import sys
import shutil
import random
import argparse
import zipfile
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description="Dong goi dataset YOLO chuan")
    p.add_argument("--scene",     type=str, default="snow_map",
                   help="Ten scene (phai co dataset/raw/<scene> va dataset/mask/<scene>)")
    p.add_argument("--val-ratio", type=float, default=0.20,
                   help="Ti le anh Validation (mac dinh: 0.20 = 20%%)")
    p.add_argument("--out-name",  type=str, default="",
                   help="Ten file ZIP dau ra (mac dinh: UCR2026_<scene>_Dataset)")
    p.add_argument("--seed",      type=int, default=42,
                   help="Seed ngau nhien de ket qua lap lai (mac dinh: 42)")
    p.add_argument("--keep-empty", action="store_true",
                   help="Giu lai anh co file label rong (empty label - hoc 'khong co gi')")
    return p.parse_args()


def main():
    args = parse_args()

    raw_dir    = os.path.join(SCRIPT_DIR, "dataset", "raw",  args.scene)
    mask_dir   = os.path.join(SCRIPT_DIR, "dataset", "mask", args.scene)
    output_dir = os.path.join(SCRIPT_DIR, "dataset", "yolo_format", args.scene)
    zip_name   = args.out_name or f"UCR2026_{args.scene}_Dataset"
    zip_path   = os.path.join(SCRIPT_DIR, zip_name + ".zip")

    # Kiem tra thu muc dau vao
    if not os.path.isdir(raw_dir):
        print(f"[LOI] Khong tim thay thu muc anh raw: {raw_dir}")
        print(f"      Hay chay collect_data.py --scene {args.scene} truoc!")
        return
    if not os.path.isdir(mask_dir):
        print(f"[LOI] Khong tim thay thu muc label: {mask_dir}")
        print(f"      Hay chay auto_label.py --scene {args.scene} truoc!")
        return

    # Lay danh sach anh co label di kem
    all_images = [f for f in os.listdir(raw_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    paired = []      # Anh co label polygon thuc su
    empty_paired = []  # Anh co file label rong (empty)
    missing = []     # Anh khong co label nao ca

    for img_name in all_images:
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        txt_path = os.path.join(mask_dir, txt_name)

        if not os.path.exists(txt_path):
            missing.append(img_name)
            continue

        # Kiem tra file label rong hay co noi dung
        if os.path.getsize(txt_path) == 0:
            empty_paired.append(img_name)
        else:
            paired.append(img_name)

    print("\n" + "=" * 60)
    print(f"  DONG GOI DATASET - UIT CAR RACING 2026")
    print(f"  Scene     : {args.scene}")
    print(f"  Tong anh  : {len(all_images)}")
    print(f"  Co label  : {len(paired)}")
    print(f"  Empty lbl : {len(empty_paired)}")
    print(f"  Thieu lbl : {len(missing)}")
    print("=" * 60)

    if missing:
        print(f"\n[CANH BAO] {len(missing)} anh chua co label (se bo qua).")

    # Quyet dinh bo du lieu de dong goi
    working_set = paired[:]
    if args.keep_empty:
        working_set += empty_paired
        print(f"[INFO] Da bao gom {len(empty_paired)} anh empty label (flag --keep-empty).")

    if len(working_set) < 10:
        print("[LOI] Qua it du lieu! Can it nhat 10 anh co label.")
        return

    # Tao cau truc thu muc YOLO
    print(f"\n[INFO] Xoa thu muc cu (neu co)...")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    # Xao tron va chia
    random.seed(args.seed)
    random.shuffle(working_set)

    val_count   = max(1, int(len(working_set) * args.val_ratio))
    train_imgs  = working_set[val_count:]
    val_imgs    = working_set[:val_count]

    print(f"[INFO] Chia du lieu: {len(train_imgs)} train / {len(val_imgs)} val")

    def copy_pair(img_list, split_name):
        copied = 0
        for img_name in img_list:
            txt_name = os.path.splitext(img_name)[0] + ".txt"
            src_img  = os.path.join(raw_dir,  img_name)
            src_txt  = os.path.join(mask_dir, txt_name)
            dst_img  = os.path.join(output_dir, 'images', split_name, img_name)
            dst_txt  = os.path.join(output_dir, 'labels', split_name, txt_name)
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_txt, dst_txt)
            copied += 1
        return copied

    print("[INFO] Dang copy TRAIN...")
    t_count = copy_pair(train_imgs, 'train')
    print("[INFO] Dang copy VAL...")
    v_count = copy_pair(val_imgs, 'val')

    # Tao file data.yaml
    yaml_content = f"""\
# UIT CAR RACING 2026 - Dataset: {args.scene}
# Tao luc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

path: /content/{zip_name}  # Duong dan tren Google Colab (thay doi neu can)
train: images/train
val:   images/val

nc: 1  # So luong class
names:
  0: road
"""
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print("[INFO] Da tao file data.yaml")

    # Tao file README nhanh
    readme_content = f"""\
# {zip_name}
Scene    : {args.scene}
Train    : {t_count} anh
Val      : {v_count} anh
Total    : {t_count + v_count} anh
Tao luc  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Class    : 0 = road (mat duong)

## Cach dung tren Google Colab
```python
from ultralytics import YOLO

# Giai nen
import zipfile
with zipfile.ZipFile('/content/drive/MyDrive/{zip_name}.zip', 'r') as z:
    z.extractall('/content/{zip_name}')

# Train
model = YOLO('/content/drive/MyDrive/best3.pt')  # Bat dau tu model cu
model.train(
    data='/content/{zip_name}/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    fliplr=0.0,      # QUAN TRONG: Khong lat anh ngang
    hsv_v=0.4,       # Augment do sang (tot cho map tuyet co anh sang dao)
    workers=2,
    cache='disk',
)
```
"""
    with open(os.path.join(output_dir, 'README.md'), 'w') as f:
        f.write(readme_content)

    # Nen thanh ZIP
    print(f"\n[INFO] Dang nen thanh {zip_path} ...")
    if os.path.exists(zip_path):
        os.remove(zip_path)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                arc_name = os.path.relpath(abs_path, os.path.dirname(output_dir))
                zf.write(abs_path, arc_name)

    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"\n{'=' * 60}")
    print(f"  DONG GOI HOAN TAT!")
    print(f"  Tap Train  : {t_count} anh")
    print(f"  Tap Val    : {v_count} anh")
    print(f"  File ZIP   : {zip_path}")
    print(f"  Kich thuoc : {zip_size_mb:.1f} MB")
    print(f"{'=' * 60}")
    print(f"\nBuoc tiep theo:")
    print(f"  1. Upload file '{zip_name}.zip' len Google Drive")
    print(f"  2. Xem README.md trong ZIP de co code train mau tren Colab")


    import subprocess
    try:
        subprocess.run(["chmod", "-R", "777", "/workspace/my_code/dataset"], check=False)
        subprocess.run(["chmod", "777", zip_path], check=False)
        print("[INFO] Da tu dong mo khoa quyen truy cap cho file ZIP va thu muc dataset.")
    except:
        pass

if __name__ == "__main__":
    main()

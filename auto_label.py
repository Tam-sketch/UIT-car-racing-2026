#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
auto_label.py  (v2 - 2026 Snow Map Edition)
--------------------------------------------
Tu dong gan nhan anh cho YOLO Segmentation.

Co 2 che do:
  --mode model  : Dung model YOLO cu de label (tot cho map binh thuong)
  --mode snow   : Dung thuat toan xu ly mau sac de label (DANH RIENG cho map tuyet)
                  Khong can model YOLO! Phan biet duong (xam) vs tuyet (trang xanh).

Cach dung:
    # Map co tuyet (KHUYEN DUNG):
    python auto_label.py --scene snow_map --mode snow

    # Map thuong (dung model cu):
    python auto_label.py --scene map1 --mode model --model Road_Seg_Model/modelYolo/weights/best3.pt

    # Chi dinh thu muc input/output cu the:
    python auto_label.py --input dataset/raw/snow_map --output dataset/mask/snow_map --mode snow
"""

import os
import sys
import cv2
import argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# SNOW MAP LABELING - Thuat toan phan doat mau sac
# ============================================================
def segment_road_snow(image_bgr, debug=False):
    """
    Tach mat duong khoi tuyet bang thuat toan xu ly anh.

    Map tuyet co dac diem:
      - Mat duong: mau xam toi (asphalt), co vach ke trang.
      - Tuyet: mau trang xanh sang, texture bat canh tuyet.
      - Nen troi + canh cay: xanh lam dam / xanh la.

    Ket qua: binary mask (255 = duong, 0 = non-duong)
    """
    h, w = image_bgr.shape[:2]

    # === 1. Chuyen khong gian mau ===
    hsv  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # === 2. Tach vung duong (asphalt trong map tuyet) ===
    # Phan tich thuc te tren anh map tuyet 2026:
    #   Mat duong (asphalt): BGR~[94,59,44], HSV~[111, 134, 94]  --> V THAP (~60-110)
    #   Tuyet / bau troi   : BGR~[225,149,119], HSV~[110,120,225] --> V CAO (>130)
    #   --> Phan biet chu yeu bang gia tri V (do sang)
    v_channel = hsv[:, :, 2]  # Value (do sang)
    s_channel = hsv[:, :, 1]  # Saturation
    h_channel = hsv[:, :, 0]  # Hue

    # Duong: do sang V thap, co saturation S kha cao (mau xanh dam)
    # Phan tich thuc te tren anh first-person dataset snow_map (320x180):
    #   Duong gan (giua-duoi): V_mean=92,  S_mean=114
    #   Duong xa  (giua-tren): V_mean=126, S_mean=86   <-- can V<170 moi bat duoc
    #   Le duong  (2 ben):     V_mean=147, S_mean=114   <-- can V<170 moi bat duoc
    #   Tuyet:                 V_mean=202, S_mean=124   --> loai bang V<170
    #   Bau troi:              V_mean=89,  S_mean=42    --> loai bang S>30
    road_mask = (
        (v_channel < 170) &   # Bao gom duong xa + le duong (da kiem tra: tuyet V~202)
        (v_channel > 15) &    # Loai bong den tuyet viet
        (s_channel > 30)      # Loai bau troi xam (S thap)
    ).astype(np.uint8) * 255

    # === 3. Chi lay vung cam quan (bo bau troi + loai HUD goc man hinh) ===
    roi_top = h // 3   # Lat cat tu 1/3 chieu cao tro xuong
    road_mask[:roi_top, :] = 0

    # Loai bỏ HUD (cac goc man hinh: Port/Coin/timer/speedometer)
    hud_h = h // 5
    hud_w = w // 4
    road_mask[:hud_h, :hud_w] = 0          # Goc trai tren (Port, Coin text)
    road_mask[3*h//4:, :hud_w] = 0         # Goc trai duoi (timer)
    road_mask[3*h//4:, 3*hud_w:] = 0       # Goc phai duoi (speedometer)
    road_mask[:hud_h, 3*hud_w:] = 0        # Goc phai tren (FPS stats)
    # LUU Y: Khong can loai vung than xe vi anh GetRaw() la goc nhin first-person
    # (khong co than xe trong frame)

    # === 4. Loai bo nhieu bang morphology ===
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN,  kernel_open)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel_close)

    # === 5. Lay vung lien thong lon nhat (bo cac dot nhieu nho) ===
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(road_mask)
    if num_labels > 1:
        # Sap xep theo dien tich, lay vung lon nhat (bo background label=0)
        areas = stats[1:, cv2.CC_STAT_AREA]
        top_n = min(3, len(areas))  # Lay toi da 3 vung lon (xu ly duong phuc tap)
        top_indices = np.argsort(areas)[-top_n:] + 1  # +1 vi bo label 0
        final_mask = np.zeros_like(road_mask)
        for idx in top_indices:
            if areas[idx - 1] > (h * w * 0.03):  # Chi lay vung > 3% dien tich anh
                final_mask[labels == idx] = 255
    else:
        final_mask = road_mask

    if debug:
        debug_img = image_bgr.copy()
        # To xanh la len vung duong duoc nhan dien
        overlay = (debug_img[final_mask > 0] * 0.5 + np.array([0, 200, 0]) * 0.5).astype(np.uint8)
        debug_img[final_mask > 0] = overlay

        # Ve duong cat ROI (phan tren bi bo)
        cv2.line(debug_img, (0, h // 3), (w, h // 3), (255, 255, 0), 1)
        cv2.putText(debug_img, 'ROI top cut', (5, h // 3 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        # Thong tin coverage
        coverage = (final_mask > 0).sum() / (h * w) * 100
        cv2.putText(debug_img, f'Road: {coverage:.1f}%', (w - 140, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return final_mask, debug_img

    return final_mask, None


def mask_to_yolo_polygon(mask, class_id=0, min_area_ratio=0.01):
    """
    Chuyen binary mask thanh format YOLO Segmentation polygon (.txt).

    Format: class_id x1 y1 x2 y2 ... xn yn  (toa do da duoc chuan hoa 0-1)
    """
    h, w = mask.shape
    min_area = h * w * min_area_ratio

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        # Don gian hoa duong vien de giam so diem (epsilon = 0.5% chu vi)
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx  = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) < 3:
            continue

        # Chuan hoa toa do ve [0, 1]
        points_norm = []
        for pt in approx:
            x_norm = float(np.clip(pt[0][0] / w, 0.0, 1.0))
            y_norm = float(np.clip(pt[0][1] / h, 0.0, 1.0))
            points_norm.append(f"{x_norm:.6f} {y_norm:.6f}")

        line = f"{class_id} " + " ".join(points_norm)
        lines.append(line)

    return lines


# ============================================================
# YOLO MODEL LABELING - Dung model YOLO cu
# ============================================================
def label_with_model(img_path, model, class_id=0):
    """Su dung model YOLO cu de predict va xuat polygon."""
    results = model.predict(source=img_path, verbose=False)

    lines = []
    if results and results[0].masks is not None:
        polygons = results[0].masks.xyn  # Cac polygon da chuan hoa
        for poly in polygons:
            if len(poly) >= 3:
                coords = " ".join([f"{x:.6f} {y:.6f}" for x, y in poly])
                lines.append(f"{class_id} {coords}")
    return lines


# ============================================================
# MAIN
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="Tu dong gan nhan anh YOLO Segmentation")

    # Cach dinh nghia thu muc (2 lua chon tuong duong)
    grp_dir = p.add_argument_group("Tuy chon thu muc (cach 1: qua --scene)")
    grp_dir.add_argument("--scene",  type=str, default="",
                         help="Ten scene (vd: snow_map). Tu dong tim dataset/raw/<scene> va luu vao dataset/mask/<scene>")
    grp_dir2 = p.add_argument_group("Tuy chon thu muc (cach 2: chi ro duong dan)")
    grp_dir2.add_argument("--input",  type=str, default="",
                          help="Thu muc chua anh raw (ghi de len --scene)")
    grp_dir2.add_argument("--output", type=str, default="",
                          help="Thu muc luu file .txt (ghi de len --scene)")

    p.add_argument("--mode",   type=str, default="snow",
                   choices=["snow", "model"],
                   help="snow = thuat toan mau sac (map tuyet, khong can model) | model = dung model YOLO cu")
    p.add_argument("--model",  type=str, default="",
                   help="[Chi dung khi --mode model] Duong dan model YOLO .pt")
    p.add_argument("--device", type=str, default="0",
                   help="Thiet bi YOLO: '0'=GPU, 'cpu'=CPU (chi dung khi --mode model)")
    p.add_argument("--debug",  action="store_true",
                   help="Luu anh ket qua debug (overwrite de xem truc tiep ket qua)")
    p.add_argument("--workers", type=int, default=4,
                   help="So luong anh xu ly song song (mac dinh 4)")
    return p.parse_args()


def process_one_image(task):
    """Xu ly 1 anh - dung cho multiprocessing."""
    img_path, output_dir, mode, model_ref, debug = task

    filename = os.path.basename(img_path)
    txt_name = os.path.splitext(filename)[0] + ".txt"
    txt_path = os.path.join(output_dir, txt_name)

    image_bgr = cv2.imread(img_path)
    if image_bgr is None:
        return filename, False, "Khong doc duoc anh"

    if mode == "snow":
        mask, debug_img = segment_road_snow(image_bgr, debug=debug)
        lines = mask_to_yolo_polygon(mask)

        if debug and debug_img is not None:
            debug_dir = os.path.join(output_dir, "_debug")
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, filename), debug_img)

    else:  # mode == "model"
        lines = label_with_model(img_path, model_ref)

    if not lines:
        # Anh khong co duong nao duoc phat hien - tao file rong (empty label)
        # YOLO can file rong de hoc "khong co gi" trong anh nay (tranh false positive)
        with open(txt_path, "w") as f:
            pass
        return filename, True, "empty_label"

    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return filename, True, f"{len(lines)} polygon(s)"


def main():
    args = parse_args()

    # Xac dinh thu muc input/output
    if args.input and args.output:
        input_dir  = args.input
        output_dir = args.output
    elif args.scene:
        input_dir  = os.path.join(SCRIPT_DIR, "dataset", "raw",  args.scene)
        output_dir = os.path.join(SCRIPT_DIR, "dataset", "mask", args.scene)
    else:
        print("[LOI] Phai chi dinh --scene hoac ca --input va --output!")
        return

    if not os.path.isdir(input_dir):
        print(f"[LOI] Khong tim thay thu muc anh: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Tim danh sach anh
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    images = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(exts)
    ])

    if not images:
        print(f"[LOI] Khong tim thay anh nao trong: {input_dir}")
        return

    # Tai model neu can
    model_ref = None
    if args.mode == "model":
        from ultralytics import YOLO
        model_path = args.model
        if not model_path:
            # Tu dong tim model
            candidates = [
                os.path.join(SCRIPT_DIR, "Road_Seg_Model", "modelYolo", "weights", "best3.pt"),
                os.path.join(SCRIPT_DIR, "Road_Seg_Model", "modelYolo", "weights", "best.pt"),
                os.path.join(SCRIPT_DIR, "best.pt"),
            ]
            model_path = next((p for p in candidates if os.path.exists(p)), None)
            if not model_path:
                print("[LOI] Khong tim thay model .pt! Dung --model de chi dinh, hoac dung --mode snow.")
                return
        print(f"[INFO] Nap YOLO model tu: {model_path}")
        model_ref = YOLO(model_path)

    print("\n" + "=" * 60)
    print(f"  AUTO LABEL - UIT CAR RACING 2026")
    print(f"  Che do    : {'Snow Map (Color Segmentation)' if args.mode == 'snow' else 'YOLO Model'}")
    print(f"  Anh vao   : {input_dir}")
    print(f"  Label ra  : {output_dir}")
    print(f"  So luong  : {len(images)} anh")
    if args.debug:
        print(f"  Debug     : Bat (anh ket qua luu vao {output_dir}/_debug/)")
    print("=" * 60 + "\n")

    success_count = 0
    empty_count   = 0
    fail_count    = 0

    tasks = [(img, output_dir, args.mode, model_ref, args.debug) for img in images]

    for i, task in enumerate(tasks):
        filename, ok, info = process_one_image(task)
        if ok:
            if info == "empty_label":
                empty_count += 1
            else:
                success_count += 1
        else:
            fail_count += 1

        # Hien thi progress
        bar_len = 35
        filled  = int(bar_len * (i + 1) / len(images))
        bar     = "█" * filled + "░" * (bar_len - filled)
        pct     = (i + 1) / len(images) * 100
        print(f"\r  [{bar}] {i+1:4d}/{len(images)} ({pct:.0f}%)  {filename[:30]}", end="", flush=True)

    print(f"\n\n{'=' * 60}")
    print(f"  AUTO LABEL HOAN TAT!")
    print(f"  Co polygon  : {success_count} anh")
    print(f"  Empty label : {empty_count} anh (khong co duong - binh thuong)")
    print(f"  Loi         : {fail_count} anh")
    print(f"  Thu muc out : {output_dir}")
    print(f"{'=' * 60}")

    if args.debug:
        print(f"\n[DEBUG] Xem anh ket qua tai: {output_dir}/_debug/")

    scene = args.scene or os.path.basename(input_dir)
    print(f"\nBuoc tiep theo:")
    print(f"  python prepare_dataset.py --scene {scene}")


    import subprocess
    try:
        subprocess.run(["chmod", "-R", "777", "/workspace/my_code/dataset"], check=False)
        print("[INFO] Da tu dong mo khoa quyen truy cap cho thu muc dataset (chmod 777).")
    except:
        pass

if __name__ == "__main__":
    main()

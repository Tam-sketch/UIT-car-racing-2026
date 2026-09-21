#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
collect_data.py  (v3 - 2026 Snow Map Edition)
----------------------------------------------
Thu thap anh tu Unity Simulator de lam dataset train YOLO.

Cach dung:
    # [KHUYEN DUNG cho map tuyet] Xe tu lai bang thuat toan mau sac, khong can model:
    python collect_data.py --scene snow_map --drive snow --max 1000

    # Xe tu lai bang model YOLO cu (cho map binh thuong):
    python collect_data.py --scene snow_map --drive av --max 800

    # Lay anh thu cong - ban tu lai trong Unity MANUAL MODE:
    python collect_data.py --scene snow_map --drive manual --max 1000
    LUU Y: Script chi lay anh, TUYET DOI khong gui lenh lai xe.
            Xe do ban dieu khien hoan toan trong Unity.
"""

import os
import sys
import stat
import cv2
import time
import argparse
import numpy as np
from datetime import datetime

# Tat GUI neu khong co man hinh X11
SHOW_GUI = os.environ.get('ENABLE_GUI', '0').lower() in ('1', 'true', 'yes') \
           and bool(os.environ.get('DISPLAY'))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.extend(["/workspace", SCRIPT_DIR])

from ucr_lib import GetStatus, GetRaw, AVControl, CloseSocket

# Duong dan model YOLO (chi dung khi --drive av)
MODEL_PATHS = [
    os.path.join(SCRIPT_DIR, "Road_Seg_Model", "modelYolo", "weights", "best3.pt"),
    os.path.join(SCRIPT_DIR, "Road_Seg_Model", "modelYolo", "weights", "best.pt"),
    os.path.join(SCRIPT_DIR, "best.pt"),
    "/workspace/my_code/Road_Seg_Model/modelYolo/weights/best3.pt",
    "/workspace/best3.pt",
]


# ─────────────────────────────────────────────────────────────
#  SNOW DRIVE  –  Tu lai bang thuat toan phan tich mau sac
#  Khong can model YOLO, hoat dong tren map tuyet nho vao
#  su khac biet do sang (V channel):
#    Mat duong asphalt: V < 120  (xanh toi)
#    Tuyet / bau troi : V > 130  (trang xanh sang)
# ─────────────────────────────────────────────────────────────
def segment_snow_road(image_bgr):
    """
    Tra ve binary mask (255=duong, 0=non-duong) cho map tuyet.
    Dua vao gia tri V (brightness) trong khong gian mau HSV.
    """
    h, w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    v = hsv[:, :, 2]
    s = hsv[:, :, 1]

    # Duong: toi (V<120), co mau (S>50), loai bong toi tuyet viet (V>20)
    mask = ((v < 120) & (v > 20) & (s > 50)).astype(np.uint8) * 255

    # Chi xet phan duoi anh (vung quan trong, bo bau troi + HUD)
    mask[:h // 3, :] = 0
    hud_h, hud_w = h // 5, w // 4
    mask[:hud_h,     :hud_w]  = 0   # Goc trai tren
    mask[3*h//4:,    :hud_w]  = 0   # Goc trai duoi
    mask[3*h//4:, 3*hud_w:]   = 0   # Goc phai duoi
    mask[:hud_h,  3*hud_w:]   = 0   # Goc phai tren

    # Morphology: giam nhieu + lam lien mach
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    return mask


def steer_from_mask(mask):
    """
    Tinh toc do va goc lai tu binary road mask.
    Tra ve (speed, angle).
    """
    h, w = mask.shape

    # Scan tung hang pixel tu duoi len, lay trung diem cua duong
    points = []
    for y in range(h - 1, h // 3, -1):
        cols = np.where(mask[y] > 0)[0]
        if len(cols) > 0:
            points.append((int(np.mean(cols)), y))

    if not points:
        # Khong thay duong → dung xe, tranh xe lao ra ngoai hon
        return 0.0, 0.0

    near_pts = [(cx, y) for cx, y in points if y >= h * 2 // 3]
    far_pts  = [(cx, y) for cx, y in points if y <  h * 2 // 3]

    near_err = (int(np.mean([cx for cx, _ in near_pts])) - w // 2) if near_pts else 0
    far_err  = (int(np.mean([cx for cx, _ in far_pts]))  - w // 2) if far_pts  else 0

    # Pha tron near + far
    error = 0.65 * near_err + 0.35 * far_err
    angle = float(np.clip(error * 0.14, -25, 25))

    # Giam toc khi re goc lon
    speed = 15.0 if abs(angle) > 18 else (20.0 if abs(angle) > 10 else 25.0)
    return speed, angle


def get_snow_debug_frame(raw_image, mask, speed, angle):
    """Tao anh debug hien thi mask + thong tin dieu khien (dung khi ENABLE_GUI=1)."""
    h, w = raw_image.shape[:2]
    overlay = raw_image.copy()
    green_layer = np.zeros_like(raw_image)
    green_layer[mask > 0] = (0, 220, 0)
    cv2.addWeighted(overlay, 0.6, green_layer, 0.4, 0, overlay)
    # Ve duong trung tam
    cx_list = [int(np.mean(np.where(mask[y] > 0)[0]))
               for y in range(h-1, h//3, -5)
               if np.any(mask[y] > 0)]
    for i in range(1, len(cx_list)):
        y1, y2 = h - 1 - (i-1)*5, h - 1 - i*5
        if 0 <= y2 < h:
            cv2.line(overlay, (cx_list[i-1], y1), (cx_list[i], y2), (0, 255, 255), 2)
    cv2.putText(overlay, f"Speed:{speed:.0f}  Angle:{angle:+.1f}", (10, h-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return overlay


# ─────────────────────────────────────────────────────────────
#  YOLO AV DRIVE  –  Dung model YOLO cu
# ─────────────────────────────────────────────────────────────
def get_steering_yolo(model, raw_image, device="0"):
    """Tinh toc do va goc lai bang YOLO segmentation model."""
    try:
        results = model.predict(source=raw_image, verbose=False, device=device)
        if not results or results[0].masks is None:
            return 20.0, 0.0

        masks = results[0].masks.data.cpu().numpy()
        seg = (np.sum(masks, axis=0) > 0).astype(np.uint8) * 255

        h, w = seg.shape
        points = []
        for y in range(h - 1, -1, -1):
            cols = np.where(seg[y] > 0)[0]
            if len(cols) > 0:
                points.append(int(np.mean(cols)))

        if not points:
            return 20.0, 0.0

        near = points[:h // 2]
        far  = points[h * 2 // 3:]
        near_err = (int(np.mean(near)) - w // 2) if near else 0
        far_err  = (int(np.mean(far))  - w // 2) if far  else 0
        error = 0.6 * near_err + 0.4 * far_err
        angle = float(np.clip(error * 0.12, -25, 25))
        speed = 20.0 if abs(angle) > 15 else 28.0
        return speed, angle
    except Exception:
        return 20.0, 0.0


# ─────────────────────────────────────────────────────────────
#  UTILS
# ─────────────────────────────────────────────────────────────
def print_progress(saved, total, extra=""):
    bar_len = 30
    filled  = int(bar_len * saved / max(total, 1))
    bar     = "█" * filled + "░" * (bar_len - filled)
    pct     = saved / max(total, 1) * 100
    print(f"\r  [{bar}] {saved:4d}/{total} ({pct:.0f}%)  {extra}", end="", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="Thu thap anh tu Unity Simulator")
    p.add_argument("--scene",    type=str,   default="snow_map",
                   help="Ten canh (vd: snow_map, map1, night)")
    p.add_argument("--drive",    type=str,   default="snow",
                   choices=["snow", "av", "manual"],
                   help="snow = tu lai bang thuat toan mau sac (map tuyet, KHUYEN DUNG) | "
                        "av = tu lai bang model YOLO cu | "
                        "manual = lay anh khi ban tu lai trong Unity")
    p.add_argument("--interval", type=float, default=0.25,
                   help="Giay giua 2 anh lien tiep (mac dinh: 0.25)")
    p.add_argument("--max",      type=int,   default=1000,
                   help="So anh toi da (mac dinh: 1000)")
    p.add_argument("--device",   type=str,   default="0",
                   help="Device YOLO: '0'=GPU, 'cpu'=CPU (chi dung khi --drive av)")
    p.add_argument("--speed",    type=float, default=0.0,
                   help="Toc do co dinh (0 = tu dong tinh theo goc re)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    save_dir = os.path.join(SCRIPT_DIR, "dataset", "raw", args.scene)

    # Tao thu muc va dam bao quyen ghi (fix loi bi khoa do Docker/host uid khac nhau)
    os.makedirs(save_dir, exist_ok=True)
    try:
        # chmod 777 cho toan bo cay thu muc dataset
        dataset_root = os.path.join(SCRIPT_DIR, "dataset")
        for dirpath, dirnames, filenames in os.walk(dataset_root):
            os.chmod(dirpath, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    except Exception as e:
        print(f"[WARN] Khong the chmod dataset: {e} (co the bi loi khi chay)")

    existing = len([f for f in os.listdir(save_dir) if f.endswith(".jpg")])

    # Chuan bi model YOLO neu can
    yolo_model = None
    if args.drive == "av":
        from ultralytics import YOLO
        mp = next((p for p in MODEL_PATHS if os.path.exists(p)), None)
        if mp is None:
            print("[LOI] Khong tim thay file .pt nao! Dung --drive snow thay the.")
            return
        print(f"[INFO] Nap YOLO model: {mp}")
        yolo_model = YOLO(mp)

    # In thong tin
    drive_label = {
        "snow":   "Tu lai - Snow Color Seg (khong can YOLO model)",
        "av":     "Tu lai - YOLO Model cu",
        "manual": "Thu cong - Ban tu lai trong Unity",
    }[args.drive]

    print("\n" + "=" * 65)
    print(f"  THU THAP DU LIEU TRAINING - UIT CAR RACING 2026")
    print(f"  Scene     : {args.scene.upper()}")
    print(f"  Drive     : {drive_label}")
    print(f"  Luu vao   : {save_dir}")
    print(f"  Interval  : {args.interval}s / anh")
    print(f"  Muc tieu  : {args.max} anh  (da co san: {existing})")
    if SHOW_GUI:
        print(f"  GUI       : Bat (ENABLE_GUI=1)")
    print("=" * 65)

    if args.drive == "manual":
        print("  >> HAY CHUYEN UNITY SANG MANUAL MODE truoc!")
        print("  >> Script chi lay anh, KHONG gui lenh lai xe.")
    elif args.drive == "snow":
        print("  >> HAY CHUYEN UNITY SANG AV MODE.")
        print("  >> Script se tu lai bang thuat toan mau sac (khong can model).")
    else:
        print("  >> HAY CHUYEN UNITY SANG AV MODE.")
    print("  >> Nhan Ctrl+C de dung bat cu luc nao.\n")

    saved_count    = 0
    last_save_time = 0.0
    start_time     = time.time()
    waiting_count  = 0
    no_road_streak = 0   # Dem lien tiep frame khong thay duong

    try:
        while saved_count < args.max:
            # ── Doc frame tu Unity ────────────────────────────────
            try:
                if args.drive == "manual":
                    # Manual mode: CHI goi GetRaw, KHONG goi GetStatus
                    # De Unity tu quan ly trang thai xe (tranh can thiep vao manual control)
                    raw_image = GetRaw()
                else:
                    # AV modes: PHAI goi GetStatus truoc GetRaw (bat buoc theo UCR protocol)
                    GetStatus()
                    raw_image = GetRaw()
            except Exception:
                time.sleep(0.1)
                continue

            if raw_image is None or raw_image.size == 0:
                waiting_count += 1
                if waiting_count % 40 == 0:
                    print(f"\n[INFO] Dang cho frame tu Unity ({waiting_count//40*2}s)...", flush=True)
                time.sleep(0.05)
                continue

            waiting_count = 0

            # ── Dieu khien xe (chi cho AV modes) ──────────────────
            if args.drive == "snow":
                mask  = segment_snow_road(raw_image)
                speed, angle = steer_from_mask(mask)

                if args.speed > 0:
                    speed = args.speed if speed > 0 else 0.0

                if speed == 0.0:
                    no_road_streak += 1
                    if no_road_streak % 10 == 1:
                        print(f"\n[CANH BAO] {no_road_streak} frame khong thay duong! Xe co the ra khoi duong.", flush=True)
                else:
                    no_road_streak = 0

                try:
                    AVControl(float(speed), float(angle))
                except Exception:
                    pass

                if SHOW_GUI:
                    dbg = get_snow_debug_frame(raw_image, mask, speed, angle)
                    small = cv2.resize(dbg, (0, 0), fx=0.5, fy=0.5)
                    cv2.imshow("Snow Drive Debug", small)
                    if cv2.waitKey(1) == ord('q'):
                        print("\n[INFO] Nguoi dung bam Q de thoat.")
                        break

                time.sleep(0.025)

            elif args.drive == "av" and yolo_model is not None:
                speed, angle = get_steering_yolo(yolo_model, raw_image, args.device)
                if args.speed > 0:
                    speed = args.speed
                try:
                    AVControl(float(speed), float(angle))
                except Exception:
                    pass
                time.sleep(0.025)

            elif args.drive == "manual":
                # TUYET DOI khong gui AVControl trong manual mode
                # Hien thi anh truc tiep de theo doi khi ENABLE_GUI=1
                if SHOW_GUI:
                    small = cv2.resize(raw_image, (0, 0), fx=0.5, fy=0.5)
                    cv2.imshow("Manual - Live Camera", small)
                    if cv2.waitKey(1) == ord('q'):
                        print("\n[INFO] Nguoi dung bam Q de thoat.")
                        break

            # ── Luu anh theo interval ─────────────────────────────
            now = time.time()
            if now - last_save_time < args.interval:
                continue
            last_save_time = now

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename  = f"{args.scene}_{timestamp}.jpg"
            filepath  = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, raw_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1

            # Hien thi progress (kem thong tin snow drive)
            if args.drive == "snow":
                extra = f"spd={speed:.0f} ang={angle:+.1f}  {filename[-20:]}"
            else:
                extra = filename[-25:]
            print_progress(saved_count, args.max, extra)

    except KeyboardInterrupt:
        print("\n[INFO] Nguoi dung dung thu thap.")
    finally:
        elapsed = time.time() - start_time
        try:
            CloseSocket()
        except Exception:
            pass
        if SHOW_GUI:
            cv2.destroyAllWindows()

    total_in_dir = len([f for f in os.listdir(save_dir) if f.endswith(".jpg")])
    print(f"\n\n{'=' * 65}")
    print(f"  HOAN TAT!")
    print(f"  Vua luu   : {saved_count} anh ({elapsed:.0f} giay)")
    print(f"  Tong cong : {total_in_dir} anh trong '{args.scene}'")
    print(f"  Duong dan : {save_dir}")
    print(f"{'=' * 65}\n")
    print("Buoc tiep theo:")
    print(f"  python auto_label.py --scene {args.scene} --mode snow")
    print(f"  python prepare_dataset.py --scene {args.scene}")


if __name__ == "__main__":
    main()

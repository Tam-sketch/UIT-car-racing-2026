# ============================================================
# UCR 2026 — TwinLiteNet ONNX controller with tuned PID + logging
# Compiled with maycay.py robust scaffolding and optimizations
# ============================================================
import os
import sys
import time
import argparse
import csv
from datetime import datetime

os.environ.pop('QT_QPA_PLATFORM', None)

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
DEBUG_SAVE = True  # Luu anh ra debug_frames/ (danh cho moi truong khong co man hinh)
SHOW_GUI = (os.environ.get('ENABLE_GUI', '').lower() in ['1', 'true', 'yes']) and bool(os.environ.get('DISPLAY'))

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="UIT Car Racing - YOLO PID")
parser.add_argument("--collect", type=str, default="", help="Ten thu muc luu anh data (vd: map1)")
parser.add_argument("--model", type=str, default="", help="Duong dan YOLO model (vd: best.pt)")
parser.add_argument("--device", type=str, default="0", help="'0'=GPU, 'cpu'=CPU")
args, unknown = parser.parse_known_args()

COLLECT_DIR = None
if args.collect:
    COLLECT_DIR = os.path.join(CURRENT_DIR, "dataset", "raw", args.collect)
    os.makedirs(COLLECT_DIR, exist_ok=True)
    print(f"[INFO] CHE DO THU THAP DU LIEU: Anh se duoc luu vao {COLLECT_DIR}", flush=True)

sys.path.extend(["/workspace", CURRENT_DIR])

from ucr_lib import GetStatus, GetRaw, AVControl, CloseSocket
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# ------------------------------------------------------------
# LOGGING — per-frame CSV for offline PID analysis
# ------------------------------------------------------------
LOG_PATH = "/workspace/pid_log.csv"
log_file = open(LOG_PATH, "w", newline="")
log_writer = csv.writer(log_file)
log_writer.writerow([
    "t", "error", "near", "far", "curve",
    "speed", "angle", "p", "i", "d", "raw",
])
print(f"[log] writing to {LOG_PATH}")


# ------------------------------------------------------------
# PERCEPTION — YOLOv8
# ------------------------------------------------------------
# Tu dong tim duong dan model linh hoat (giong maycay.py)
MODEL_PATHS = [
    "/workspace/my_code/Road_Seg_Model/modelYolo/weights/best7.pt",
    "/workspace/my_code/Road_Seg_Model/modelYolo/weights/best.pt",
    "/workspace/Road_Seg_Model/modelYolo/weights/best.pt",
    os.path.join(CURRENT_DIR, "Road_Seg_Model", "modelYolo", "weights", "best7.pt"),
    os.path.join(CURRENT_DIR, "Road_Seg_Model", "modelYolo", "weights", "best.pt"),
]
if args.model:
    model_path = args.model
    if not os.path.exists(model_path) and os.path.exists(os.path.join(CURRENT_DIR, model_path)):
        model_path = os.path.join(CURRENT_DIR, model_path)
else:
    model_path = next((p for p in MODEL_PATHS if os.path.exists(p)), MODEL_PATHS[0])

DEVICE = args.device
model = YOLO(model_path)
print(f"[perception] YOLO loaded: {model_path} | Device: {DEVICE}", flush=True)

def get_segmentation(raw_image):
    """BGR uint8 frame -> binary road mask (0 or 255)."""
    # [DA KHOI PHUC] Chay o do phan giai goc (mac din imgsz=640) de dam bao do chinh xac
    results = model.predict(source=raw_image, verbose=False, device=DEVICE)

    if not results or results[0].masks is None:
        return np.zeros(raw_image.shape[:2], dtype=np.uint8)

    masks = results[0].masks.data.cpu().numpy()  # (N, H, W)
    combined_mask = (np.sum(masks, axis=0) > 0).astype(np.uint8) * 255
    return combined_mask


# ------------------------------------------------------------
# PID CONTROLLER — continuous, anti-windup, adaptive D
# ------------------------------------------------------------
class SteeringPID:
    """
    Continuous PID for lateral lane-following.

    Tuned to prevent edge-to-edge overshoot during large-error recovery:
    - Strong D-term (kd = 0.75) for pre-emptive release
    - Adaptive D boost during fast transients
    - Lower output_limit (20°) to reduce accumulated yaw rate
    - Faster D filter (alpha = 0.55) for lower lag
    - Wider rate_limit (4°/frame) so release is not throttled
    """

    def __init__(self,
                 kp=0.32,
                 ki=0.003,
                 kd=0.75,
                 integral_limit=3.0,
                 derivative_alpha=0.55,
                 output_limit=20.0,
                 rate_limit=4.0,
                 adaptive_d_boost=0.10,
                 adaptive_d_cap=2.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.derivative_alpha = derivative_alpha
        self.output_limit = output_limit
        self.rate_limit = rate_limit
        self.adaptive_d_boost = adaptive_d_boost
        self.adaptive_d_cap = adaptive_d_cap

        # State
        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_derivative = 0.0
        self.prev_output = 0.0
        self.first_call = True

        # Exposed for logging / debug
        self.last_p = 0.0
        self.last_i = 0.0
        self.last_d = 0.0
        self.last_raw = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_derivative = 0.0
        self.prev_output = 0.0
        self.first_call = True

    def step(self, error):
        # Proportional
        p_term = self.kp * error

        # Integral with anti-windup
        self.integral += error
        self.integral = np.clip(self.integral,
                                -self.integral_limit,
                                +self.integral_limit)
        i_term = self.ki * self.integral

        # Derivative with low-pass filter
        if self.first_call:
            raw_derivative = 0.0
        else:
            raw_derivative = error - self.prev_error

        self.filtered_derivative = (
            self.derivative_alpha * raw_derivative
            + (1 - self.derivative_alpha) * self.filtered_derivative
        )

        # Adaptive D boost during fast transients
        velocity_mag = abs(self.filtered_derivative)
        d_boost = 1.0 + self.adaptive_d_boost * min(velocity_mag, 20.0)
        d_boost = min(d_boost, self.adaptive_d_cap)
        d_term = self.kd * d_boost * self.filtered_derivative

        # Sum + smooth saturation
        raw_output = p_term + i_term + d_term
        output = self.output_limit * np.tanh(raw_output / self.output_limit)

        # Rate limit
        delta = np.clip(output - self.prev_output,
                        -self.rate_limit,
                        +self.rate_limit)
        output = self.prev_output + delta

        # Store
        self.prev_error = error
        self.prev_output = output
        self.first_call = False

        self.last_p = p_term
        self.last_i = i_term
        self.last_d = d_term
        self.last_raw = raw_output

        return output


# ------------------------------------------------------------
# GEOMETRY — near/far centerline extraction
# ------------------------------------------------------------
class LaneWidthEstimator:
    """Used for debug visualization only; not for control decisions."""

    def __init__(self):
        self.base_width = None
        self.current_width = None
        self.width_profile = []

    def measure_width(self, gray_image, step=5, fraction=2):
        height, width = gray_image.shape
        max_y = height - height // 3
        min_y = height - height // fraction
        widths = []
        self.width_profile = []
        for y in range(max_y, min_y, -step):
            row = gray_image[y, :]
            nz = np.where(row > 0)[0]
            if len(nz) > 1:
                lane_width = nz[-1] - nz[0]
                widths.append(lane_width)
                self.width_profile.append((y, lane_width))
        if widths:
            self.current_width = np.median(widths)
            if self.base_width is None:
                self.base_width = self.current_width
        return self.current_width, self.width_profile


def extract_centerline(seg_mask, speed, max_speed=35.0):
    """
    Row-wise centerline extraction with speed-adaptive near/far blending.
    Compiled with maycay.py's Numpy Vectorization for maximum speed.
    """
    gray = seg_mask if len(seg_mask.shape) == 2 else cv2.cvtColor(seg_mask, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # [TOI UU] Vectorized row centers (from maycay.py)
    row_centers = []
    y_coords, x_coords = np.nonzero(gray)
    if len(y_coords) > 0:
        unique_y, inverse_indices = np.unique(y_coords, return_inverse=True)
        sum_x = np.bincount(inverse_indices, weights=x_coords)
        count_x = np.bincount(inverse_indices)
        mean_x = (sum_x / count_x).astype(int)
        
        pts = list(zip(mean_x, unique_y))
        pts.reverse()  # Tu duoi len tren (y giam dan)
        row_centers = pts

    if not row_centers:
        return 0.0, 0.0, 0.0, 0.0, []

    near_pts = [(cx, y) for cx, y in row_centers if y >= h // 2]
    far_pts  = [(cx, y) for cx, y in row_centers if y <  h // 3]

    near_error = (int(np.mean([cx for cx, _ in near_pts])) - w // 2) if near_pts else 0
    far_error  = (int(np.mean([cx for cx, _ in far_pts]))  - w // 2) if far_pts  else 0

    bottom_cx = row_centers[-1][0]
    top_cx    = row_centers[0][0]
    curve_error = top_cx - bottom_cx

    # Speed-adaptive near/far weighting
    speed_ratio = np.clip(speed / max_speed, 0.0, 1.0)
    w_far = 0.25 + 0.40 * speed_ratio
    if near_error * far_error < 0:
        w_far *= 0.5
    w_near = 1.0 - w_far

    blended_error = w_near * near_error + w_far * far_error

    # Deadband
    if abs(blended_error) < 2.0:
        blended_error = 0.0

    return blended_error, near_error, far_error, curve_error, row_centers


# ------------------------------------------------------------
# SPEED PLANNER — curvature-aware
# ------------------------------------------------------------
def plan_speed(curve_error, max_speed=35.0, min_speed=22.0, width=320):
    curve_ratio = min(abs(curve_error) / (width // 2), 1.0)
    target = max_speed * (1.0 - 0.55 * curve_ratio)
    target = max(min_speed, target)
    return target, curve_ratio


# ------------------------------------------------------------
# DEBUG OVERLAY
# ------------------------------------------------------------
def show_image_safe(win_name, img):
    """Safely show or save image regardless of headless environment (from maycay.py)"""
    if SHOW_GUI:
        try:
            small = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow(win_name, small)
        except Exception:
            pass
    if DEBUG_SAVE:
        debug_dir = os.path.join(CURRENT_DIR, "debug_frames")
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, f"{win_name}_live.jpg"), img)


def draw_debug(gray, row_centers, lane_cx, blended_error, speed, curve_ratio,
               angle, near_error, far_error, width_est, pid):
    h, w = gray.shape
    debug = np.zeros((h, w, 3), dtype=np.uint8)
    debug[gray > 0] = (90, 90, 90)

    for cx, y in row_centers:
        debug[y, cx] = (0, 255, 0)

    cv2.line(debug, (lane_cx, 0), (lane_cx, h), (0, 0, 255), 2)

    if row_centers:
        near_pts = [(cx, y) for cx, y in row_centers if y >= h // 2]
        far_pts  = [(cx, y) for cx, y in row_centers if y <  h // 3]
        if near_pts:
            near_cx = int(np.mean([cx for cx, _ in near_pts]))
            cv2.line(debug, (near_cx, h // 2), (near_cx, h), (0, 255, 255), 2)
        if far_pts:
            far_arr = np.array(far_pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(debug, [far_arr], False, (255, 0, 0), 2)

    # Info panel
    cv2.putText(debug, f"blend={blended_error:+.1f}  near={near_error:+d}  far={far_error:+d}",
                (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(debug, f"speed={speed:.1f}  curve={curve_ratio:.2f}  angle={angle:+.1f}",
                (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(debug, f"P={pid.last_p:+.1f}  I={pid.last_i:+.1f}  D={pid.last_d:+.1f}",
                (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1)
    
    if width_est and width_est.current_width:
        cv2.putText(debug, f"width={width_est.current_width:.0f} (base={width_est.base_width:.0f})",
                    (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    show_image_safe("Lane Debug", debug)


# ------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------
if __name__ == "__main__":
    MAX_SPEED = 45.0
    MIN_SPEED = 22.0

    pid = SteeringPID(
        kp=0.32,
        ki=0.003,
        kd=0.75,
        integral_limit=3.0,
        derivative_alpha=0.55,
        output_limit=20.0,
        rate_limit=4.0,
        adaptive_d_boost=0.10,
        adaptive_d_cap=2.5,
    )

    width_est = LaneWidthEstimator()
    speed = MIN_SPEED

    frame_count = 0
    waiting_count = 0
    connected_announced = False

    print("[INFO] Start loop (Waiting for Unity)...", flush=True)

    try:
        while True:
            # --- Robust Connection Handling (from maycay.py) ---
            try:
                state = GetStatus()
                raw_image = GetRaw()
            except Exception as e:
                time.sleep(0.05)
                continue

            if raw_image is None or raw_image.size == 0:
                waiting_count += 1
                if waiting_count % 30 == 0:
                    print("[INFO] Dang cho frame tu Unity...", flush=True)
                time.sleep(0.05)
                continue

            if not connected_announced:
                print(f"[INFO] >>> DA KET NOI! Frame {raw_image.shape} <<<", flush=True)
                connected_announced = True

            waiting_count = 0
            frame_count += 1

            # --- Data Collection (from maycay.py) ---
            if COLLECT_DIR and frame_count % 3 == 0:
                try:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    cv2.imwrite(os.path.join(COLLECT_DIR, f"{args.collect}_{ts}.jpg"),
                                raw_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                except Exception:
                    pass

            # --- Perception ---
            seg_mask = get_segmentation(raw_image)
            show_image_safe("Raw Image", raw_image)
            show_image_safe("YOLO Segmentation", seg_mask)

            # --- Geometry ---
            blended_error, near_e, far_e, curve_e, row_pts = \
                extract_centerline(seg_mask, speed, MAX_SPEED)

            gray_for_width = seg_mask if len(seg_mask.shape) == 2 else \
                             cv2.cvtColor(seg_mask, cv2.COLOR_BGR2GRAY)
            width_est.measure_width(gray_for_width)

            # --- PID Control ---
            angle = pid.step(blended_error)

            # --- Speed Planning ---
            target_speed, curve_ratio = plan_speed(curve_e, MAX_SPEED, MIN_SPEED)
            speed = 0.7 * speed + 0.3 * target_speed

            # --- Send Command ---
            try:
                AVControl(float(speed), float(angle))
            except Exception as e:
                if frame_count % 10 == 0:
                    print(f"[ERROR AVControl]: {e}", flush=True)

            # --- Logging ---
            log_writer.writerow([
                time.time(),
                blended_error, near_e, far_e, curve_e,
                speed, angle,
                pid.last_p, pid.last_i, pid.last_d, pid.last_raw,
            ])

            # --- Debug Overlay ---
            lane_cx = row_pts[-1][0] if row_pts else 160
            draw_debug(
                gray_for_width, row_pts, lane_cx,
                blended_error, speed, curve_ratio,
                angle, near_e, far_e, width_est, pid,
            )

            # --- Console Output ---
            if frame_count % 3 == 0 or frame_count == 1:
                print(f"err={blended_error:+.1f} near={near_e:+d} far={far_e:+d} "
                      f"curve={curve_ratio:.2f} speed={speed:.1f} angle={angle:+.2f} "
                      f"P={pid.last_p:+.1f} I={pid.last_i:+.1f} D={pid.last_d:+.1f}", flush=True)

            if SHOW_GUI:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        print("Closing socket and windows...")
        try:
            AVControl(0.0, 0.0)
        except:
            pass
        
        log_file.close()
        print(f"[log] saved {LOG_PATH}")
        
        try:
            CloseSocket()
            cv2.destroyAllWindows()
        except:
            pass

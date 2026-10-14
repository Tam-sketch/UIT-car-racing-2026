# ==============================================================
# UIT CAR RACING 2026 - V5.6
#   V5.5  : cover_bottom vào log + GUI + console
#   V5.6  : force reset sớm khi |Δheading| ≥ 45° (GetStatus['Heading'])
# ==============================================================

from ucr_lib import GetStatus, GetRaw, AVControl, CloseSocket
import cv2
import csv
import os
import time
import numpy as np
from collections import deque
from ultralytics import YOLO
try:
    from skimage.morphology import skeletonize
    SKEL_OK = True
except ImportError:
    SKEL_OK = False
    print("[WARN] skimage not found")

# ==============================================================
# CONFIG
# ==============================================================
ROAD_MODEL_PATH = "/workspace/best14.pt"
SIGN_MODEL_PATH = "/workspace/sign.pt"

ROAD_IMGSZ = 320
SIGN_IMGSZ = 640
SIGN_CONF  = 0.7

SKEL_MIN_ARM_SIZE        = 5
SKEL_JUNCTION_DILATE     = 3
SKEL_DX_LEFT_THRESH      = 0.12
SKEL_DX_RIGHT_THRESH     = 0.12
MASK_CLOSE_KERNEL        = 5
MASK_MIN_AREA_RATIO      = 0.005
SKEL_S_VERIFY_X_HALF     = 15
SKEL_S_VERIFY_ROW_MIN_PX = 3
SKEL_S_VERIFY_MARGIN     = 8
TOPO_CONFIRM_FRAMES      = 3

VOTER_ACTIVATE_FRAMES = 10 #5
VOTER_TIMEOUT_FRAMES  = 100

RESET_DEBOUNCE_FRAMES = 15

FORCE_ANGLE               = 25.0
TURN_CONDITION_Y_LO_RATIO = 0.69
TURN_CONDITION_COVERAGE   = 0.80 #0.82

HEADING_RESET_DEG = 45.0

SLOW_SPEED = 15
MAX_SPEED  = 26.0
MIN_SPEED  = 0.0

SIGN_HOLD_FRAMES = 30

ASYM_STRENGTH      = 0.20
ASYM_THRESHOLD     = 8.0
ASYM_EMA           = 0.7
ASYM_CAP           = 12.0
ASYM_DECAY         = 0.80
FAR_DELTA_CLIP     = 25.0
D_FAR_CLIP         = 6.0
D_FAR_EMA_ALPHA    = 0.4
LEAD_RATE_GAIN     = 0.8

HARD_STEP_PX          = 30.0
ERR_STEP_PX           = 18.0
COV_DROP              = 0.22
JAGGED_PX             = 15.0
OCCL_HANGOVER         = 4
FAR_OUTLIER_PX        = 40.0
FAR_OUTLIER_NEAR_PX   = 20.0
FAR_SPREAD_PX         = 40.0
FAR_WINDOW            = 5

EMERGENCY_STREAK_REQ = 2
EMG_HOLD_AFTER_OCCL  = 5

LOG_PATH    = "/workspace/my_code/pid_log_v5.csv"
ENABLE_GUI  = os.environ.get("ENABLE_GUI", "1") != "0"

# ==============================================================
# HELPERS
# ==============================================================
def angle_diff_deg(a, b):
    """Chênh lệch góc trong [-180, 180]."""
    return (a - b + 180.0) % 360.0 - 180.0


def compute_max_speed(force_active, commit, sign_seen_recently):
    if force_active:
        return SLOW_SPEED
    if commit is not None:
        return SLOW_SPEED
    if sign_seen_recently:
        return SLOW_SPEED
    return MAX_SPEED

# ==============================================================
# LOAD MODELS
# ==============================================================
print("Loading models...")
road_model = YOLO(ROAD_MODEL_PATH)
sign_model = YOLO(SIGN_MODEL_PATH)
print(f"Road classes: {road_model.names}")
print(f"Sign classes: {sign_model.names}")

# ==============================================================
# PERCEPTION
# ==============================================================
def keep_largest_component(mask_bin):
    binary = (mask_bin > 0).astype(np.uint8)
    if binary.sum() == 0:
        return mask_bin
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return mask_bin
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    return ((labels == largest_idx).astype(np.uint8)) * 255


def get_road_masks(raw_image):
    if raw_image.dtype != np.uint8:
        raw_image = raw_image.astype(np.uint8)
    results = road_model.predict(source=raw_image, imgsz=ROAD_IMGSZ, verbose=False)
    if not results or results[0].masks is None:
        h, w = raw_image.shape[:2]
        empty = np.zeros((h, w), dtype=np.uint8)
        return empty, empty
    masks = results[0].masks.data.cpu().numpy()
    full = (np.sum(masks, axis=0) > 0).astype(np.uint8) * 255
    return full, keep_largest_component(full)


def sign_detect(raw_image, conf=SIGN_CONF, imgsz=SIGN_IMGSZ):
    results = sign_model.predict(source=raw_image, conf=conf, imgsz=imgsz, verbose=False)
    classes = []
    if results and results[0].boxes is not None and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            classes.append(sign_model.names[cls_id])
    return classes


# ==============================================================
# SKELETON TOPOLOGY
# ==============================================================
def clean_mask_for_skeleton(mask):
    binary = (mask > 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                   (MASK_CLOSE_KERNEL, MASK_CLOSE_KERNEL))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if num > 2:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = 1 + int(np.argmax(areas))
        binary = (labels == largest).astype(np.uint8)
    return binary


def skeleton_topology(mask, W, H):
    result = {'left': False, 'straight': False, 'right': False,
              'n_arms': 0, 'junction_y': -1, 'jx': -1,
              'arm_top_left': -1, 'arm_top_right': -1,
              'baseline_y': -1, 'strip_top_y': -1, 'ok': False}

    if not SKEL_OK or mask is None or W <= 0 or H <= 0:
        result['straight'] = True
        return result

    binary = clean_mask_for_skeleton(mask)
    if binary.sum() < MASK_MIN_AREA_RATIO * W * H:
        result['straight'] = True
        return result

    skel = skeletonize(binary > 0).astype(np.uint8)
    if skel.sum() < 15:
        result['straight'] = True
        return result

    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neigh = cv2.filter2D(skel, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    jmask = (skel == 1) & (neigh >= 3)

    if not jmask.any():
        result['straight'] = True
        result['ok'] = True
        return result

    jys, jxs = np.where(jmask)
    jy = int(np.max(jys))
    at_jy = jxs[jys == jy]
    jx = int(np.mean(at_jy))

    jmask_u8 = (jmask.astype(np.uint8) * 255)
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                    (SKEL_JUNCTION_DILATE, SKEL_JUNCTION_DILATE))
    jmask_dil = cv2.dilate(jmask_u8, kd, iterations=1)
    skel_arms = skel.copy()
    skel_arms[jmask_dil > 0] = 0

    num, labels = cv2.connectedComponents(skel_arms, connectivity=8)
    arms_info = []
    for lid in range(1, num):
        ys, xs = np.where(labels == lid)
        if len(ys) < SKEL_MIN_ARM_SIZE:
            continue
        dists = (ys - jy) ** 2 + (xs - jx) ** 2
        idx = int(np.argmax(dists))
        ey, ex = int(ys[idx]), int(xs[idx])
        y_min = int(ys.min())
        arms_info.append((ey, ex, y_min))

    result['n_arms'] = len(arms_info)
    result['junction_y'] = jy
    result['jx'] = jx

    L = R = False
    arm_top_left = None
    arm_top_right = None
    for (ey, ex, y_min) in arms_info:
        dy = ey - jy
        if dy > 5:
            continue
        dx = ex - jx
        dx_norm = dx / float(W)
        if dx_norm < -SKEL_DX_LEFT_THRESH:
            L = True
            if arm_top_left is None or y_min < arm_top_left:
                arm_top_left = y_min
        elif dx_norm > SKEL_DX_RIGHT_THRESH:
            R = True
            if arm_top_right is None or y_min < arm_top_right:
                arm_top_right = y_min

    result['arm_top_left']  = arm_top_left  if arm_top_left  is not None else -1
    result['arm_top_right'] = arm_top_right if arm_top_right is not None else -1

    S = False
    baseline_y = -1
    strip_top_y = jy - 5

    arm_tops = [y for y in (arm_top_left, arm_top_right) if y is not None]
    if arm_tops and jy > 5:
        baseline_y = min(arm_tops)
        x_lo = max(0, jx - SKEL_S_VERIFY_X_HALF)
        x_hi = min(W, jx + SKEL_S_VERIFY_X_HALF)
        y = jy - 5
        while y >= 0:
            row_px = int(binary[y, x_lo:x_hi].sum())
            if row_px >= SKEL_S_VERIFY_ROW_MIN_PX:
                strip_top_y = y
                y -= 1
            else:
                break
        if strip_top_y <= baseline_y - SKEL_S_VERIFY_MARGIN:
            S = True

    result['baseline_y']  = baseline_y
    result['strip_top_y'] = strip_top_y

    if not L and not S and not R:
        S = True

    result['left']     = L
    result['straight'] = S
    result['right']    = R
    result['ok']       = True
    return result


class TopoLatch:
    def __init__(self, confirm_frames=TOPO_CONFIRM_FRAMES):
        self.confirm_frames = confirm_frames
        self.latched = {'left': False, 'straight': True, 'right': False}
        self._pending = None
        self._cnt = 0

    def _key(self, topo):
        return (bool(topo['left']), bool(topo['straight']), bool(topo['right']))

    def update(self, raw):
        k = self._key(raw)
        k_lat = self._key(self.latched)
        if k == k_lat:
            self._pending = None
            self._cnt = 0
            return dict(self.latched)
        if self._pending == k:
            self._cnt += 1
        else:
            self._pending = k
            self._cnt = 1
        if self._cnt >= self.confirm_frames:
            self.latched = {'left': k[0], 'straight': k[1], 'right': k[2]}
            self._pending = None
            self._cnt = 0
            print(f"[TOPO-LATCH] → L{int(k[0])}S{int(k[1])}R{int(k[2])}")
        return dict(self.latched)


# ==============================================================
# PID v20
# ==============================================================
class OcclusionGate:
    def __init__(self, err_step_px=18.0, coverage_drop_frac=0.22,
                 jagged_px=15.0, hard_step_px=30.0,
                 hangover_frames=4, coverage_ema_alpha=0.9,
                 far_outlier_px=40.0, far_outlier_near_px=20.0,
                 far_spread_px=40.0, far_window=5):
        self.err_step_px = err_step_px
        self.coverage_drop_frac = coverage_drop_frac
        self.jagged_px = jagged_px
        self.hard_step_px = hard_step_px
        self.hangover_frames = hangover_frames
        self.cov_ema_alpha = coverage_ema_alpha
        self.far_outlier_px = far_outlier_px
        self.far_outlier_near_px = far_outlier_near_px
        self.far_spread_px = far_spread_px
        self.far_window = far_window
        self.cov_ema = None
        self.prev_error = 0.0
        self.initialized = False
        self.last_jagged = 0.0
        self.last_err_step = 0.0
        self.last_cov_drop = 0.0
        self.last_trigger_count = 0
        self.last_hard_step = False
        self.hangover_left = 0
        self.far_hist = deque(maxlen=far_window)
        self.last_far_outlier = False
        self.last_far_spread = 0.0

    def reset(self):
        self.cov_ema = None
        self.prev_error = 0.0
        self.initialized = False
        self.last_jagged = 0.0
        self.last_err_step = 0.0
        self.last_cov_drop = 0.0
        self.last_trigger_count = 0
        self.last_hard_step = False
        self.hangover_left = 0
        self.far_hist.clear()
        self.last_far_outlier = False
        self.last_far_spread = 0.0

    @staticmethod
    def _jaggedness(row_centers):
        if len(row_centers) < 6:
            return 0.0
        cxs = np.array([cx for cx, _ in row_centers], dtype=np.float32)
        d2 = np.diff(cxs, n=2)
        return float(np.mean(np.abs(d2)))

    def step(self, error, coverage, row_centers, valid, far_e=0.0, near_e=0.0):
        self.far_hist.append(float(far_e))
        if len(self.far_hist) >= 2:
            self.last_far_spread = float(max(self.far_hist) - min(self.far_hist))
        else:
            self.last_far_spread = 0.0

        if not self.initialized:
            self.cov_ema = coverage
            self.prev_error = error
            self.initialized = True
            self.last_hard_step = False
            self.hangover_left = 0
            self.last_far_outlier = False
            return False

        self.cov_ema = (self.cov_ema_alpha * self.cov_ema
                        + (1.0 - self.cov_ema_alpha) * coverage)

        if not valid:
            self.prev_error = error
            self.last_trigger_count = 5
            self.last_hard_step = False
            self.hangover_left = self.hangover_frames
            return True

        err_step = abs(error - self.prev_error)
        cov_drop = 0.0
        if self.cov_ema > 0.05:
            cov_drop = max(0.0, (self.cov_ema - coverage) / self.cov_ema)
        jagged = self._jaggedness(row_centers)

        far_outlier = (abs(far_e) > self.far_outlier_px
                       and abs(near_e) < self.far_outlier_near_px)
        self.last_far_outlier = far_outlier
        far_spread_trig = (len(self.far_hist) >= self.far_window
                           and self.last_far_spread > self.far_spread_px)

        signals = 0
        if err_step > self.err_step_px: signals += 1
        if cov_drop > self.coverage_drop_frac: signals += 1
        if jagged > self.jagged_px: signals += 1
        if far_outlier: signals += 1
        if far_spread_trig: signals += 1

        self.last_err_step = err_step
        self.last_cov_drop = cov_drop
        self.last_jagged = jagged
        self.last_trigger_count = signals
        self.prev_error = error

        hard = err_step > self.hard_step_px
        self.last_hard_step = hard
        raw_occl = hard or (signals >= 2)

        if raw_occl:
            self.hangover_left = self.hangover_frames
            return True
        if self.hangover_left > 0:
            self.hangover_left -= 1
            return True
        return False


class AuthorityManager:
    def __init__(self, output_limit, tau=0.35, sat_frac=0.9,
                 sat_frames=3, lose_enter_px=30.0, lose_exit_px=20.0):
        self.output_limit = output_limit
        self.tau = tau
        self.sat_frac = sat_frac
        self.sat_frames = sat_frames
        self.lose_enter_px = lose_enter_px
        self.lose_exit_px = lose_exit_px
        self.sat_streak = 0
        self.losing = False
        self.last_d_err = 0.0
        self.last_error_pred = 0.0

    def reset(self):
        self.sat_streak = 0
        self.losing = False
        self.last_d_err = 0.0
        self.last_error_pred = 0.0

    def step(self, error, prev_error, pid_output):
        d_err = error - prev_error
        error_pred = error + self.tau * d_err
        self.last_d_err = d_err
        self.last_error_pred = error_pred
        if abs(pid_output) > self.sat_frac * self.output_limit:
            self.sat_streak += 1
        else:
            self.sat_streak = 0
        if not self.losing:
            if (self.sat_streak >= self.sat_frames
                    and abs(error_pred) > abs(error)
                    and abs(error) > self.lose_enter_px):
                self.losing = True
        else:
            if abs(error) < self.lose_exit_px or self.sat_streak < self.sat_frames:
                self.losing = False
        return (1.5 if self.losing else 1.0,
                0.7 if self.losing else 1.0,
                0.6 if self.losing else 1.0)


class SteeringPID:
    def __init__(self, kp=0.32, ki=0.0, kd=0.6,
                 integral_limit=2.0, derivative_alpha=0.55,
                 output_limit=22.0, rate_limit=10.0, deadband=1.5,
                 emergency_output_limit=25.0, emergency_rate_limit=20.0,
                 emergency_error_px=30.0, emergency_streak_req=2,
                 authority=None):
        self.kp = kp; self.ki = ki; self.kd = kd
        self.integral_limit = integral_limit
        self.derivative_alpha = derivative_alpha
        self.output_limit = output_limit
        self.rate_limit = rate_limit
        self.deadband = deadband
        self.emergency_output_limit = emergency_output_limit
        self.emergency_rate_limit = emergency_rate_limit
        self.emergency_error_px = emergency_error_px
        self.emergency_streak_req = emergency_streak_req
        self.authority = authority if authority is not None else AuthorityManager(output_limit)
        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_derivative = 0.0
        self.prev_output = 0.0
        self.first_call = True
        self.last_p = self.last_i = self.last_d = self.last_raw = 0.0
        self.last_kp_mult = 1.0
        self.last_kd_mult = 1.0
        self.last_output_limit = output_limit
        self.last_rate_limit = rate_limit
        self.emergency_authority = False
        self.emg_streak = 0
        self.forced_emergency = False

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_derivative = 0.0
        self.prev_output = 0.0
        self.first_call = True
        self.authority.reset()
        self.emergency_authority = False
        self.emg_streak = 0
        self.forced_emergency = False

    def soft_reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_derivative = 0.0
        self.first_call = True

    def step(self, error, now=None):
        kp_eff = self.kp * (1.5 if self.authority.losing else 1.0)
        kd_eff = self.kd * (0.7 if self.authority.losing else 1.0)
        self.last_kp_mult = kp_eff / self.kp if self.kp else 1.0
        self.last_kd_mult = kd_eff / self.kd if self.kd else 1.0

        if self.forced_emergency:
            self.emg_streak = self.emergency_streak_req
            out_lim = self.emergency_output_limit
            rate_lim = self.emergency_rate_limit
            self.emergency_authority = True
        else:
            if abs(error) > self.emergency_error_px:
                self.emg_streak += 1
            else:
                self.emg_streak = 0
            if self.emg_streak >= self.emergency_streak_req:
                out_lim = self.emergency_output_limit
                rate_lim = self.emergency_rate_limit
                self.emergency_authority = True
            else:
                out_lim = self.output_limit
                rate_lim = self.rate_limit
                self.emergency_authority = False
        self.last_output_limit = out_lim
        self.last_rate_limit = rate_lim

        if abs(error) < self.deadband:
            e = np.sign(error) * (abs(error) - self.deadband)
        else:
            e = error - np.sign(error) * self.deadband

        p_term = kp_eff * e
        self.integral += e
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.integral

        raw_d = 0.0 if self.first_call else (error - self.prev_error)
        self.filtered_derivative = (self.derivative_alpha * raw_d
                                    + (1.0 - self.derivative_alpha) * self.filtered_derivative)
        d_term = kd_eff * self.filtered_derivative

        raw_output = p_term + i_term + d_term
        pid_output = out_lim * np.tanh(raw_output / out_lim)
        if self.authority.losing:
            self.integral = 0.0
        self.authority.step(error, self.prev_error, pid_output)

        delta = np.clip(pid_output - self.prev_output, -rate_lim, rate_lim)
        output = self.prev_output + delta
        output = float(np.clip(output, -25.0, 25.0))

        self.prev_error = error
        self.prev_output = output
        self.first_call = False
        self.last_p = p_term
        self.last_i = i_term
        self.last_d = d_term
        self.last_raw = raw_output
        return output


def largest_run_bounds(binary_row):
    nz = np.where(binary_row > 0)[0]
    if len(nz) == 0:
        return None
    if len(nz) == 1:
        return int(nz[0]), int(nz[0])
    runs = []
    start = nz[0]; prev = nz[0]
    for x in nz[1:]:
        if x != prev + 1:
            runs.append((start, prev)); start = x
        prev = x
    runs.append((start, prev))
    s, e = max(runs, key=lambda r: r[1] - r[0])
    return int(s), int(e)


def extract_centerline(seg_mask, prev_curve_smooth,
                       min_rows=20, coverage_threshold=0.18):
    gray = seg_mask if len(seg_mask.shape) == 2 else \
           cv2.cvtColor(seg_mask, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    row_centers = []
    for y in range(h - 1, -1, -1):
        b = largest_run_bounds(gray[y, :])
        if b is not None:
            s, e = b
            row_centers.append((int((s + e) // 2), y))
    coverage = np.count_nonzero(gray) / float(h * w)
    if len(row_centers) < min_rows or coverage < coverage_threshold:
        return 0.0, 0.0, 0.0, 0.0, [], False, coverage
    ys_arr = np.array([y for _, y in row_centers])
    y_bot = int(np.percentile(ys_arr, 98))
    y_top = int(np.percentile(ys_arr,  2))
    span = max(y_bot - y_top, 1)
    y_split_near = y_bot - int(span * 0.45)
    y_split_far  = y_top + int(span * 0.35)
    near_pts = [(cx, y) for cx, y in row_centers if y >= y_split_near]
    far_pts  = [(cx, y) for cx, y in row_centers if y <= y_split_far]

    def band_center(pts):
        if not pts: return None
        return int(np.median([cx for cx, _ in pts]))

    near_cx = band_center(near_pts)
    far_cx  = band_center(far_pts)
    near_error = (near_cx - w // 2) if near_cx is not None else 0
    far_error  = (far_cx  - w // 2) if far_cx  is not None else 0
    top_band = [cx for cx, y in row_centers if y <= y_split_far]
    bot_band = [cx for cx, y in row_centers if y >= y_split_near]
    top_cx = int(np.median(top_band)) if top_band else 0
    bot_cx = int(np.median(bot_band)) if bot_band else 0
    curve_error = top_cx - bot_cx
    curve_smooth = 0.90 * prev_curve_smooth + 0.10 * curve_error
    return near_error, far_error, curve_error, curve_smooth, \
           row_centers, True, coverage


def detect_asymmetry(gray, max_bias_px=45.0,
                     band_y0_frac=0.18, band_y1_frac=0.60):
    h, w = gray.shape
    bottom_cxs = []
    for y in range(int(h * 0.80), h, 4):
        nz = np.where(gray[y, :] > 0)[0]
        if len(nz) >= 2:
            bottom_cxs.append((nz[0] + nz[-1]) // 2)
    if not bottom_cxs:
        return 0.0
    ref_cx = int(np.median(bottom_cxs))
    lws, rws = [], []
    y0, y1 = int(h * band_y0_frac), int(h * band_y1_frac)
    for y in range(y0, y1, 4):
        nz = np.where(gray[y, :] > 0)[0]
        if len(nz) < 2:
            continue
        lw = ref_cx - nz[0]
        rw = nz[-1] - ref_cx
        if lw > 0 and rw > 0:
            lws.append(lw); rws.append(rw)
    if not lws:
        return 0.0
    lw = float(np.median(lws)); rw = float(np.median(rws))
    total = lw + rw
    if total < 40:
        return 0.0
    ratio = (rw - lw) / total
    return ratio * max_bias_px


def plan_speed(curve_smoothed, max_speed, min_speed=0.0, curve_scale=80.0):
    curve_ratio = min(abs(curve_smoothed) / curve_scale, 1.0)
    return max_speed - (max_speed - min_speed) * curve_ratio, curve_ratio


def error_speed_factor(error, start_px=8.0, full_px=35.0, min_factor=0.30):
    a = abs(error)
    if a <= start_px: return 1.0
    if a >= full_px: return min_factor
    t = (a - start_px) / (full_px - start_px)
    return 1.0 - t * (1.0 - min_factor)


class LaneFollower:
    def __init__(self):
        self.authority = AuthorityManager(
            output_limit=22.0, tau=0.35, sat_frac=0.9, sat_frames=3,
            lose_enter_px=30.0, lose_exit_px=20.0)
        self.pid = SteeringPID(
            kp=0.32, ki=0.0, kd=0.6,
            integral_limit=2.0, derivative_alpha=0.55,
            output_limit=22.0, rate_limit=10.0, deadband=1.5,
            emergency_output_limit=25.0, emergency_rate_limit=20.0,
            emergency_error_px=30.0,
            emergency_streak_req=EMERGENCY_STREAK_REQ,
            authority=self.authority)
        self.gate = OcclusionGate(
            err_step_px=ERR_STEP_PX, coverage_drop_frac=COV_DROP,
            jagged_px=JAGGED_PX, hard_step_px=HARD_STEP_PX,
            hangover_frames=OCCL_HANGOVER,
            far_outlier_px=FAR_OUTLIER_PX,
            far_outlier_near_px=FAR_OUTLIER_NEAR_PX,
            far_spread_px=FAR_SPREAD_PX, far_window=FAR_WINDOW)
        self.speed = MIN_SPEED
        self.reset_tracking()

    def reset_tracking(self):
        self.pid.reset()
        self.gate.reset()
        self.last_angle = 0.0
        self.curve_smooth = 0.0
        self.prev_valid = False
        self.prev_far_error = 0.0
        self.d_far_filt = 0.0
        self.asym_smooth = 0.0
        self.occl_streak = 0
        self.clean_streak = 0
        self.occl_emer_hold = 0

    def step(self, lane_mask, now, max_speed=MAX_SPEED,
             force_straight=False, near_only=False):
        max_speed = float(max_speed)
        slow_mode = (max_speed <= SLOW_SPEED + 0.5)

        gray = lane_mask
        near_e, far_e, curve_e, self.curve_smooth, row_pts, valid, coverage = \
            extract_centerline(gray, self.curve_smooth)
        if near_only:
            far_e = near_e

        far_delta = float(np.clip(far_e - near_e, -FAR_DELTA_CLIP, FAR_DELTA_CLIP))
        if valid and self.prev_valid:
            d_far_raw = float(far_e - self.prev_far_error)
        else:
            d_far_raw = 0.0
        d_far = float(np.clip(d_far_raw, -D_FAR_CLIP, D_FAR_CLIP))
        self.d_far_filt = (D_FAR_EMA_ALPHA * d_far
                           + (1.0 - D_FAR_EMA_ALPHA) * self.d_far_filt)
        lead = 0.5 * far_delta + LEAD_RATE_GAIN * self.d_far_filt
        blended = float(near_e) + lead
        self.prev_far_error = float(far_e)

        occl = self.gate.step(blended, coverage, row_pts, valid,
                              far_e=far_e, near_e=near_e)

        if occl:
            self.occl_streak += 1
            self.clean_streak = 0
            self.occl_emer_hold = EMG_HOLD_AFTER_OCCL
        else:
            self.clean_streak += 1
            self.occl_streak = max(0, self.occl_streak - 1)
            if self.occl_emer_hold > 0:
                self.occl_emer_hold -= 1

        if not occl and valid:
            asym_raw = detect_asymmetry(gray)
            self.asym_smooth = ASYM_EMA * self.asym_smooth + (1.0 - ASYM_EMA) * asym_raw
            if abs(near_e) < 5.0 and abs(self.asym_smooth) < ASYM_CAP:
                self.asym_smooth *= ASYM_DECAY
            self.asym_smooth = float(np.clip(self.asym_smooth, -ASYM_CAP, ASYM_CAP))

        if abs(self.asym_smooth) < ASYM_THRESHOLD:
            asym_applied = 0.0
        else:
            asym_applied = ASYM_STRENGTH * self.asym_smooth
        eff_error = blended + asym_applied

        forced_emer = self.occl_emer_hold > 0
        self.pid.forced_emergency = forced_emer

        if valid:
            if not self.prev_valid:
                self.pid.soft_reset()
                self.gate.reset()
                print("[pid] soft reset after invalid")
            if force_straight:
                angle = 0.0
                self.pid.reset()
            else:
                angle = self.pid.step(eff_error, now)

            target_speed, curve_ratio = plan_speed(self.curve_smooth, max_speed, MIN_SPEED)
            err_speed_mult = error_speed_factor(eff_error, start_px=8.0,
                                                full_px=35.0, min_factor=0.30)
            auth_speed = 0.6 if self.authority.losing else 1.0
            anom_speed = 0.5 if abs(self.asym_smooth) > 15 else 1.0
            speed_mult = min(auth_speed, anom_speed)
            target_speed = max(target_speed * err_speed_mult * speed_mult, 0.0)

            if slow_mode:
                speed = target_speed
            else:
                speed = 0.7 * self.speed + 0.3 * target_speed
            speed = min(speed, max_speed)
            self.last_angle = angle
        else:
            angle = 0.0 if force_straight else self.last_angle
            curve_ratio = 0.0
            if slow_mode:
                speed = max_speed
            else:
                speed = 0.85 * self.speed
            speed = min(speed, max_speed)

        self.speed = float(speed)
        self.prev_valid = valid

        info = dict(
            near=near_e, far=far_e, curve_smooth=self.curve_smooth, lead=lead,
            blended=blended, eff_error=eff_error, asym=self.asym_smooth,
            occl=bool(occl), forced_emer=bool(forced_emer),
            occl_emer_hold=self.occl_emer_hold,
            emer_auth=bool(self.pid.emergency_authority),
            losing=bool(self.authority.losing),
            valid=bool(valid), coverage=coverage, curve_ratio=curve_ratio,
            row_pts=row_pts, max_speed=max_speed, slow_mode=slow_mode,
        )
        return float(angle), self.speed, info


# ==============================================================
# VOTER
# ==============================================================
class SignVoter:
    def __init__(self, activate_frames=VOTER_ACTIVATE_FRAMES,
                 timeout_frames=VOTER_TIMEOUT_FRAMES):
        self.activate_frames = activate_frames
        self.timeout_frames = timeout_frames
        self.counts = {}
        self.activated = set()
        self.candidate_cnt = {}
        self.committed = None
        self.last_seen_frame = -1
        self.just_timed_out = False

    def update(self, classes, frame_idx):
        self.just_timed_out = False
        classes_set = set(classes)

        if classes_set:
            self.last_seen_frame = frame_idx

        if (self.last_seen_frame >= 0 and
                (frame_idx - self.last_seen_frame) > self.timeout_frames):
            print(f"[VOTER] timeout {self.timeout_frames}f — reset")
            self.reset()
            self.just_timed_out = True
            return

        for cls in list(self.candidate_cnt.keys()):
            if cls not in classes_set:
                self.candidate_cnt[cls] = 0

        for cls in classes_set:
            self.candidate_cnt[cls] = self.candidate_cnt.get(cls, 0) + 1
            if self.candidate_cnt[cls] >= self.activate_frames and cls not in self.activated:
                self.activated.add(cls)
                self.counts[cls] = 0
                print(f"[VOTER] activate {cls}")

        for cls in classes_set:
            if cls in self.activated:
                self.counts[cls] = self.counts.get(cls, 0) + 1

        if self.counts:
            best = max(self.counts, key=self.counts.get)
            if best != self.committed:
                print(f"[VOTER] commit={best} counts={self.counts}")
                self.committed = best

    def reset(self):
        self.counts = {}
        self.activated = set()
        self.candidate_cnt = {}
        self.committed = None
        self.last_seen_frame = -1


# ==============================================================
# TOPO RESET DETECTOR
# ==============================================================
class TopoResetDetector:
    def __init__(self, debounce=RESET_DEBOUNCE_FRAMES):
        self.debounce = debounce
        self.armed = False
        self.stable_cnt = 0

    def update(self, latch):
        nbits = int(latch['left']) + int(latch['straight']) + int(latch['right'])

        if nbits >= 2:
            self.armed = True
            self.stable_cnt = 0
            return False

        if self.armed and nbits == 1:
            self.stable_cnt += 1
            if self.stable_cnt >= self.debounce:
                self.armed = False
                self.stable_cnt = 0
                print(f"[RESET] topo → 1 bit confirmed")
                return True
        else:
            self.stable_cnt = 0

        return False

    def reset(self):
        self.armed = False
        self.stable_cnt = 0


# ==============================================================
# FORCE TURN LOGIC
# ==============================================================
def decide_force_dir(commit, topo):
    if commit is None:
        return 0
    if commit == 'turn_left':   return -1
    if commit == 'turn_right':  return +1
    if commit == 'straight':    return 0
    has_S = bool(topo['straight'])
    if commit == 'no_turn_left':
        return 0 if has_S else +1
    if commit == 'no_turn_right':
        return 0 if has_S else -1
    return 0


def check_turn_condition(seg_mask, H, W):
    y_lo = int(H * TURN_CONDITION_Y_LO_RATIO)
    region = seg_mask[y_lo:H, :]
    if region.size == 0:
        return False
    coverage = (region > 0).sum() / region.size
    return coverage >= TURN_CONDITION_COVERAGE


# ==============================================================
# MAIN
# ==============================================================
if __name__ == "__main__":
    # os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    # log_file = open(LOG_PATH, "w", newline="", buffering=1)
    # log_writer = csv.writer(log_file)
    # log_writer.writerow([
    #     "t", "frame", "commit", "counts", "sign_seen_rec",
    #     "force_active", "force_dir",
    #     "topo_L", "topo_S", "topo_R",
    #     "cover_bottom",
    #     "heading", "d_heading",
    #     "max_speed", "speed", "angle",
    # ])
    # print(f"[log] {LOG_PATH}")

    topo_latch     = TopoLatch(confirm_frames=TOPO_CONFIRM_FRAMES)
    voter          = SignVoter()
    reset_detector = TopoResetDetector()
    follower       = LaneFollower()

    force_active = False
    force_dir    = 0
    force_heading_0 = None
    frame_idx    = 0
    last_sign_seen_frame = -10**9

    try:
        while True:
            frame_idx += 1
            state_raw = GetStatus()
            cur_heading = (float(state_raw.get('Heading', 0.0))
                           if isinstance(state_raw, dict) else 0.0)
            raw_image = GetRaw()

            seg_mask, lane_mask = get_road_masks(raw_image)
            H, W = seg_mask.shape

            # ---------- Topo ----------
            skel_raw = skeleton_topology(seg_mask, W, H)
            topo = topo_latch.update(skel_raw)

            # ---------- Sign ----------
            sign_classes = sign_detect(raw_image)
            if sign_classes:
                last_sign_seen_frame = frame_idx
            sign_seen_recently = (frame_idx - last_sign_seen_frame) <= SIGN_HOLD_FRAMES

            voter.update(sign_classes, frame_idx)
            commit = voter.committed

            # ---------- Topo reset ----------
            if reset_detector.update(topo):
                voter.reset()
                force_active = False
                force_dir = 0
                force_heading_0 = None
                follower.reset_tracking()
                commit = None

            if voter.just_timed_out:
                force_active = False
                force_dir = 0
                force_heading_0 = None
                follower.reset_tracking()

            # ---------- Bottom coverage ----------
            y_lo_cov = int(H * TURN_CONDITION_Y_LO_RATIO)
            region_cov = seg_mask[y_lo_cov:H, :]
            cover_bottom = ((region_cov > 0).sum() / region_cov.size
                            if region_cov.size > 0 else 0.0)

            # ---------- Force early reset theo heading ----------
            d_heading = 0.0
            if force_active and force_heading_0 is not None:
                d_heading = angle_diff_deg(cur_heading, force_heading_0)
                if abs(d_heading) >= HEADING_RESET_DEG:
                    print(f"[RESET-HEADING] Δ={d_heading:+.1f}° ≥ "
                          f"{HEADING_RESET_DEG}° → reset sớm")
                    voter.reset()
                    reset_detector.reset()
                    force_active = False
                    force_dir = 0
                    force_heading_0 = None
                    follower.reset_tracking()
                    commit = None

            # ---------- max speed ----------
            cur_max_speed = compute_max_speed(force_active, commit, sign_seen_recently)

            # ---------- CONTROL ----------
            if force_active:
                angle = force_dir * FORCE_ANGLE
                speed = SLOW_SPEED
                AVControl(speed, angle)
                tag = "FORCE"
            else:
                force_candidate = decide_force_dir(commit, topo)

                if force_candidate != 0 and check_turn_condition(seg_mask, H, W):
                    force_active = True
                    force_dir = force_candidate
                    force_heading_0 = cur_heading
                    follower.reset_tracking()
                    angle = force_dir * FORCE_ANGLE
                    speed = SLOW_SPEED
                    AVControl(speed, angle)
                    tag = "FORCE-START"
                    print(f"[FORCE] start dir={force_dir} commit={commit} "
                          f"hdg0={force_heading_0:.1f}")
                else:
                    angle, speed, info = follower.step(
                        lane_mask, time.time(), max_speed=cur_max_speed)
                    AVControl(speed, angle)
                    if force_candidate != 0:
                        tag = "SLOW-PID"
                    elif commit is not None:
                        tag = "COMMIT-PID"
                    else:
                        tag = "PID"

            print(f"[{tag}] f={frame_idx} commit={commit} "
                  f"counts={voter.counts} "
                  f"topo=L{int(topo['left'])}S{int(topo['straight'])}R{int(topo['right'])} "
                  f"cov={cover_bottom:.2f} "
                  f"hdg={cur_heading:.1f} dH={d_heading:+.1f} "
                  f"seen_rec={int(sign_seen_recently)} "
                  f"max_spd={cur_max_speed:.0f} "
                  f"force={int(force_active)}({force_dir}) "
                  f"ang={angle:+.2f} spd={speed:.2f}")

            # log_writer.writerow([
            #     f"{time.time():.3f}", frame_idx,
            #     commit if commit else "",
            #     str(voter.counts),
            #     int(sign_seen_recently),
            #     int(force_active), force_dir,
            #     int(topo['left']), int(topo['straight']), int(topo['right']),
            #     f"{cover_bottom:.3f}",
            #     f"{cur_heading:.1f}",
            #     f"{d_heading:+.1f}",
            #     f"{cur_max_speed:.1f}",
            #     f"{speed:.2f}", f"{angle:+.2f}",
            # ])

            if ENABLE_GUI:
                cv2.imshow('Raw', raw_image)
                vis = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2BGR)
                if SKEL_OK and seg_mask is not None:
                    b = clean_mask_for_skeleton(seg_mask)
                    if b.sum() > 0:
                        sk = skeletonize(b > 0).astype(np.uint8)
                        vis[sk > 0] = (0, 255, 255)
                cv2.putText(vis,
                            f"commit={commit} seen_rec={int(sign_seen_recently)} "
                            f"max_spd={cur_max_speed:.0f}",
                            (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.putText(vis,
                            f"topo L{int(topo['left'])}S{int(topo['straight'])}R{int(topo['right'])}",
                            (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                cv2.putText(vis,
                            f"counts={voter.counts} cov={cover_bottom:.2f} "
                            f"spd={speed:.2f} max_spd={cur_max_speed:.0f}",
                            (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
                cv2.putText(vis,
                            f"hdg={cur_heading:.1f} dH={d_heading:+.1f} "
                            f"force={int(force_active)}({force_dir})",
                            (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1)
                cv2.imshow('Mask', vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        print('Closing...')
        log_file.close()
        AVControl(0, 0)
        CloseSocket()
        cv2.destroyAllWindows()

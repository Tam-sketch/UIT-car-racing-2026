# ==============================================================
# UIT CAR RACING 2026 - COMBINED v3 + SKELETON GATE + FIX LOOP + T-JUNCTION v2
#
# Fix log:
#   Fix 1: SignVoter.block() + category lock cho no_turn_*
#   Fix 2: Navigator normal → slow_down CHỈ khi potential_dir != 0
#   Fix 3: Navigator potential_dir == 0 dùng block() thay vì reset()
#   Fix 4: T-junction detection — L+R arm dài + junction ở bottom + mask wide
#   Fix 5: Debug hiển thị TOPOLOGY lên màn hình Gate Debug
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
    print("[WARN] skimage not found — gate disabled")


# ==============================================================
# CONFIG
# ==============================================================
ROAD_MODEL_PATH = "/workspace/best12.pt"
SIGN_MODEL_PATH = "/workspace/sign.pt"

ROAD_IMGSZ = (192, 320)
SIGN_IMGSZ = 640
SIGN_CONF  = 0.85

MIN_FRAMES = 4
MAX_GAP    = 1
BIAS_RATIO = 0.3
SIGN_TIMEOUT_FRAMES = 600

BLOCKED_GONE_THRESH = 0

Y_LO_RATIO = 0.2
Y_HI_RATIO = 0.7
MIN_BRANCH_WIDTH = 12
TOP_HALF_RATIO = 0.55
TOP_HALF_EMPTY_THRESH = 0.02

SKEL_MIN_ARM_SIZE        = 5
SKEL_JUNCTION_DILATE     = 3
SKEL_S_VERIFY_X_HALF     = 15
SKEL_S_VERIFY_ROW_MIN_PX = 3
SKEL_S_VERIFY_MARGIN     = 8
MASK_CLOSE_KERNEL        = 5
MASK_MIN_AREA_RATIO      = 0.005

SKEL_GATE_ENABLED       = True
SKEL_GATE_MIN_ARM_LEN   = 10
SKEL_GATE_MIN_ARM_TILT  = 0.30
SKEL_GATE_Y_LO_RATIO    = 0.10
SKEL_GATE_Y_HI_RATIO    = 0.90

# [NEW] T-junction gate
T_JUNC_Y_RATIO      = 0.75
T_JUNC_MIN_ARM_LEN  = 18
T_JUNC_WIDE_RATIO   = 1.25
T_JUNC_USE_WIDE     = True

BRANCH_CHECK_Y_LO_RATIO = 0.7
BRANCH_CHECK_Y_HI_RATIO = 1.00
BRANCH_ZONE_X_LO_RATIO  = 0.1
BRANCH_ZONE_X_HI_RATIO  = 0.9
BRANCH_COVERAGE_THRESH  = 0.8
BRANCH_CONFIRM_FRAMES   = 1
BRANCH_MIN_WIDTH_RATIO  = 1.5
BRANCH_REQUIRE_TOPO     = True

SLOW_SPEED = 15
TURN_SPEED = 12

TURN_FF_BEFORE      = 25
TURN_PROP_K         = 0.15
TURN_MAX_ANGLE      = 25
TURN_CLAMP_SAME_DIR = True
STRAIGHT_ERROR_DEADBAND = 2.0

STOP_HOLD_FRAMES = 0
CROSS_ZERO_DEADBAND = 5.0
TURN_STABLE_THRESH  = 12.0

NEW_LANE_CENTER_THRESH = 65.0
NEW_LANE_STABLE_COUNT  = 2
NEW_LANE_MIN_FRAMES    = 8
NEW_LANE_WIDTH_LO      = 0.5
NEW_LANE_WIDTH_HI      = 1.5
MAX_TURN_FRAMES        = 90

TURN_MASK_LOST_TRIGGER_FRAMES = 20
TURN_MASK_RECOVER_FRAMES = 2

BIAS_DECAY_FACTOR = 0.3

COOLDOWN_FRAMES                = 0
COOLDOWN_SPEED_CAP             = 18
COOLDOWN_FORCE_STRAIGHT_FRAMES = 0
POST_TURN_NEAR_ONLY_FRAMES     = 0
TURN_EXIT_FAR_THRESH           = 65.0

INTERSECTION_RATIO   = 1.35
INTERSECTION_CONFIRM = 2
INTERSECTION_MIN_ABS = 120

TOPO_TURN_CONFIRM = 2

EARLY_TOPO_Y_LO_RATIO = 0.08
EARLY_TOPO_Y_HI_RATIO = 0.72
EARLY_TOPO_CONFIRM = 3
EARLY_TOPO_TTL_FRAMES = 45
EARLY_TOPO_MIN_WIDTH_RATIO = 0.28
EARLY_TOPO_MAX_WIDTH_RATIO = 0.85
EARLY_TRIGGER_MIN_CUR_W = 120
EARLY_TRIGGER_WIDTH_RATIO = 1.30
EARLY_TRIGGER_CONFIRM = 2

MAX_SPEED = 26.0
MIN_SPEED = 0.0

ASYM_STRENGTH = 0.20
ASYM_THRESHOLD = 8.0
ASYM_EMA = 0.7
ASYM_CAP = 12.0
ASYM_DECAY = 0.80
FAR_DELTA_CLIP = 25.0
D_FAR_CLIP = 6.0
D_FAR_EMA_ALPHA = 0.4
LEAD_RATE_GAIN = 0.8

HARD_STEP_PX = 30.0
ERR_STEP_PX  = 18.0
COV_DROP     = 0.22
JAGGED_PX    = 15.0
OCCL_HANGOVER = 4
FAR_OUTLIER_PX = 40.0
FAR_OUTLIER_NEAR_PX = 20.0
FAR_SPREAD_PX = 40.0
FAR_WINDOW = 5

EMERGENCY_STREAK_REQ = 2
EMG_HOLD_AFTER_OCCL = 5

LOG_PATH = "/workspace/my_code/pid_log_gate.csv"
LOG_FLUSH_EVERY = 10
ENABLE_GUI = os.environ.get("ENABLE_GUI", "1") != "0"


# ==============================================================
# LOAD MODELS
# ==============================================================
print("Loading models...")
road_model = YOLO(ROAD_MODEL_PATH)
sign_model = YOLO(SIGN_MODEL_PATH)
print(f"Road model classes: {road_model.names}")
print(f"Sign model classes: {sign_model.names}")


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
        empty = np.zeros(ROAD_IMGSZ, dtype=np.uint8)
        return empty, empty
    masks = results[0].masks.data.cpu().numpy()
    full = (np.sum(masks, axis=0) > 0).astype(np.uint8) * 255
    return full, keep_largest_component(full)


# ==============================================================
# SKELETON GRAPH + GATE
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


def trace_arm_polyline(arm_mask, junction_px):
    ys, xs = np.where(arm_mask > 0)
    if len(xs) == 0:
        return []
    pixels = set(zip(xs.tolist(), ys.tolist()))
    jx, jy = junction_px
    dists = (xs - jx) ** 2 + (ys - jy) ** 2
    start_idx = int(np.argmin(dists))
    start = (int(xs[start_idx]), int(ys[start_idx]))
    visited = {start}
    polyline = [start]
    current = start
    while True:
        x, y = current
        found = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nb = (x + dx, y + dy)
                if nb in pixels and nb not in visited:
                    visited.add(nb)
                    polyline.append(nb)
                    current = nb
                    found = True
                    break
            if found:
                break
        if not found:
            break
    return polyline


def build_skeleton_graph(mask, W, H):
    result = {'ok': False, 'junctions': [], 'arms': {},
              'n_junctions': 0, 'n_arms_total': 0}
    if not SKEL_OK:
        return result
    if mask is None or W <= 0 or H <= 0:
        return result

    binary = clean_mask_for_skeleton(mask)
    if binary.sum() < MASK_MIN_AREA_RATIO * W * H:
        return result

    skel = skeletonize(binary > 0).astype(np.uint8)
    if skel.sum() < 15:
        return result

    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neigh = cv2.filter2D(skel, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    jmask = (skel == 1) & (neigh >= 3)

    if not jmask.any():
        result['ok'] = True
        return result

    jys, jxs = np.where(jmask)
    jpoints = list(zip(jxs.tolist(), jys.tolist()))
    used = [False] * len(jpoints)
    junction_clusters = []
    for i, (jx, jy) in enumerate(jpoints):
        if used[i]:
            continue
        cluster = [(jx, jy)]
        used[i] = True
        for j in range(i + 1, len(jpoints)):
            if used[j]:
                continue
            jx2, jy2 = jpoints[j]
            if (jx2 - jx) ** 2 + (jy2 - jy) ** 2 < SKEL_JUNCTION_DILATE ** 2:
                cluster.append((jx2, jy2))
                used[j] = True
        cj_x = int(np.mean([p[0] for p in cluster]))
        cj_y = int(np.mean([p[1] for p in cluster]))
        junction_clusters.append((cj_x, cj_y))

    result['n_junctions'] = len(junction_clusters)

    jmask_u8 = (jmask.astype(np.uint8) * 255)
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SKEL_JUNCTION_DILATE, SKEL_JUNCTION_DILATE))
    jmask_dil = cv2.dilate(jmask_u8, kd, iterations=1)
    skel_arms = skel.copy()
    skel_arms[jmask_dil > 0] = 0
    num, labels = cv2.connectedComponents(skel_arms, connectivity=8)

    for jidx, (jx, jy) in enumerate(junction_clusters):
        arms_local = {'L': [], 'S': [], 'R': []}
        for lid in range(1, num):
            arm_mask = (labels == lid).astype(np.uint8)
            ys_a, xs_a = np.where(arm_mask > 0)
            if len(xs_a) < SKEL_MIN_ARM_SIZE:
                continue
            dists = (xs_a - jx) ** 2 + (ys_a - jy) ** 2
            if float(np.min(dists)) > SKEL_JUNCTION_DILATE ** 2:
                continue
            poly = trace_arm_polyline(arm_mask, (jx, jy))
            if len(poly) < SKEL_MIN_ARM_SIZE:
                continue
            end_x, end_y = poly[-1]
            vx = end_x - jx
            vy = end_y - jy
            if abs(vx) < SKEL_GATE_MIN_ARM_TILT * abs(vy) + 5:
                if vy < 0:
                    key = 'S'
                else:
                    continue
            elif vx < 0:
                key = 'L'
            else:
                key = 'R'
            arms_local[key].append(poly)
        result['arms'][jidx] = arms_local

    total = sum(len([a for a in arms.values() if a])
                for arms in result['arms'].values())
    result['n_arms_total'] = total
    result['junctions'] = [(jx, jy, i) for i, (jx, jy) in enumerate(junction_clusters)]
    result['ok'] = True
    return result


def compute_skeleton_gate(graph, H):
    if not SKEL_GATE_ENABLED:
        return None
    if not graph.get('ok'):
        return None
    if graph['n_junctions'] == 0:
        return None

    has_any_valid_junction = False
    for jx, jy, jidx in graph['junctions']:
        if jy < SKEL_GATE_Y_LO_RATIO * H or jy > SKEL_GATE_Y_HI_RATIO * H:
            continue
        has_any_valid_junction = True
        arms = graph['arms'].get(jidx, {})
        has_L = any(len(p) >= SKEL_GATE_MIN_ARM_LEN for p in arms.get('L', []))
        has_R = any(len(p) >= SKEL_GATE_MIN_ARM_LEN for p in arms.get('R', []))
        if has_L or has_R:
            return True
    if has_any_valid_junction:
        return False
    return None


def skeleton_verify_straight(mask, W, H):
    info = {'has_straight': True, 'ok': False, 'jy': -1, 'n_arms': 0}
    if not SKEL_OK:
        return True, info
    if mask is None or W <= 0 or H <= 0:
        return True, info
    binary = clean_mask_for_skeleton(mask)
    if binary.sum() < MASK_MIN_AREA_RATIO * W * H:
        return True, info
    skel = skeletonize(binary > 0).astype(np.uint8)
    if skel.sum() < 15:
        return True, info
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neigh = cv2.filter2D(skel, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    jmask = (skel == 1) & (neigh >= 3)
    if not jmask.any():
        info['has_straight'] = True
        info['ok'] = True
        return True, info
    jys, jxs = np.where(jmask)
    jy = int(np.max(jys))
    info['jy'] = jy
    info['n_arms'] = len(jys)
    info['ok'] = True
    return True, info


def detect_branch_bottom(mask, W, H):
    binary = (mask > 0).astype(np.uint8)
    y_lo = int(H * BRANCH_CHECK_Y_LO_RATIO)
    y_hi = min(H, int(H * BRANCH_CHECK_Y_HI_RATIO))
    if y_hi <= y_lo:
        return 0.0, 0.0
    region = binary[y_lo:y_hi, :]
    if region.size == 0:
        return 0.0, 0.0
    x_lo = int(W * BRANCH_ZONE_X_LO_RATIO)
    x_hi = int(W * BRANCH_ZONE_X_HI_RATIO)
    if x_hi <= x_lo:
        return 0.0, 0.0
    lz = region[:, :x_lo]; rz = region[:, x_hi:]
    return ((float(lz.sum()) / lz.size) if lz.size > 0 else 0.0,
            (float(rz.sum()) / rz.size) if rz.size > 0 else 0.0)


# ==============================================================
# SIGN VOTER
# ==============================================================
class SignVoter:
    def __init__(self, min_frames=4, max_gap=3, timeout=SIGN_TIMEOUT_FRAMES):
        self.min_frames = min_frames
        self.max_gap = max_gap
        self.timeout = timeout
        self.current_class = None
        self.count = 0
        self.gap = 0
        self.committed = None
        self._age = 0
        self._commit_age = 0
        self._prev_classes = set()
        self._blocked_until_gone = None
        self._blocked_gone_frames = 0
        self._category_lock = None
        self.BLOCKED_GONE_THRESH = BLOCKED_GONE_THRESH

    def block(self, cls):
        if cls is None:
            return
        self._blocked_until_gone = cls
        self._blocked_gone_frames = 0
        if cls.startswith('no_turn_'):
            self._category_lock = 'no_turn'
        else:
            self._category_lock = None
        self.current_class = None
        self.count = 0
        self.gap = 0
        self.committed = None
        self._age = 0
        self._commit_age = 0
        print(f"[VOTER] block '{cls}' until sign gone")

    def update(self, classes):
        if self._blocked_until_gone is not None:
            if self._blocked_until_gone in classes:
                self.committed = None
                self._blocked_gone_frames = 0
                self._prev_classes = set(classes)
                return
            else:
                self._blocked_gone_frames += 1
                if self._category_lock == 'no_turn':
                    any_no_turn = any(c.startswith('no_turn_') for c in classes)
                    if any_no_turn and self._blocked_gone_frames < self.BLOCKED_GONE_THRESH:
                        self.committed = None
                        self._prev_classes = set(classes)
                        return
                if self._blocked_gone_frames >= self.BLOCKED_GONE_THRESH:
                    print(f"[VOTER] unblock '{self._blocked_until_gone}' "
                          f"(gone {self._blocked_gone_frames} frames)")
                    self._blocked_until_gone = None
                    self._category_lock = None
                    self._blocked_gone_frames = 0
                else:
                    self.committed = None
                    self._prev_classes = set(classes)
                    return

        current_set = set(classes)
        newly_appeared = current_set - self._prev_classes
        if newly_appeared:
            cls = list(newly_appeared)[0]
            print(f"[VOTER] new sign appeared: {cls} (all={classes})")
        elif classes:
            cls = classes[0]
        else:
            cls = None
        self._prev_classes = current_set

        if self.committed is not None:
            self._age += 1
            self._commit_age += 1
            if self._age >= self.timeout:
                print(f"[VOTER] timeout — drop {self.committed}")
                self.committed = None
                self._age = 0
                self._commit_age = 0

        if cls is None:
            self.gap += 1
            if self.gap > self.max_gap:
                self.current_class = None
                self.count = 0
            return

        if cls == self.current_class:
            self.count += 1
            self.gap = 0
        else:
            self.current_class = cls
            self.count = 1
            self.gap = 0

        if self.count >= self.min_frames:
            if self.gap == 0 and self.committed != cls:
                print(f"[VOTER] committed={cls} (count={self.count})")
                self.committed = cls
                self._age = 0
                self._commit_age = 0

    def get_committed(self): return self.committed
    def get_commit_age(self): return self._commit_age

    def reset(self):
        self.current_class = None; self.count = 0; self.gap = 0
        self.committed = None; self._age = 0; self._commit_age = 0
        self._prev_classes = set()


class IntersectionDetector:
    def __init__(self, ratio_threshold=INTERSECTION_RATIO,
                 confirm_frames=INTERSECTION_CONFIRM,
                 min_abs_width=INTERSECTION_MIN_ABS):
        self.ratio_threshold = ratio_threshold
        self.confirm_frames = confirm_frames
        self.min_abs_width = min_abs_width
        self._history = deque(maxlen=confirm_frames)
        self.last_ratio = 0.0

    def update(self, base_width, current_width):
        if base_width is None or base_width <= 0 or current_width is None:
            self._history.append(False); return False
        if current_width < self.min_abs_width:
            self._history.append(False); return False
        ratio = current_width / base_width
        self.last_ratio = ratio
        self._history.append(ratio >= self.ratio_threshold)
        return len(self._history) == self.confirm_frames and all(self._history)

    def reset(self):
        self._history.clear(); self.last_ratio = 0.0


def _compute_top_half_stats(binary, H):
    top_h = max(1, int(H * TOP_HALF_RATIO))
    tz = binary[:top_h, :]
    cov = (float(np.count_nonzero(tz)) / tz.size) if tz.size else 0.0
    return (cov < TOP_HALF_EMPTY_THRESH), cov


def get_topology(mask, W, H):
    if mask is None or W <= 0 or H <= 0:
        return {'left': False, 'straight': True, 'right': False,
                'straight_ratio': 1.0, 'top_half_empty': False, 'top_half_coverage': 1.0}
    binary = (mask > 0).astype(np.uint8)
    y0b = int(H * 0.72); y1b = int(H * 0.95)
    Ls, Rs = [], []
    for y in range(y0b, y1b, 3):
        xs = np.flatnonzero(binary[y])
        if xs.size >= 2:
            w = int(xs[-1] - xs[0])
            if int(W * 0.15) <= w <= int(W * 0.90):
                Ls.append(int(xs[0])); Rs.append(int(xs[-1]))
    LL = float(np.median(Ls)) if Ls else 0.25 * W
    RR = float(np.median(Rs)) if Rs else 0.75 * W
    if (RR - LL) < max(12, 0.20 * W) or (RR - LL) > 0.85 * W:
        LL, RR = 0.25 * W, 0.75 * W
    mg = max(4.0, 0.02 * W)
    lb = max(0.0, LL - mg); rb = min(float(W), RR + mg)
    y0 = max(0, int(H * Y_LO_RATIO)); y1 = min(H - 1, int(H * Y_HI_RATIO))
    if y1 <= y0:
        te, tc = _compute_top_half_stats(binary, H)
        return {'left': False, 'straight': True, 'right': False,
                'straight_ratio': 1.0, 'top_half_empty': te, 'top_half_coverage': tc}
    ys = np.linspace(y0, y1, 28).astype(int)
    lh, ch, rh = [], [], []
    for y in ys:
        row = binary[y]
        lx = max(0, min(W, int(round(lb)))); rx = max(lx, min(W, int(round(rb))))
        lc = (np.count_nonzero(row[:lx]) / lx) if lx > 0 else 0
        cc = (np.count_nonzero(row[lx:rx]) / (rx - lx)) if rx > lx else 0
        rc = (np.count_nonzero(row[rx:]) / (W - rx)) if rx < W else 0
        lh.append(lc >= 0.06); ch.append(cc >= 0.05); rh.append(rc >= 0.06)
    def lr(v):
        b = c = 0
        for x in v:
            c = c + 1 if x else 0
            b = max(b, c)
        return b
    lrt = sum(lh) / len(lh); crt = sum(ch) / len(ch); rrt = sum(rh) / len(rh)
    left = lrt >= 0.18 and lr(lh) >= 4
    right = rrt >= 0.18 and lr(rh) >= 4
    straight = crt >= 0.20
    if not left and not straight and not right:
        straight = True
    te, tc = _compute_top_half_stats(binary, H)
    return {'left': bool(left), 'straight': bool(straight), 'right': bool(right),
            'straight_ratio': float(crt), 'top_half_empty': bool(te), 'top_half_coverage': float(tc)}


def decide_bias(committed, topo, base_width, scale=1.0):
    if base_width is None or base_width <= 0 or committed is None:
        return 0
    b = 0
    if committed == 'turn_left': b = +BIAS_RATIO * base_width
    elif committed == 'turn_right': b = -BIAS_RATIO * base_width
    elif committed == 'no_turn_left': b = -BIAS_RATIO * base_width if topo['right'] else 0
    elif committed == 'no_turn_right': b = +BIAS_RATIO * base_width if topo['left'] else 0
    return int(b * scale)


class LaneWidthEstimator:
    def __init__(self):
        self.base_width = None; self.current_width = None
    def measure_width(self, gray_image, step=5, fraction=2):
        H, W = gray_image.shape
        my = H - H // 3; ny = H - H // fraction
        ws = []
        for y in range(my, ny, -step):
            nz = np.where(gray_image[y, :] > 0)[0]
            if len(nz) > 1: ws.append(nz[-1] - nz[0])
        if ws:
            self.current_width = np.median(ws)
            if self.base_width is None: self.base_width = self.current_width
        return self.current_width
    def reset_baseline(self): self.base_width = None
    def is_wide_change(self, threshold=0.2):
        if self.base_width is None or self.current_width is None: return False
        return self.current_width > self.base_width * (1 + threshold)


# ==============================================================
# NAVIGATOR
# ==============================================================
class SlowStopTurnNavigator:
    def __init__(self, voter):
        self.voter = voter
        self.state = 'normal'
        self._cnt = 0
        self._turn_dir = 0
        self._pending_turn_dir = 0
        self._final_commit = None
        self._stable_cnt = 0
        self._initial_sign = 0
        self._crossed_zero = False
        self._bias_decay = False
        self.just_entered_cooldown = False
        self._last_ff_used = 0.0
        self._last_angle_raw = 0.0
        self._last_exit_reason = ""
        self._new_lane_cnt = 0
        self._new_lane_offset = -1.0
        self.heading_ok = True
        self._turn_mask_lost_cnt = 0
        self._turn_mask_recover_cnt = 0
        self._turn_mask_wait = False
        self._topo_match_cnt = 0
        self._forced_opposite = False
        self._forced_mask_cnt = 0
        self._last_gate_block_frame = -1

    def _is_T_junction(self, graph, H=192, base_w=None, cur_w=None):
        """[FIX v2] T-junction THỰC: L+R arm dài + junction ở gần + mask wide."""
        if graph is None or not graph.get('ok'):
            return False

        has_L = False
        has_R = False
        junction_close = False

        for jx, jy, jidx in graph.get('junctions', []):
            if jy >= T_JUNC_Y_RATIO * H:
                junction_close = True
            arms = graph['arms'].get(jidx, {})
            if any(len(p) >= T_JUNC_MIN_ARM_LEN for p in arms.get('L', [])):
                has_L = True
            if any(len(p) >= T_JUNC_MIN_ARM_LEN for p in arms.get('R', [])):
                has_R = True

        wide_enough = True
        if T_JUNC_USE_WIDE and base_w and base_w > 0 and cur_w:
            wide_enough = (cur_w >= T_JUNC_WIDE_RATIO * base_w)

        return has_L and has_R and junction_close and wide_enough

    def update(self, sign_classes, committed, topo, mask_trigger,
               blended_error, bottom_cx, W, base_w, cur_w,
               is_real_straight=None,
               branch_lcov=0.0, branch_rcov=0.0,
               skel_gate=None, frame_idx=0,
               graph=None, H=192):
        self.just_entered_cooldown = False

        # --- NORMAL ---
        if self.state == 'normal':
            if committed is not None:
                potential_dir = self._decide_turn(
                    committed, topo,
                    is_real_straight=is_real_straight,
                    branch_lcov=branch_lcov, branch_rcov=branch_rcov,
                    base_w=base_w, cur_w=cur_w,
                    graph=graph, H=H,
                )
                if potential_dir != 0:
                    print(f"[NAV] → SLOW_DOWN (sign={committed}, dir={potential_dir})")
                    self.state = 'slow_down'
                    self._cnt = 0
                    self._forced_mask_cnt = 0
                elif committed in ('no_turn_left', 'no_turn_right', 'straight'):
                    self.voter.block(committed)
            return 0, 0, False

        # --- SLOW_DOWN ---
        if self.state == 'slow_down':
            if committed is None:
                print(f"[NAV] slow_down → normal (commit lost)")
                self.state = 'normal'
                self._cnt = 0
                self._topo_match_cnt = 0
                self._forced_mask_cnt = 0
                return 0, 0, False

            potential_dir = self._decide_turn(
                committed, topo,
                is_real_straight=is_real_straight,
                branch_lcov=branch_lcov, branch_rcov=branch_rcov,
                base_w=base_w, cur_w=cur_w,
                graph=graph, H=H)

            if self._forced_opposite and mask_trigger:
                self._forced_mask_cnt += 1
            else:
                self._forced_mask_cnt = 0

            topo_match = False
            if potential_dir < 0 and topo['left']: topo_match = True
            elif potential_dir > 0 and topo['right']: topo_match = True
            if topo_match: self._topo_match_cnt += 1
            else: self._topo_match_cnt = 0

            safe_geometry = (cur_w is not None and base_w is not None
                             and cur_w >= EARLY_TRIGGER_MIN_CUR_W
                             and cur_w >= base_w * EARLY_TRIGGER_WIDTH_RATIO)
            ctm = ((potential_dir < 0 and topo.get('left')) or
                   (potential_dir > 0 and topo.get('right')))
            topo_trigger = (self._topo_match_cnt >= TOPO_TURN_CONFIRM and
                            (ctm or safe_geometry))

            if mask_trigger:
                if skel_gate is False:
                    print(f"[NAV] mask_trigger BLOCKED by skel_gate f={frame_idx}")
                    self._last_gate_block_frame = frame_idx
                    return SLOW_SPEED, None, 'slow'

                mask_topo_match = False
                if self._forced_opposite:
                    if self._forced_mask_cnt >= 2:
                        mask_topo_match = True
                    else:
                        return SLOW_SPEED, None, 'slow'
                elif potential_dir < 0 and topo['left']: mask_topo_match = True
                elif potential_dir > 0 and topo['right']: mask_topo_match = True
                elif potential_dir == 0: mask_topo_match = True

                if not mask_topo_match:
                    return SLOW_SPEED, None, 'slow'

                if potential_dir == 0:
                    print(f"[NAV] intersection dir=0 → normal, block sign")
                    if committed is not None:
                        self.voter.block(committed)
                    self.state = 'normal'
                    self._cnt = 0
                    self._turn_dir = 0
                    self._pending_turn_dir = 0
                    self._final_commit = None
                    self._topo_match_cnt = 0
                    self._forced_mask_cnt = 0
                    return 0, 0, False

                print(f"[NAV] → STOPPING (dir={potential_dir}, gate={skel_gate})")
                self.state = 'stopping'; self._cnt = 0
                self._final_commit = committed
                self._pending_turn_dir = potential_dir
                self._topo_match_cnt = 0; self._forced_mask_cnt = 0
                return 0, 0, True

            if topo_trigger and potential_dir != 0:
                print(f"[NAV] → STOPPING (topo fallback, dir={potential_dir})")
                self.state = 'stopping'; self._cnt = 0
                self._final_commit = committed
                self._pending_turn_dir = potential_dir
                self._topo_match_cnt = 0
                return 0, 0, True

            return SLOW_SPEED, None, 'slow'

        # --- STOPPING ---
        if self.state == 'stopping':
            self._cnt += 1
            if committed is not None: self._final_commit = committed
            self._turn_dir = self._pending_turn_dir
            if self._turn_dir == 0: return 0, 0, True
            if self._cnt < STOP_HOLD_FRAMES: return 0, 0, True
            print(f"[NAV] stop done dir={self._turn_dir}")
            self.state = 'turning'; self._cnt = 0; self._stable_cnt = 0
            self._initial_sign = 0; self._crossed_zero = False
            self._bias_decay = False; self._new_lane_cnt = 0
            self._new_lane_offset = -1.0; self._topo_match_cnt = 0
            self._turn_mask_lost_cnt = 0; self._turn_mask_recover_cnt = 0
            self._turn_mask_wait = False; self._pending_turn_dir = 0
            self._forced_mask_cnt = 0
            return 0, 0, True

        # --- TURNING ---
        if self.state == 'turning':
            mh = (W > 0 and bottom_cx >= 0 and cur_w is not None and cur_w > 0)
            if self._turn_mask_wait:
                if mh: self._turn_mask_recover_cnt += 1
                else: self._turn_mask_recover_cnt = 0
                if self._turn_mask_recover_cnt >= TURN_MASK_RECOVER_FRAMES:
                    self._turn_mask_wait = False; self._turn_mask_lost_cnt = 0
                    self._turn_mask_recover_cnt = 0
                else:
                    self._last_ff_used = 0.0; self._last_angle_raw = 0.0
                    return 0, 0, True
            if not mh: self._turn_mask_lost_cnt += 1
            else: self._turn_mask_lost_cnt = 0
            if self._turn_mask_lost_cnt >= TURN_MASK_LOST_TRIGGER_FRAMES:
                self._turn_mask_wait = True; self._turn_mask_recover_cnt = 0
                self._last_ff_used = 0.0; self._last_angle_raw = 0.0
                return 0, 0, True

            self._cnt += 1
            ff = TURN_FF_BEFORE
            ar = ff * self._turn_dir + TURN_PROP_K * blended_error
            if TURN_CLAMP_SAME_DIR:
                if self._turn_dir > 0:
                    angle = float(np.clip(ar, 0.0, TURN_MAX_ANGLE))
                else:
                    angle = float(np.clip(ar, -TURN_MAX_ANGLE, 0.0))
            else:
                angle = float(np.clip(ar, -TURN_MAX_ANGLE, TURN_MAX_ANGLE))
            self._last_ff_used = float(ff * self._turn_dir)
            self._last_angle_raw = float(ar)

            if self._initial_sign == 0 and abs(blended_error) > CROSS_ZERO_DEADBAND:
                self._initial_sign = 1 if blended_error > 0 else -1
            if (self._initial_sign != 0 and not self._crossed_zero
                    and blended_error * self._initial_sign < 0
                    and abs(blended_error) > CROSS_ZERO_DEADBAND):
                self._crossed_zero = True; self._bias_decay = True

            if self._crossed_zero and abs(blended_error) < TURN_STABLE_THRESH:
                self._stable_cnt += 1
            else:
                self._stable_cnt = 0

            ls = (base_w and cur_w and base_w > 0
                  and NEW_LANE_WIDTH_LO * base_w < cur_w < NEW_LANE_WIDTH_HI * base_w)
            if W > 0 and bottom_cx >= 0:
                off = abs(bottom_cx - W / 2.0); self._new_lane_offset = off
                if off < NEW_LANE_CENTER_THRESH and ls and self.heading_ok:
                    self._new_lane_cnt += 1
                else:
                    self._new_lane_cnt = 0
            else:
                self._new_lane_cnt = 0; self._new_lane_offset = -1.0

            passed_min = self._cnt >= NEW_LANE_MIN_FRAMES
            nld = self._new_lane_cnt >= NEW_LANE_STABLE_COUNT
            to = self._cnt >= MAX_TURN_FRAMES
            done = False; reason = ""
            if to: done = True; reason = f"timeout({self._cnt})"
            elif passed_min and nld:
                done = True; reason = f"new_lane(bc={bottom_cx})"
            if done:
                self._last_exit_reason = reason
                self.state = 'cooldown'; self._cnt = 0
                self._stable_cnt = 0; self._new_lane_cnt = 0
                self.voter.reset(); self.just_entered_cooldown = True
            return TURN_SPEED, angle, True

        # --- COOLDOWN ---
        if self.state == 'cooldown':
            self._cnt += 1
            if self._cnt >= COOLDOWN_FRAMES:
                self.state = 'normal'; self._cnt = 0; self._turn_dir = 0
                self._pending_turn_dir = 0; self._final_commit = None
                self._bias_decay = False; self._last_ff_used = 0.0
                self._last_angle_raw = 0.0; self._new_lane_cnt = 0
                self._new_lane_offset = -1.0; self._topo_match_cnt = 0
                self._forced_opposite = False; self._forced_mask_cnt = 0
            return 0, 0, False

        return 0, 0, False

    def _decide_turn(self, sign, topo, is_real_straight=None,
                     branch_lcov=0.0, branch_rcov=0.0,
                     base_w=None, cur_w=None, graph=None, H=192):
        self._forced_opposite = False
        if sign == 'turn_left':  return -1
        if sign == 'turn_right': return +1
        if sign == 'straight':   return 0

        t_junction = self._is_T_junction(graph, H=H, base_w=base_w, cur_w=cur_w)

        bbd = False
        if base_w is not None and base_w > 0 and cur_w is not None:
            wo = cur_w >= BRANCH_MIN_WIDTH_RATIO * base_w
            co = branch_lcov >= BRANCH_COVERAGE_THRESH and branch_rcov >= BRANCH_COVERAGE_THRESH
            to = (bool(topo.get('left') or topo.get('right'))
                  if BRANCH_REQUIRE_TOPO else True)
            if wo and co and to: bbd = True

        if bbd or t_junction:
            has_straight = False
            if t_junction:
                print(f"[NAV] T-junction detected → has_straight=False (override)")
        else:
            ts = bool(topo.get('straight', False))
            ss = (bool(is_real_straight) if is_real_straight is not None else ts)
            has_straight = ts or ss

        if sign == 'no_turn_left':
            if has_straight: return 0
            self._forced_opposite = True
            print(f"[NAV] no_turn_left + NO straight "
                  f"(T_junc={int(t_junction)}) → FORCE turn_right")
            return +1
        if sign == 'no_turn_right':
            if has_straight: return 0
            self._forced_opposite = True
            print(f"[NAV] no_turn_right + NO straight "
                  f"(T_junc={int(t_junction)}) → FORCE turn_left")
            return -1
        return 0

    def bias_scale(self): return BIAS_DECAY_FACTOR if self._bias_decay else 1.0
    def get_ff_used(self): return self._last_ff_used
    def get_angle_raw(self): return self._last_angle_raw
    def get_exit_reason(self): return self._last_exit_reason
    def get_new_lane_cnt(self): return self._new_lane_cnt


# ==============================================================
# PID v20
# ==============================================================
class OcclusionGate:
    def __init__(self, err_step_px=18.0, coverage_drop_frac=0.22, jagged_px=15.0,
                 hard_step_px=30.0, hangover_frames=4, coverage_ema_alpha=0.9,
                 far_outlier_px=40.0, far_outlier_near_px=20.0,
                 far_spread_px=40.0, far_window=5):
        self.err_step_px = err_step_px; self.coverage_drop_frac = coverage_drop_frac
        self.jagged_px = jagged_px; self.hard_step_px = hard_step_px
        self.hangover_frames = hangover_frames; self.cov_ema_alpha = coverage_ema_alpha
        self.far_outlier_px = far_outlier_px; self.far_outlier_near_px = far_outlier_near_px
        self.far_spread_px = far_spread_px; self.far_window = far_window
        self.cov_ema = None; self.prev_error = 0.0; self.initialized = False
        self.hangover_left = 0; self.far_hist = deque(maxlen=far_window)
        self.last_far_outlier = False; self.last_far_spread = 0.0
        self.last_hard_step = False; self.last_jagged = 0.0
        self.last_err_step = 0.0; self.last_cov_drop = 0.0; self.last_trigger_count = 0
    def reset(self):
        self.cov_ema = None; self.prev_error = 0.0; self.initialized = False
        self.hangover_left = 0; self.far_hist.clear()
        self.last_far_outlier = False; self.last_far_spread = 0.0
        self.last_hard_step = False; self.last_jagged = 0.0
        self.last_err_step = 0.0; self.last_cov_drop = 0.0; self.last_trigger_count = 0
    @staticmethod
    def _jag(rc):
        if len(rc) < 6: return 0.0
        cxs = np.array([cx for cx, _ in rc], dtype=np.float32)
        return float(np.mean(np.abs(np.diff(cxs, n=2))))
    def step(self, error, cov, rc, valid, far_e=0.0, near_e=0.0):
        self.far_hist.append(float(far_e))
        if len(self.far_hist) >= 2:
            self.last_far_spread = float(max(self.far_hist) - min(self.far_hist))
        if not self.initialized:
            self.cov_ema = cov; self.prev_error = error; self.initialized = True
            return False
        self.cov_ema = self.cov_ema_alpha * self.cov_ema + (1 - self.cov_ema_alpha) * cov
        if not valid:
            self.prev_error = error; self.hangover_left = self.hangover_frames
            return True
        es = abs(error - self.prev_error)
        cd = max(0.0, (self.cov_ema - cov) / self.cov_ema) if self.cov_ema > 0.05 else 0.0
        jg = self._jag(rc)
        fo = abs(far_e) > self.far_outlier_px and abs(near_e) < self.far_outlier_near_px
        self.last_far_outlier = fo
        fs = len(self.far_hist) >= self.far_window and self.last_far_spread > self.far_spread_px
        sig = 0
        if es > self.err_step_px: sig += 1
        if cd > self.coverage_drop_frac: sig += 1
        if jg > self.jagged_px: sig += 1
        if fo: sig += 1
        if fs: sig += 1
        self.last_err_step = es; self.last_cov_drop = cd; self.last_jagged = jg
        self.last_trigger_count = sig
        self.prev_error = error
        hard = es > self.hard_step_px
        self.last_hard_step = hard
        if hard or sig >= 2:
            self.hangover_left = self.hangover_frames; return True
        if self.hangover_left > 0:
            self.hangover_left -= 1; return True
        return False


class AuthorityManager:
    def __init__(self, output_limit, tau=0.35, sat_frac=0.9, sat_frames=3,
                 lose_enter_px=30.0, lose_exit_px=20.0):
        self.output_limit = output_limit; self.tau = tau; self.sat_frac = sat_frac
        self.sat_frames = sat_frames; self.lose_enter_px = lose_enter_px
        self.lose_exit_px = lose_exit_px; self.sat_streak = 0; self.losing = False
        self.last_d_err = 0.0; self.last_error_pred = 0.0
    def reset(self): self.sat_streak = 0; self.losing = False
    def step(self, error, prev_error, pid_output):
        d = error - prev_error; ep = error + self.tau * d
        self.last_d_err = d; self.last_error_pred = ep
        if abs(pid_output) > self.sat_frac * self.output_limit: self.sat_streak += 1
        else: self.sat_streak = 0
        if not self.losing:
            if self.sat_streak >= self.sat_frames and abs(ep) > abs(error) and abs(error) > self.lose_enter_px:
                self.losing = True
        else:
            if abs(error) < self.lose_exit_px or self.sat_streak < self.sat_frames:
                self.losing = False


class SteeringPID:
    def __init__(self, kp=0.32, ki=0.0, kd=0.6, integral_limit=2.0,
                 derivative_alpha=0.55, output_limit=22.0, rate_limit=10.0,
                 deadband=1.5, emergency_output_limit=25.0,
                 emergency_rate_limit=20.0, emergency_error_px=30.0,
                 emergency_streak_req=2, authority=None):
        self.kp=kp; self.ki=ki; self.kd=kd; self.integral_limit=integral_limit
        self.derivative_alpha=derivative_alpha; self.output_limit=output_limit
        self.rate_limit=rate_limit; self.deadband=deadband
        self.emergency_output_limit=emergency_output_limit
        self.emergency_rate_limit=emergency_rate_limit
        self.emergency_error_px=emergency_error_px
        self.emergency_streak_req=emergency_streak_req
        self.authority = authority or AuthorityManager(output_limit)
        self.reset()
    def reset(self):
        self.integral=0.0; self.prev_error=0.0; self.filtered_derivative=0.0
        self.prev_output=0.0; self.first_call=True; self.authority.reset()
        self.emergency_authority=False; self.emg_streak=0
        self.forced_emergency=False
        self.last_p=self.last_i=self.last_d=self.last_raw=0.0
        self.last_output_limit=self.output_limit
        self.last_rate_limit=self.rate_limit
        self.last_kp_mult=1.0; self.last_kd_mult=1.0
    def soft_reset(self):
        self.integral=0.0; self.prev_error=0.0; self.filtered_derivative=0.0; self.first_call=True
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
            if abs(error) > self.emergency_error_px: self.emg_streak += 1
            else: self.emg_streak = 0
            if self.emg_streak >= self.emergency_streak_req:
                out_lim = self.emergency_output_limit
                rate_lim = self.emergency_rate_limit
                self.emergency_authority = True
            else:
                out_lim = self.output_limit; rate_lim = self.rate_limit
                self.emergency_authority = False
        self.last_output_limit = out_lim; self.last_rate_limit = rate_lim
        if abs(error) < self.deadband:
            e = np.sign(error) * (abs(error) - self.deadband)
        else:
            e = error - np.sign(error) * self.deadband
        p = kp_eff * e
        self.integral += e
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i = self.ki * self.integral
        rd = 0.0 if self.first_call else (error - self.prev_error)
        self.filtered_derivative = (self.derivative_alpha * rd
                                    + (1 - self.derivative_alpha) * self.filtered_derivative)
        d = kd_eff * self.filtered_derivative
        raw = p + i + d
        po = out_lim * np.tanh(raw / out_lim)
        if self.authority.losing: self.integral = 0.0
        self.authority.step(error, self.prev_error, po)
        delta = np.clip(po - self.prev_output, -rate_lim, rate_lim)
        out = self.prev_output + delta
        out = float(np.clip(out, -25.0, 25.0))
        self.prev_error = error; self.prev_output = out; self.first_call = False
        self.last_p = p; self.last_i = i; self.last_d = d; self.last_raw = raw
        return out


def largest_run_bounds(br):
    nz = np.where(br > 0)[0]
    if len(nz) == 0: return None
    if len(nz) == 1: return int(nz[0]), int(nz[0])
    runs = []; s = nz[0]; p = nz[0]
    for x in nz[1:]:
        if x != p + 1: runs.append((s, p)); s = x
        p = x
    runs.append((s, p))
    a, b = max(runs, key=lambda r: r[1] - r[0])
    return int(a), int(b)


def extract_centerline(sm, prev_cs, min_rows=20, coverage_threshold=0.18):
    g = sm if len(sm.shape) == 2 else cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    rc = []
    for y in range(h - 1, -1, -1):
        b = largest_run_bounds(g[y, :])
        if b is not None:
            s, e = b; rc.append((int((s + e) // 2), y))
    cov = np.count_nonzero(g) / float(h * w)
    if len(rc) < min_rows or cov < coverage_threshold:
        return 0.0, 0.0, 0.0, 0.0, [], False, cov
    ya = np.array([y for _, y in rc])
    yb = int(np.percentile(ya, 98)); yt = int(np.percentile(ya, 2))
    sp = max(yb - yt, 1)
    ys_n = yb - int(sp * 0.45); ys_f = yt + int(sp * 0.35)
    np_ = [(cx, y) for cx, y in rc if y >= ys_n]
    fp_ = [(cx, y) for cx, y in rc if y <= ys_f]
    def bc_(p):
        return int(np.median([cx for cx, _ in p])) if p else None
    ncx = bc_(np_); fcx = bc_(fp_)
    ne = (ncx - w // 2) if ncx is not None else 0
    fe = (fcx - w // 2) if fcx is not None else 0
    tb = [cx for cx, y in rc if y <= ys_f]; bb = [cx for cx, y in rc if y >= ys_n]
    tcx = int(np.median(tb)) if tb else 0; bcx = int(np.median(bb)) if bb else 0
    ce = tcx - bcx
    cs = 0.90 * prev_cs + 0.10 * ce
    return ne, fe, ce, cs, rc, True, cov


def detect_asymmetry(gray, mb=45.0, y0f=0.18, y1f=0.60):
    h, w = gray.shape
    bcs = []
    for y in range(int(h * 0.80), h, 4):
        nz = np.where(gray[y, :] > 0)[0]
        if len(nz) >= 2: bcs.append((nz[0] + nz[-1]) // 2)
    if not bcs: return 0.0
    ref = int(np.median(bcs))
    lws, rws = [], []
    for y in range(int(h * y0f), int(h * y1f), 4):
        nz = np.where(gray[y, :] > 0)[0]
        if len(nz) < 2: continue
        lw = ref - nz[0]; rw = nz[-1] - ref
        if lw > 0 and rw > 0: lws.append(lw); rws.append(rw)
    if not lws: return 0.0
    lw = float(np.median(lws)); rw = float(np.median(rws))
    if lw + rw < 40: return 0.0
    return (rw - lw) / (lw + rw) * mb


def plan_speed(cs, mx=26.0, mn=0.0, scale=80.0):
    cr = min(abs(cs) / scale, 1.0)
    return mx - (mx - mn) * cr, cr


def error_speed_factor(error, sp=8.0, fp=35.0, mf=0.30):
    a = abs(error)
    if a <= sp: return 1.0
    if a >= fp: return mf
    t = (a - sp) / (fp - sp)
    return 1.0 - t * (1.0 - mf)


class TurnErrorEstimator:
    def __init__(self):
        self.bh = deque(maxlen=3); self.far_off_raw = None
    def reset(self): self.bh.clear(); self.far_off_raw = None
    def step(self, gray, bias_pixel, speed, max_speed, wide_change):
        h, w = gray.shape
        binary = gray > 0
        rc = binary.sum(axis=1)
        rows = np.flatnonzero(rc > 0)
        if rows.size == 0:
            self.reset(); return 0.0, -1
        xs = np.arange(w, dtype=np.float32)
        cxr = (binary[rows] * xs).sum(axis=1) / rc[rows]
        self.bh.append(float(cxr[-1])); bcx = int(round(float(np.median(self.bh))))
        tgt = w / 2.0 + bias_pixel
        ns = rows >= h // 2; fs = rows < h // 3
        hf = (not wide_change) and bool(fs.any())
        ne = float(cxr[ns].mean() - tgt) if ns.any() else 0.0
        fe = float(cxr[fs].mean() - tgt) if hf else 0.0
        self.far_off_raw = float(cxr[fs].mean() - w / 2.0) if hf else None
        if abs(ne) + abs(fe) < 10: wn, wf = 0.7, 0.3
        else:
            wf = float(np.clip(0.5 + 0.25 * (speed / max_speed), 0.35, 0.75))
            wn = float(np.clip(1.0 - wf - 0.05, 0.25, 0.65))
            if ne * fe < 0: wf *= 0.7; wn = float(np.clip(1.0 - wf - 0.05, 0.25, 0.65))
        bl = wn * ne + wf * fe
        if abs(bl) < STRAIGHT_ERROR_DEADBAND: bl = 0.0
        return float(bl), bcx


class LaneFollower:
    def __init__(self):
        self.authority = AuthorityManager(output_limit=22.0)
        self.pid = SteeringPID(kp=0.32, ki=0.0, kd=0.6, integral_limit=2.0,
                               derivative_alpha=0.55, output_limit=22.0,
                               rate_limit=10.0, deadband=1.5,
                               emergency_output_limit=25.0,
                               emergency_rate_limit=20.0,
                               emergency_error_px=30.0,
                               emergency_streak_req=EMERGENCY_STREAK_REQ,
                               authority=self.authority)
        self.gate = OcclusionGate(err_step_px=ERR_STEP_PX, coverage_drop_frac=COV_DROP,
                                  jagged_px=JAGGED_PX, hard_step_px=HARD_STEP_PX,
                                  hangover_frames=OCCL_HANGOVER,
                                  far_outlier_px=FAR_OUTLIER_PX,
                                  far_outlier_near_px=FAR_OUTLIER_NEAR_PX,
                                  far_spread_px=FAR_SPREAD_PX, far_window=FAR_WINDOW)
        self.speed = MIN_SPEED
        self.reset_tracking()
    def reset_tracking(self):
        self.pid.reset(); self.gate.reset()
        self.last_angle = 0.0; self.curve_smooth = 0.0
        self.prev_valid = False; self.prev_far_error = 0.0
        self.d_far_filt = 0.0; self.asym_smooth = 0.0; self.occl_emer_hold = 0
    def step(self, lane_mask, now, speed_cap=None, force_straight=False, near_only=False):
        ne, fe, ce, self.curve_smooth, rpts, valid, cov = extract_centerline(lane_mask, self.curve_smooth)
        if near_only: fe = ne
        fd = float(np.clip(fe - ne, -FAR_DELTA_CLIP, FAR_DELTA_CLIP))
        dfr = float(fe - self.prev_far_error) if (valid and self.prev_valid) else 0.0
        df = float(np.clip(dfr, -D_FAR_CLIP, D_FAR_CLIP))
        self.d_far_filt = D_FAR_EMA_ALPHA * df + (1 - D_FAR_EMA_ALPHA) * self.d_far_filt
        lead = 0.5 * fd + LEAD_RATE_GAIN * self.d_far_filt
        bl = float(ne) + lead
        self.prev_far_error = float(fe)
        occl = self.gate.step(bl, cov, rpts, valid, far_e=fe, near_e=ne)
        if occl: self.occl_emer_hold = EMG_HOLD_AFTER_OCCL
        else:
            if self.occl_emer_hold > 0: self.occl_emer_hold -= 1
        if not occl and valid:
            ar = detect_asymmetry(lane_mask)
            self.asym_smooth = ASYM_EMA * self.asym_smooth + (1 - ASYM_EMA) * ar
            if abs(ne) < 5.0 and abs(self.asym_smooth) < ASYM_CAP:
                self.asym_smooth *= ASYM_DECAY
            self.asym_smooth = float(np.clip(self.asym_smooth, -ASYM_CAP, ASYM_CAP))
        aa = ASYM_STRENGTH * self.asym_smooth if abs(self.asym_smooth) >= ASYM_THRESHOLD else 0.0
        eff = bl + aa
        self.pid.forced_emergency = self.occl_emer_hold > 0
        if valid:
            if not self.prev_valid: self.pid.soft_reset(); self.gate.reset()
            if force_straight: angle = 0.0; self.pid.reset()
            else: angle = self.pid.step(eff, now)
            ts, cr = plan_speed(self.curve_smooth, MAX_SPEED, MIN_SPEED)
            esm = error_speed_factor(eff)
            ath = 0.6 if self.authority.losing else 1.0
            anm = 0.5 if abs(self.asym_smooth) > 15 else 1.0
            sm = min(ath, anm)
            ts = max(ts * esm * sm, 0.0)
            speed = 0.7 * self.speed + 0.3 * ts
            self.last_angle = angle
        else:
            angle = 0.0 if force_straight else self.last_angle
            cr = 0.0; speed = 0.85 * self.speed
        if speed_cap is not None: speed = min(speed, float(speed_cap))
        self.speed = float(speed); self.prev_valid = valid
        info = dict(near=ne, far=fe, curve_smooth=self.curve_smooth, lead=lead,
                    blended=bl, eff_error=eff, asym=self.asym_smooth,
                    occl=bool(occl), forced_emer=bool(self.pid.forced_emergency),
                    occl_emer_hold=self.occl_emer_hold,
                    emer_auth=bool(self.pid.emergency_authority),
                    losing=bool(self.authority.losing),
                    valid=bool(valid), coverage=cov, curve_ratio=cr, row_pts=rpts,
                    far_outlier=self.gate.last_far_outlier,
                    far_spread=self.gate.last_far_spread)
        return float(angle), self.speed, info


# ==============================================================
# SIGN DETECT
# ==============================================================
def sign_detect(raw_image, conf=SIGN_CONF, imgsz=SIGN_IMGSZ):
    results = sign_model.predict(source=raw_image, conf=conf, imgsz=imgsz, verbose=False)
    classes = []
    if results and results[0].boxes is not None and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            classes.append(sign_model.names[int(box.cls[0])])
    return classes


# ==============================================================
# DEBUG VIEW
# ==============================================================
def draw_debug(gray, info, pid, angle, speed, nav_text):
    h, w = gray.shape
    debug = np.zeros((h, w, 3), dtype=np.uint8)
    debug[gray > 0] = (90, 90, 90)
    cv2.line(debug, (w // 2, 0), (w // 2, h), (0, 0, 255), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(debug, nav_text, (10, 18), font, 0.45, (0, 255, 255), 1)
    if info:
        for cx, y in info["row_pts"]:
            debug[y, cx] = (0, 255, 0)
        cv2.putText(debug, f"blend={info['blended']:+.1f} near={info['near']:+.0f} far={info['far']:+.0f}",
                    (10, 36), font, 0.45, (255, 255, 255), 1)
        cv2.putText(debug, f"spd={speed:.1f} a={angle:+.1f}",
                    (10, 54), font, 0.45, (200, 200, 255), 1)
    else:
        cv2.putText(debug, f"NAV spd={speed:.1f} a={angle:+.1f}",
                    (10, 36), font, 0.45, (255, 255, 255), 1)
    cv2.imshow("Lane Debug", debug)


def draw_gate_debug(seg_mask, graph, skel_gate, W, H, topo=None):
    vis = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2BGR)
    if not SKEL_OK or not graph.get('ok'):
        cv2.putText(vis, "SKEL FAIL", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return vis
    binary = clean_mask_for_skeleton(seg_mask)
    if binary.sum() > 0:
        skel = skeletonize(binary > 0).astype(np.uint8)
        vis[skel > 0] = (0, 255, 255)
    arm_colors = {'L': (255, 128, 0), 'S': (0, 255, 0), 'R': (255, 0, 128)}
    for jidx, arms in graph.get('arms', {}).items():
        for key, polys in arms.items():
            color = arm_colors.get(key, (128, 128, 128))
            for poly in polys:
                for (x, y) in poly:
                    if 0 <= x < W and 0 <= y < H:
                        vis[y, x] = color
    for jx, jy, jidx in graph.get('junctions', []):
        cv2.circle(vis, (jx, jy), 5, (0, 0, 255), -1)
    if skel_gate is True:
        txt, color = "GATE=ALLOW", (0, 255, 0)
    elif skel_gate is False:
        txt, color = "GATE=DENY", (0, 0, 255)
    else:
        txt, color = "GATE=NONE", (150, 150, 150)
    cv2.putText(vis, txt, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.putText(vis, f"n_junc={graph.get('n_junctions', 0)} n_arms={graph.get('n_arms_total', 0)}",
                (5, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    # ---- TOPOLOGY overlay ----
    if topo is not None:
        # Vẽ 3 vùng L / S / R
        x_L = int(0.40 * W)
        x_R = int(0.60 * W)
        y0  = int(Y_LO_RATIO * H)
        y1  = int(Y_HI_RATIO * H)

        cv2.line(vis, (x_L, y0), (x_L, y1), (90, 90, 90), 1)
        cv2.line(vis, (x_R, y0), (x_R, y1), (90, 90, 90), 1)
        cv2.line(vis, (0, y0), (W, y0), (90, 90, 90), 1)
        cv2.line(vis, (0, y1), (W, y1), (90, 90, 90), 1)

        # Dấu ● cho từng vùng (xanh = true)
        def _mark(ok, x_center):
            color = (0, 255, 0) if ok else (70, 70, 70)
            cv2.circle(vis, (x_center, y0 - 6), 5, color, -1)

        _mark(topo.get('left', False),     int(0.20 * W))
        _mark(topo.get('straight', False), int(0.50 * W))
        _mark(topo.get('right', False),    int(0.80 * W))

        # Text box dưới cùng
        L = "L" if topo.get('left')     else "-"
        S = "S" if topo.get('straight') else "-"
        R = "R" if topo.get('right')    else "-"
        sr  = topo.get('straight_ratio', 0.0)
        the = topo.get('top_half_empty', False)
        thc = topo.get('top_half_coverage', 0.0)

        lines = [
            f"TOPO: [{L} {S} {R}]",
            f"  sr={sr:.2f} top_empty={int(the)} cov={thc:.3f}",
        ]
        y_text = H - 26
        for i, txt in enumerate(lines):
            cv2.putText(vis, txt, (5, y_text + i * 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

    return vis


# ==============================================================
# MAIN LOOP
# ==============================================================
if __name__ == "__main__":
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log_file = open(LOG_PATH, "w", newline="", buffering=1)
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "t", "frame", "nav_state", "commit", "turn_dir",
        "base_w", "cur_w", "ratio", "turn_err", "bottom_cx",
        "near", "far", "blended", "eff_error",
        "occl", "emer_auth", "losing",
        "speed", "angle",
        "skel_ok", "skel_n_junc", "skel_n_arms",
        "skel_gate", "blocked_sign",
        "v3_skel_jy", "branch_lcov", "branch_rcov",
    ])
    print(f"[log] writing to {LOG_PATH}")

    lane_width_estimator = LaneWidthEstimator()
    voter = SignVoter(min_frames=MIN_FRAMES, max_gap=MAX_GAP)
    navigator = SlowStopTurnNavigator(voter=voter)
    intersection_detector = IntersectionDetector()
    follower = LaneFollower()
    turn_est = TurnErrorEstimator()

    was_turning = False
    frame_idx = 0

    try:
        while True:
            frame_idx += 1
            t_now = time.time()
            GetStatus()
            raw_image = GetRaw()

            seg_mask, lane_mask = get_road_masks(raw_image)
            H, W = seg_mask.shape

            graph = build_skeleton_graph(seg_mask, W, H)
            skel_gate = compute_skeleton_gate(graph, H)

            topo = get_topology(seg_mask, W, H)
            is_real_straight, skel_info = skeleton_verify_straight(seg_mask, W, H)
            branch_lcov, branch_rcov = detect_branch_bottom(seg_mask, W, H)

            sign_classes = sign_detect(raw_image)
            if navigator.state not in ('turning', 'cooldown'):
                voter.update(sign_classes)
            committed = voter.get_committed()

            lane_width_estimator.measure_width(seg_mask)
            base_w = lane_width_estimator.base_width
            cur_w = lane_width_estimator.current_width
            mask_trigger = intersection_detector.update(base_w, cur_w)

            bias_pixel_pre = 0
            turn_err = 0.0
            bottom_cx_pre = -1
            if navigator.state == 'turning':
                if not was_turning: turn_est.reset()
                was_turning = True
                bias_pixel_pre = decide_bias(committed, topo, base_w,
                                             scale=navigator.bias_scale())
                turn_err, bottom_cx_pre = turn_est.step(
                    seg_mask, bias_pixel_pre, TURN_SPEED, MAX_SPEED,
                    lane_width_estimator.is_wide_change())
                navigator.heading_ok = (turn_est.far_off_raw is None
                                        or abs(turn_est.far_off_raw) < TURN_EXIT_FAR_THRESH)
            else:
                was_turning = False

            effective_trigger = (mask_trigger and navigator.state == 'slow_down')
            nav_speed, nav_angle, is_active = navigator.update(
                sign_classes, committed, topo, effective_trigger,
                turn_err, bottom_cx_pre, W, base_w, cur_w,
                is_real_straight=is_real_straight,
                branch_lcov=branch_lcov, branch_rcov=branch_rcov,
                skel_gate=skel_gate, frame_idx=frame_idx,
                graph=graph, H=H,
            )

            if navigator.just_entered_cooldown:
                lane_width_estimator.reset_baseline()
                follower.reset_tracking()
            if navigator.state in ('stopping', 'turning', 'cooldown'):
                intersection_detector.reset()

            info = {}
            angle = 0.0
            speed = follower.speed if follower.speed > 0 else 15.0

            if is_active is True:
                speed = float(nav_speed)
                angle = float(nav_angle) if nav_angle is not None else 0.0
                follower.speed = speed
                AVControl(speed, angle)
                print(f"[NAV] {navigator.state} dir={navigator._turn_dir} "
                      f"gate={skel_gate} ang={angle:+.1f} spd={speed}")
            else:
                speed_cap = SLOW_SPEED if is_active == 'slow' else None
                angle, speed, info = follower.step(lane_mask, t_now, speed_cap=speed_cap)
                AVControl(speed, angle)

            pid = follower.pid
            has = bool(info)
            ratio_log = (cur_w / base_w) if (base_w and base_w > 0) else 0
            gate_log = -1 if skel_gate is None else int(skel_gate)
            blocked_log = voter._blocked_until_gone if voter._blocked_until_gone else ""

            log_writer.writerow([
                f"{t_now:.3f}", frame_idx, navigator.state,
                committed if committed else "",
                navigator._turn_dir,
                base_w if base_w else 0, cur_w if cur_w else 0,
                f"{ratio_log:.3f}",
                f"{turn_err:+.3f}", bottom_cx_pre,
                f"{info['near']:.1f}" if has else "",
                f"{info['far']:.1f}" if has else "",
                f"{info['blended']:.3f}" if has else "",
                f"{info['eff_error']:.3f}" if has else "",
                int(info['occl']) if has else "",
                int(info['emer_auth']) if has else "",
                int(info['losing']) if has else "",
                f"{speed:.3f}", f"{angle:.3f}",
                int(graph.get('ok', False)),
                graph.get('n_junctions', 0),
                graph.get('n_arms_total', 0),
                gate_log, blocked_log,
                skel_info['jy'],
                f"{branch_lcov:.3f}", f"{branch_rcov:.3f}",
            ])
            if frame_idx % LOG_FLUSH_EVERY == 0:
                log_file.flush()

            if ENABLE_GUI:
                nav_text = (f"NAV={navigator.state} commit={committed} "
                            f"gate={skel_gate} blk={blocked_log}")
                draw_debug(lane_mask, info, pid, angle, speed, nav_text)
                gate_vis = draw_gate_debug(seg_mask, graph, skel_gate, W, H, topo=topo)
                cv2.imshow('Raw', raw_image)
                cv2.imshow('Gate Debug', gate_vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        print('Closing socket and windows...')
        log_file.close()
        AVControl(0, 0)
        CloseSocket()
        cv2.destroyAllWindows()

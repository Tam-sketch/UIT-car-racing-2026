# ==============================================================
# UIT CAR RACING 2026 - COMBINED
#
#   LANE FOLLOWING  : PID + OcclusionGate + AuthorityManager
#                     (lấy từ ucr_main_v20.py)
#   INTERSECTION    : SignVoter + Topology + SlowStopTurnNavigator
#                     (lấy từ maycayv9.py V4.6.1)
#
# Luồng điều khiển mỗi frame:
#   normal / slow_down / cooldown -> LaneFollower (PID v20)
#                                    * slow_down : speed cap = SLOW_SPEED
#                                    * cooldown  : speed cap = COOLDOWN_SPEED_CAP,
#                                                  vài frame đầu ép góc = 0
#   stopping / turning            -> Navigator điều khiển (FF + P như v9)
#
# Đã bỏ khỏi maycayv9: calculate_steering_angle (heuristic),
#   calculate_hybrid_normal_angle, SteeringPID cũ  -> thay bằng PID v20.
# ==============================================================

from ucr_lib import GetStatus, GetRaw, AVControl, CloseSocket
import cv2
import csv
import os
import time
import numpy as np
from collections import deque
from ultralytics import YOLO


# ==============================================================
# CONFIG
# ==============================================================
# --- Models ---
# Chỉ dùng 1 road model cho cả PID lẫn topology (đỡ tốn thời gian infer).
# maycayv9 dùng best12.pt, ucr_main_v20 dùng best.pt -> đổi ở đây nếu muốn.
ROAD_MODEL_PATH = "/workspace/best12.pt"
SIGN_MODEL_PATH = "/workspace/sign.pt"

ROAD_IMGSZ = (192, 320)
SIGN_IMGSZ = 640
SIGN_CONF  = 0.7

# --- Sign voter ---
MIN_FRAMES = 4
MAX_GAP    = 3
BIAS_RATIO = 0.3
SIGN_TIMEOUT_FRAMES = 300

# --- Topology ---
Y_LO_RATIO = 0.15
Y_HI_RATIO = 0.70
MIN_BRANCH_WIDTH = 12
TOP_HALF_RATIO = 0.55           # vùng trên ảnh (y = 0 -> 0.55*H)
TOP_HALF_EMPTY_THRESH = 0.02    # < 2% pixel có mask -> coi là trống

# --- Speed (navigator) ---
SLOW_SPEED = 10
TURN_SPEED = 12

# --- Turn angle ---
TURN_FF_BEFORE      = 20
TURN_PROP_K         = 0.15
TURN_MAX_ANGLE      = 25
TURN_CLAMP_SAME_DIR = True
STRAIGHT_ERROR_DEADBAND = 2.0   # deadband cho turn error

# --- Stop/Turn ---
STOP_HOLD_FRAMES = 0

# --- Cross-zero + stable ---
CROSS_ZERO_DEADBAND = 5.0
TURN_STABLE_THRESH  = 12.0

# --- New lane detection ---
NEW_LANE_CENTER_THRESH = 40.0
NEW_LANE_STABLE_COUNT  = 3
NEW_LANE_MIN_FRAMES    = 15
NEW_LANE_WIDTH_LO      = 0.5
NEW_LANE_WIDTH_HI      = 1.5
MAX_TURN_FRAMES        = 90

# --- Turn mask-loss recovery ---
TURN_MASK_LOST_TRIGGER_FRAMES = 20
TURN_MASK_RECOVER_FRAMES = 2

# --- Bias decay ---
BIAS_DECAY_FACTOR = 0.3

# --- Cooldown ---
COOLDOWN_FRAMES                = 30
COOLDOWN_SPEED_CAP             = 18
COOLDOWN_FORCE_STRAIGHT_FRAMES = 5
POST_TURN_NEAR_ONLY_FRAMES     = 15   # sau rẽ: PID chỉ nhìn near, bỏ far (còn thấy nhánh ngã tư)
TURN_EXIT_FAR_THRESH           = 60.0 # thoát turn: far centroid phải gần tâm ảnh (xe đã thẳng hàng)

# --- Mask intersection ---
INTERSECTION_RATIO   = 1.35
INTERSECTION_CONFIRM = 2
INTERSECTION_MIN_ABS = 120

# --- Topology fallback for turn trigger ---
TOPO_TURN_CONFIRM = 2

# --- Early intersection detection ---
EARLY_TOPO_Y_LO_RATIO = 0.08
EARLY_TOPO_Y_HI_RATIO = 0.72
EARLY_TOPO_CONFIRM = 3
EARLY_TOPO_TTL_FRAMES = 45
EARLY_TOPO_MIN_WIDTH_RATIO = 0.28
EARLY_TOPO_MAX_WIDTH_RATIO = 0.85
EARLY_TRIGGER_MIN_CUR_W = 120
EARLY_TRIGGER_WIDTH_RATIO = 1.30
EARLY_TRIGGER_CONFIRM = 2

# --- Lane following (PID, từ ucr_main_v20) ---
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

# --- Log / GUI ---
LOG_PATH = "/workspace/my_code/pid_log_combined.csv"
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
    """
    Trả về (full_mask, lane_mask):
      full_mask : union mọi mask      -> topology / lane width / turning (maycayv9)
      lane_mask : thành phần lớn nhất -> centerline + PID (ucr_main_v20)
    """
    if raw_image.dtype != np.uint8:
        raw_image = raw_image.astype(np.uint8)
    results = road_model.predict(
        source=raw_image, imgsz=ROAD_IMGSZ, verbose=False
    )
    if not results or results[0].masks is None:
        empty = np.zeros(ROAD_IMGSZ, dtype=np.uint8)
        return empty, empty
    masks = results[0].masks.data.cpu().numpy()
    full = (np.sum(masks, axis=0) > 0).astype(np.uint8) * 255
    return full, keep_largest_component(full)




# ==============================================================
# INTERSECTION LOGIC (từ maycayv9 V4.6.1)
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

    def update(self, classes):
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
            if self.committed != cls:
                print(f"[VOTER] committed={cls} (count={self.count})")
                self.committed = cls
                self._age = 0
                self._commit_age = 0

    def get_committed(self):
        return self.committed

    def get_commit_age(self):
        return self._commit_age

    def reset(self):
        self.current_class = None
        self.count = 0
        self.gap = 0
        self.committed = None
        self._age = 0
        self._commit_age = 0
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
        self.last_cur_w = 0.0

    def update(self, base_width, current_width):
        self.last_cur_w = current_width if current_width else 0.0

        if base_width is None or base_width <= 0 or current_width is None:
            self._history.append(False)
            return False

        if current_width < self.min_abs_width:
            self._history.append(False)
            self.last_ratio = current_width / base_width
            return False

        ratio = current_width / base_width
        self.last_ratio = ratio
        frame_hit = ratio >= self.ratio_threshold
        self._history.append(frame_hit)

        if len(self._history) == self.confirm_frames and all(self._history):
            return True
        return False

    def reset(self):
        self._history.clear()
        self.last_ratio = 0.0
        self.last_cur_w = 0.0


class SlowStopTurnNavigator:
    """
    V4.6.1:
      - slow_down → stopping chỉ khi mask_trigger + topo match hướng
      - no_turn_left / no_turn_right:
          * Nửa trên ảnh (y=0..H/2) CÓ mask  → đi thẳng.
          * Nửa trên ảnh TRỐNG mask         → FORCE rẽ ngược hướng cấm.
      - Exit turning khi bottom_cx quay về gần W/2 (new lane detected)
    """
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
        self.heading_ok = True   # main set mỗi frame khi turning

        self._turn_mask_lost_cnt = 0
        self._turn_mask_recover_cnt = 0
        self._turn_mask_wait = False

        self._topo_match_cnt = 0

        # V4.6.1: cờ báo sign ép rẽ ngược hướng cấm

        self._early_topo = {'left': False, 'straight': True,
                            'right': False, 'straight_ratio': 1.0,
                            'top_half_empty': False,
                            'top_half_coverage': 1.0}
        self._early_topo_cnt = 0
        self._early_topo_age = 0
        self._early_trigger_cnt = 0

    def update_early_topology(self, early_topo):
        has_branch = bool(early_topo.get('left') or early_topo.get('right'))
        if has_branch:
            if (early_topo.get('left') == self._early_topo.get('left') and
                    early_topo.get('right') == self._early_topo.get('right')):
                self._early_topo_cnt += 1
            else:
                self._early_topo_cnt = 1
            self._early_topo = dict(early_topo)
            self._early_topo_age = 0
        else:
            self._early_topo_age += 1
            if self._early_topo_age > EARLY_TOPO_TTL_FRAMES:
                self._early_topo = {'left': False, 'straight': True,
                                    'right': False, 'straight_ratio': 1.0,
                                    'top_half_empty': False,
                                    'top_half_coverage': 1.0}
                self._early_topo_cnt = 0

    def update(self, sign_classes, committed, topo, mask_trigger,
               blended_error,
               bottom_cx, W, base_w, cur_w):
        self.just_entered_cooldown = False

        early_ready = self._early_topo_cnt >= EARLY_TOPO_CONFIRM
        if early_ready:
            topo_for_decision = {
                'left':     bool(topo.get('left')     or self._early_topo.get('left')),
                'straight': bool(topo.get('straight') or self._early_topo.get('straight')),
                'right':    bool(topo.get('right')    or self._early_topo.get('right')),
                # Các field phụ chỉ tin frame hiện tại
                'straight_ratio':    float(topo.get('straight_ratio', 1.0)),
                'top_half_empty':    bool(topo.get('top_half_empty', False)),
                'top_half_coverage': float(topo.get('top_half_coverage', 1.0)),
            }
        else:
            topo_for_decision = topo

        # --- NORMAL ---
        if self.state == 'normal':
            if committed is not None:
                print(f"[NAV] → SLOW_DOWN (sign seen: {committed})")
                self.state = 'slow_down'
                self._cnt = 0
            return 0, 0, False

        # --- SLOW_DOWN ---
        if self.state == 'slow_down':
            potential_dir = self._decide_turn(committed, topo_for_decision)

            topo_match = False

            if potential_dir < 0 and (topo['left'] or (early_ready and self._early_topo['left'])):
                topo_match = True
            elif potential_dir > 0 and (topo['right'] or (early_ready and self._early_topo['right'])):
                topo_match = True

            if topo_match:
                self._topo_match_cnt += 1
            else:
                self._topo_match_cnt = 0

            safe_geometry = (
                cur_w is not None and base_w is not None and
                cur_w >= EARLY_TRIGGER_MIN_CUR_W and
                cur_w >= base_w * EARLY_TRIGGER_WIDTH_RATIO
            )
            current_topo_match = (
                (potential_dir < 0 and topo.get('left')) or
                (potential_dir > 0 and topo.get('right'))
            )
            topo_trigger = (
                self._topo_match_cnt >= TOPO_TURN_CONFIRM and
                (current_topo_match or safe_geometry)
            )

            if mask_trigger:
                mask_topo_match = False

                if potential_dir < 0 and topo['left']:
                    mask_topo_match = True
                elif potential_dir > 0 and topo['right']:
                    mask_topo_match = True
                elif potential_dir == 0:
                    mask_topo_match = True

                if not mask_topo_match:
                    print(
                        f"[NAV] slow_down — mask_trigger nhưng topo mismatch "
                        f"(dir={potential_dir}, "
                        f"topo=L{int(topo['left'])}"
                        f"S{int(topo['straight'])}"
                        f"R{int(topo['right'])}) "
                        f"— chờ nhánh đúng hướng"
                    )
                    return SLOW_SPEED, None, 'slow'

                if potential_dir == 0:
                    print(
                        f"[NAV] intersection — commit={committed}, "
                        f"dir=0 -> STRAIGHT, skip STOPPING"
                    )

                    self.state = 'normal'
                    self._cnt = 0
                    self._turn_dir = 0
                    self._pending_turn_dir = 0
                    self._final_commit = None
                    self._topo_match_cnt = 0
            
                    self.voter.reset()

                    return 0, 0, False

                print(
                    f"[NAV] → STOPPING "
                    f"(mask+topo match, dir={potential_dir})"
                )

                self.state = 'stopping'
                self._cnt = 0
                self._final_commit = committed
                self._pending_turn_dir = potential_dir

                self._topo_match_cnt = 0
                return 0, 0, True

            if topo_trigger:
                print(
                    f"[NAV] → STOPPING "
                    f"(topo fallback {self._topo_match_cnt}/"
                    f"{TOPO_TURN_CONFIRM}, dir={potential_dir}, "
                    f"topo=L{int(topo['left'])}"
                    f"S{int(topo['straight'])}"
                    f"R{int(topo['right'])})"
                )

                self.state = 'stopping'
                self._cnt = 0
                self._final_commit = committed
                self._pending_turn_dir = potential_dir

                self._topo_match_cnt = 0
                return 0, 0, True

            return SLOW_SPEED, None, 'slow'

        # --- STOPPING ---
        if self.state == 'stopping':
            self._cnt += 1

            if committed is not None:
                self._final_commit = committed

            self._turn_dir = self._pending_turn_dir

            if self._turn_dir == 0:
                print(
                    f"[NAV] STOPPING — waiting for latched turn_dir "
                    f"(commit={self._final_commit}, "
                    f"pending={self._pending_turn_dir})"
                )
                return 0, 0, True

            if self._cnt < STOP_HOLD_FRAMES:
                return 0, 0, True

            print(
                f"[NAV] stop done — commit={self._final_commit}, "
                f"turn_dir={self._turn_dir}"
            )

            self.state = 'turning'
            self._cnt = 0
            self._stable_cnt = 0
            self._initial_sign = 0
            self._crossed_zero = False
            self._bias_decay = False
            self._last_ff_used = 0.0
            self._last_angle_raw = 0.0
            self._last_exit_reason = ""
            self._new_lane_cnt = 0
            self._new_lane_offset = -1.0
            self._topo_match_cnt = 0
            self._turn_mask_lost_cnt = 0
            self._turn_mask_recover_cnt = 0
            self._turn_mask_wait = False

            self._pending_turn_dir = 0

            return 0, 0, True

        # --- TURNING ---
        if self.state == 'turning':
            mask_healthy = (
                W > 0
                and bottom_cx >= 0
                and cur_w is not None
                and cur_w > 0
            )

            if self._turn_mask_wait:
                if mask_healthy:
                    self._turn_mask_recover_cnt += 1
                else:
                    self._turn_mask_recover_cnt = 0

                if self._turn_mask_recover_cnt >= TURN_MASK_RECOVER_FRAMES:
                    self._turn_mask_wait = False
                    self._turn_mask_lost_cnt = 0
                    self._turn_mask_recover_cnt = 0
                    print(
                        f"[NAV] TURN RESUME — mask recovered "
                        f"({TURN_MASK_RECOVER_FRAMES} frames), "
                        f"turn_dir={self._turn_dir}"
                    )
                else:
                    self._last_ff_used = 0.0
                    self._last_angle_raw = 0.0
                    print(
                        f"[NAV] TURN WAIT — mask lost, "
                        f"recover={self._turn_mask_recover_cnt}/"
                        f"{TURN_MASK_RECOVER_FRAMES} "
                        f"bc={bottom_cx} cur_w={cur_w}"
                    )
                    return 0, 0, True

            if not mask_healthy:
                self._turn_mask_lost_cnt += 1
            else:
                self._turn_mask_lost_cnt = 0

            if self._turn_mask_lost_cnt >= TURN_MASK_LOST_TRIGGER_FRAMES:
                self._turn_mask_wait = True
                self._turn_mask_recover_cnt = 0
                self._last_ff_used = 0.0
                self._last_angle_raw = 0.0
                print(
                    f"[NAV] TURN INTERRUPT — mask lost "
                    f"{self._turn_mask_lost_cnt} frames; STOP and wait for new road"
                )
                return 0, 0, True

            self._cnt += 1

            ff = TURN_FF_BEFORE
            angle_raw = ff * self._turn_dir + TURN_PROP_K * blended_error

            if TURN_CLAMP_SAME_DIR:
                if self._turn_dir > 0:
                    angle = float(np.clip(angle_raw, 0.0, TURN_MAX_ANGLE))
                else:
                    angle = float(np.clip(angle_raw, -TURN_MAX_ANGLE, 0.0))
            else:
                angle = float(np.clip(angle_raw, -TURN_MAX_ANGLE, TURN_MAX_ANGLE))

            self._last_ff_used = float(ff * self._turn_dir)
            self._last_angle_raw = float(angle_raw)

            if self._initial_sign == 0:
                if abs(blended_error) > CROSS_ZERO_DEADBAND:
                    self._initial_sign = 1 if blended_error > 0 else -1

            if (self._initial_sign != 0
                    and not self._crossed_zero
                    and blended_error * self._initial_sign < 0
                    and abs(blended_error) > CROSS_ZERO_DEADBAND):
                self._crossed_zero = True
                self._bias_decay = True
                print(f"[NAV] CROSS-ZERO @cnt={self._cnt} err={blended_error:+.1f}")

            if self._crossed_zero and abs(blended_error) < TURN_STABLE_THRESH:
                self._stable_cnt += 1
            else:
                self._stable_cnt = 0

            lane_stable = (base_w and cur_w
                           and base_w > 0
                           and NEW_LANE_WIDTH_LO * base_w < cur_w < NEW_LANE_WIDTH_HI * base_w)
            if W > 0 and bottom_cx >= 0:
                offset = abs(bottom_cx - W / 2.0)
                self._new_lane_offset = offset
                if offset < NEW_LANE_CENTER_THRESH and lane_stable and self.heading_ok:
                    self._new_lane_cnt += 1
                else:
                    self._new_lane_cnt = 0
            else:
                self._new_lane_cnt = 0
                self._new_lane_offset = -1.0

            passed_min = self._cnt >= NEW_LANE_MIN_FRAMES
            new_lane_detected = self._new_lane_cnt >= NEW_LANE_STABLE_COUNT
            timeout = self._cnt >= MAX_TURN_FRAMES

            done = False
            reason = ""
            if timeout:
                done = True
                reason = f"timeout({self._cnt})"
            elif passed_min and new_lane_detected:
                done = True
                reason = (f"new_lane(bc={bottom_cx},W={W},"
                          f"off={self._new_lane_offset:.0f})")

            if not done and self._cnt % 5 == 0:
                print(f"[NAV] turning cnt={self._cnt} "
                      f"bc={bottom_cx} W={W} "
                      f"offset={self._new_lane_offset:.0f} "
                      f"new_lane_cnt={self._new_lane_cnt} "
                      f"lane_stable={lane_stable} "
                      f"cur_w={cur_w} base_w={base_w}")

            if done:
                self._last_exit_reason = reason
                print(f"[NAV] turn done [{reason}] — frames={self._cnt} "
                      f"crossed={self._crossed_zero} "
                      f"new_lane_cnt={self._new_lane_cnt} "
                      f"err={blended_error:+.1f}")
                self.state = 'cooldown'
                self._cnt = 0
                self._stable_cnt = 0
                self._new_lane_cnt = 0
                self.voter.reset()
                self.just_entered_cooldown = True

            return TURN_SPEED, angle, True

        # --- COOLDOWN ---
        if self.state == 'cooldown':
            self._cnt += 1
            if self._cnt >= COOLDOWN_FRAMES:
                print(f"[NAV] → NORMAL")
                self.state = 'normal'
                self._cnt = 0
                self._turn_dir = 0
                self._pending_turn_dir = 0
                self._final_commit = None
                self._bias_decay = False
                self._last_ff_used = 0.0
                self._last_angle_raw = 0.0
                self._last_exit_reason = ""
                self._new_lane_cnt = 0
                self._new_lane_offset = -1.0
                self._topo_match_cnt = 0
                self._early_topo_cnt = 0
                self._early_topo_age = 0
                self._early_topo = {'left': False, 'straight': True,
                                    'right': False, 'straight_ratio': 1.0,
                                    'top_half_empty': False,
                                    'top_half_coverage': 1.0}
                self._turn_mask_lost_cnt = 0
                self._turn_mask_recover_cnt = 0
                self._turn_mask_wait = False
            return 0, 0, False

        return 0, 0, False

    def _decide_turn(self, sign, topo):
        """
        turn_left -> -1, turn_right -> +1,
        straight / no_turn_left / no_turn_right -> 0 (đi thẳng).
        (Đã bỏ logic FORCE rẽ ngược hướng cấm của V4.6.1.)
        """
        if sign == 'turn_left':
            return -1
        if sign == 'turn_right':
            return +1
        return 0

    def bias_scale(self):
        return BIAS_DECAY_FACTOR if self._bias_decay else 1.0

    def get_ff_used(self):
        return self._last_ff_used

    def get_angle_raw(self):
        return self._last_angle_raw

    def get_exit_reason(self):
        return self._last_exit_reason

    def get_new_lane_cnt(self):
        return self._new_lane_cnt



# ==============================================================
# SIGN DETECT
# ==============================================================
def sign_detect(raw_image, conf=SIGN_CONF, imgsz=SIGN_IMGSZ):
    results = sign_model.predict(
        source=raw_image, conf=conf, imgsz=imgsz, verbose=False
    )
    classes = []
    if results and results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = sign_model.names[cls_id]
            classes.append(cls_name)
    return classes



# ==============================================================
# TOPOLOGY
# ==============================================================
def _compute_top_half_stats(binary, H):
    """V4.6.1: tính mật độ mask ở nửa trên ảnh (y = 0..H*TOP_HALF_RATIO)."""
    if binary is None or H <= 0:
        return False, 1.0
    top_h = max(1, int(H * TOP_HALF_RATIO))
    top_zone = binary[:top_h, :]
    if top_zone.size > 0:
        cov = float(np.count_nonzero(top_zone)) / top_zone.size
    else:
        cov = 0.0
    return (cov < TOP_HALF_EMPTY_THRESH), cov


def get_topology(mask, W, H,
                 y_lo_ratio=Y_LO_RATIO,
                 y_hi_ratio=Y_HI_RATIO,
                 min_branch_width=MIN_BRANCH_WIDTH):
    """
    Topology V4.6.1

    Tra ve:
        {'left': bool, 'straight': bool, 'right': bool,
         'straight_ratio': float,
         'top_half_empty': bool,
         'top_half_coverage': float}
    """

    if mask is None or W <= 0 or H <= 0:
        return {'left': False, 'straight': True, 'right': False,
                'straight_ratio': 1.0,
                'top_half_empty': False, 'top_half_coverage': 1.0}

    binary = (mask > 0).astype(np.uint8)

    bottom_y0 = int(H * 0.72)
    bottom_y1 = int(H * 0.95)

    lane_left_samples = []
    lane_right_samples = []

    for y in range(bottom_y0, bottom_y1, 3):
        xs = np.flatnonzero(binary[y])
        if xs.size >= 2:
            width = int(xs[-1] - xs[0])
            if int(W * 0.15) <= width <= int(W * 0.90):
                lane_left_samples.append(int(xs[0]))
                lane_right_samples.append(int(xs[-1]))

    if lane_left_samples:
        lane_left = float(np.median(lane_left_samples))
        lane_right = float(np.median(lane_right_samples))
    else:
        lane_left = 0.25 * W
        lane_right = 0.75 * W

    lane_width = lane_right - lane_left

    if lane_width < max(float(min_branch_width), 0.20 * W) \
            or lane_width > 0.85 * W:
        lane_left = 0.25 * W
        lane_right = 0.75 * W
        lane_width = lane_right - lane_left

    margin = max(4.0, 0.02 * W)
    left_boundary = max(0.0, lane_left - margin)
    right_boundary = min(float(W), lane_right + margin)

    y0 = max(0, int(H * y_lo_ratio))
    y1 = min(H - 1, int(H * y_hi_ratio))

    if y1 <= y0:
        top_empty, top_cov = _compute_top_half_stats(binary, H)
        return {'left': False, 'straight': True, 'right': False,
                'straight_ratio': 1.0,
                'top_half_empty': bool(top_empty),
                'top_half_coverage': float(top_cov)}

    NUM_Y_SAMPLES = 28
    SIDE_COVERAGE_THRESH = 0.06
    CENTER_COVERAGE_THRESH = 0.05
    SIDE_VOTE_RATIO = 0.18
    CENTER_VOTE_RATIO = 0.20
    MIN_BRANCH_RUN = 4

    sample_ys = np.linspace(y0, y1, NUM_Y_SAMPLES).astype(int)

    left_hits = []
    center_hits = []
    right_hits = []

    left_coverages = []
    center_coverages = []
    right_coverages = []

    for y in sample_ys:
        row = binary[y]

        lx = int(round(left_boundary))
        rx = int(round(right_boundary))
        lx = max(0, min(W, lx))
        rx = max(lx, min(W, rx))

        left_zone = row[:lx]
        center_zone = row[lx:rx]
        right_zone = row[rx:]

        left_cov = (
            float(np.count_nonzero(left_zone)) / len(left_zone)
            if len(left_zone) else 0.0
        )
        center_cov = (
            float(np.count_nonzero(center_zone)) / len(center_zone)
            if len(center_zone) else 0.0
        )
        right_cov = (
            float(np.count_nonzero(right_zone)) / len(right_zone)
            if len(right_zone) else 0.0
        )

        left_coverages.append(left_cov)
        center_coverages.append(center_cov)
        right_coverages.append(right_cov)

        left_hits.append(left_cov >= SIDE_COVERAGE_THRESH)
        center_hits.append(center_cov >= CENTER_COVERAGE_THRESH)
        right_hits.append(right_cov >= SIDE_COVERAGE_THRESH)

    def longest_run(values):
        best = 0
        cur = 0
        for value in values:
            if value:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return best

    left_ratio = sum(left_hits) / len(left_hits)
    center_ratio = sum(center_hits) / len(center_hits)
    right_ratio = sum(right_hits) / len(right_hits)

    left_run = longest_run(left_hits)
    right_run = longest_run(right_hits)

    left = (
        left_ratio >= SIDE_VOTE_RATIO
        and left_run >= MIN_BRANCH_RUN
    )

    right = (
        right_ratio >= SIDE_VOTE_RATIO
        and right_run >= MIN_BRANCH_RUN
    )

    straight = center_ratio >= CENTER_VOTE_RATIO

    if not straight:
        center_area = float(np.mean(center_coverages))
        straight = center_area >= CENTER_COVERAGE_THRESH * 0.75

    if right_run < MIN_BRANCH_RUN:
        right = False

    if left_run < MIN_BRANCH_RUN:
        left = False

    if not left and not straight and not right:
        straight = True

    top_empty, top_cov = _compute_top_half_stats(binary, H)

    return {
        'left': bool(left),
        'straight': bool(straight),
        'right': bool(right),
        'straight_ratio': float(center_ratio),
        'top_half_empty': bool(top_empty),
        'top_half_coverage': float(top_cov),
    }


def get_early_topology(mask, W, H):
    """
    Early topology detector.
    """
    if mask is None or W <= 0 or H <= 0:
        return {'left': False, 'straight': True, 'right': False,
                'straight_ratio': 1.0,
                'top_half_empty': False, 'top_half_coverage': 1.0}

    binary = (mask > 0).astype(np.uint8)

    yb0 = int(H * 0.72)
    yb1 = int(H * 0.95)
    lefts, rights = [], []
    for y in range(yb0, yb1, 3):
        xs = np.flatnonzero(binary[y])
        if xs.size >= 2:
            w = int(xs[-1] - xs[0])
            if int(W * EARLY_TOPO_MIN_WIDTH_RATIO) <= w <= int(W * EARLY_TOPO_MAX_WIDTH_RATIO):
                lefts.append(int(xs[0]))
                rights.append(int(xs[-1]))

    if lefts:
        lane_left = float(np.median(lefts))
        lane_right = float(np.median(rights))
    else:
        lane_left = 0.25 * W
        lane_right = 0.75 * W

    if lane_right - lane_left < W * 0.20:
        lane_left, lane_right = 0.25 * W, 0.75 * W

    margin = max(4.0, 0.02 * W)
    lx = max(0, min(W, int(round(lane_left - margin))))
    rx = max(lx, min(W, int(round(lane_right + margin))))

    y0 = max(0, int(H * EARLY_TOPO_Y_LO_RATIO))
    y1 = min(H - 1, int(H * EARLY_TOPO_Y_HI_RATIO))
    ys = np.linspace(y0, y1, 36).astype(int)

    left_hits, center_hits, right_hits = [], [], []
    for y in ys:
        row = binary[y]
        lz, cz, rz = row[:lx], row[lx:rx], row[rx:]
        lc = np.count_nonzero(lz) / len(lz) if len(lz) else 0.0
        cc = np.count_nonzero(cz) / len(cz) if len(cz) else 0.0
        rc = np.count_nonzero(rz) / len(rz) if len(rz) else 0.0
        left_hits.append(lc >= 0.05)
        center_hits.append(cc >= 0.04)
        right_hits.append(rc >= 0.05)

    def ratio(v):
        return sum(v) / len(v) if v else 0.0

    def longest_run(v):
        best = cur = 0
        for x in v:
            cur = cur + 1 if x else 0
            best = max(best, cur)
        return best

    left = ratio(left_hits) >= 0.14 and longest_run(left_hits) >= 4
    right = ratio(right_hits) >= 0.14 and longest_run(right_hits) >= 4
    straight = ratio(center_hits) >= 0.18

    if not straight and not left and not right:
        straight = True

    top_empty, top_cov = _compute_top_half_stats(binary, H)

    return {'left': bool(left), 'straight': bool(straight),
            'right': bool(right),
            'straight_ratio': float(ratio(center_hits)),
            'top_half_empty': bool(top_empty),
            'top_half_coverage': float(top_cov)}



# ==============================================================
# DECISION -> bias
# ==============================================================
def decide_bias(committed, topo, base_width, scale=1.0):
    if base_width is None or base_width <= 0 or committed is None:
        return 0
    bias = 0
    if committed == 'turn_left':
        bias = +BIAS_RATIO * base_width
    elif committed == 'turn_right':
        bias = -BIAS_RATIO * base_width
    elif committed == 'straight':
        bias = 0
    elif committed == 'no_turn_left':
        bias = -BIAS_RATIO * base_width if topo['right'] else 0
    elif committed == 'no_turn_right':
        bias = +BIAS_RATIO * base_width if topo['left'] else 0
    return int(bias * scale)



# ==============================================================
# LANE WIDTH ESTIMATOR
# ==============================================================
class LaneWidthEstimator:
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
            non_zero_cols = np.where(row > 0)[0]
            if len(non_zero_cols) > 1:
                left = non_zero_cols[0]
                right = non_zero_cols[-1]
                lane_width = right - left
                widths.append(lane_width)
                self.width_profile.append((y, lane_width))

        if widths:
            self.current_width = np.median(widths)
            if self.base_width is None:
                self.base_width = self.current_width

        return self.current_width, self.width_profile

    def is_wide_change(self, threshold=0.2):
        if self.base_width is None or not self.width_profile:
            return False
        for _, w in self.width_profile:
            if w > self.base_width * (1 + threshold):
                return True
        return False

    def reset_baseline(self):
        self.base_width = None



# ==============================================================
# LANE FOLLOWING CORE (từ ucr_main_v20)
# ==============================================================
class OcclusionGate:
    def __init__(self, err_step_px=18.0, coverage_drop_frac=0.22,
                 jagged_px=15.0, hard_step_px=30.0,
                 hangover_frames=4,
                 coverage_ema_alpha=0.9,
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
        self.authority = authority if authority is not None \
                         else AuthorityManager(output_limit)
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
        # [BQ] Forced-emergency flag; set externally when occl is active
        # or within EMG_HOLD_AFTER_OCCL frames of the last occlusion.
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
        # Preserve authority state and prev_output so `losing` can
        # accumulate across occlusion gaps and the rate limiter does
        # not see a spurious jump.
        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_derivative = 0.0
        self.first_call = True

    def step(self, error, now=None):
        kp_eff = self.kp * (1.5 if self.authority.losing else 1.0)
        kd_eff = self.kd * (0.7 if self.authority.losing else 1.0)
        self.last_kp_mult = kp_eff / self.kp if self.kp else 1.0
        self.last_kd_mult = kd_eff / self.kd if self.kd else 1.0

        # [BQ] Forced emergency overrides the streak logic.
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
        self.integral = np.clip(self.integral,
                                -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.integral

        raw_d = 0.0 if self.first_call else (error - self.prev_error)
        self.filtered_derivative = (
            self.derivative_alpha * raw_d
            + (1.0 - self.derivative_alpha) * self.filtered_derivative
        )
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


def plan_speed(curve_smoothed, max_speed=26.0, min_speed=0.0, curve_scale=80.0):
    curve_ratio = min(abs(curve_smoothed) / curve_scale, 1.0)
    return max_speed - (max_speed - min_speed) * curve_ratio, curve_ratio


def error_speed_factor(error, start_px=8.0, full_px=35.0, min_factor=0.30):
    a = abs(error)
    if a <= start_px: return 1.0
    if a >= full_px: return min_factor
    t = (a - start_px) / (full_px - start_px)
    return 1.0 - t * (1.0 - min_factor)



# ==============================================================
# TURN ERROR (chỉ dùng trong state 'turning')
# Port gọn từ calculate_steering_angle của maycayv9:
#   - lỗi near/far = centroid từng hàng so với (W/2 + bias)
#   - wide road -> bỏ far points
#   - bottom_cx để navigator nhận ra làn mới
# ==============================================================
class TurnErrorEstimator:
    def __init__(self):
        self.bottom_hist = deque(maxlen=3)
        self.far_off_raw = None   # far centroid - W/2 (không bias); None nếu không có far

    def reset(self):
        self.bottom_hist.clear()
        self.far_off_raw = None

    def step(self, gray, bias_pixel, speed, max_speed, wide_change):
        h, w = gray.shape
        binary = gray > 0
        row_counts = binary.sum(axis=1)
        rows = np.flatnonzero(row_counts > 0)
        if rows.size == 0:
            self.reset()
            return 0.0, -1

        xs = np.arange(w, dtype=np.float32)
        cx_rows = (binary[rows] * xs).sum(axis=1) / row_counts[rows]

        # bottom_cx: hàng dưới cùng có mask, median 3 frame
        self.bottom_hist.append(float(cx_rows[-1]))
        bottom_cx = int(round(float(np.median(self.bottom_hist))))

        target = w / 2.0 + bias_pixel
        near_sel = rows >= h // 2
        far_sel = rows < h // 3
        has_far = (not wide_change) and bool(far_sel.any())

        near_err = float(cx_rows[near_sel].mean() - target) if near_sel.any() else 0.0
        far_err = float(cx_rows[far_sel].mean() - target) if has_far else 0.0
        self.far_off_raw = float(cx_rows[far_sel].mean() - w / 2.0) if has_far else None

        if abs(near_err) + abs(far_err) < 10:
            w_near, w_far = 0.7, 0.3
        else:
            w_far = float(np.clip(0.5 + 0.25 * (speed / max_speed), 0.35, 0.75))
            w_near = float(np.clip(1.0 - w_far - 0.05, 0.25, 0.65))
            if near_err * far_err < 0:
                w_far *= 0.7
                w_near = float(np.clip(1.0 - w_far - 0.05, 0.25, 0.65))

        blended = w_near * near_err + w_far * far_err
        if abs(blended) < STRAIGHT_ERROR_DEADBAND:
            blended = 0.0
        return float(blended), bottom_cx


# ==============================================================
# LANE FOLLOWER  (toàn bộ pipeline PID của ucr_main_v20)
# ==============================================================
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
        """Gọi sau khi rẽ xong: làn mới nên xoá hết trạng thái cũ (trừ speed)."""
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

    def step(self, lane_mask, now, speed_cap=None, force_straight=False,
             near_only=False):
        gray = lane_mask
        near_e, far_e, curve_e, self.curve_smooth, row_pts, valid, coverage = \
            extract_centerline(gray, self.curve_smooth)
        if near_only:
            far_e = near_e   # bỏ far/lead: tránh bị nhánh ngã tư kéo lệch

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
            target_speed, curve_ratio = plan_speed(self.curve_smooth, MAX_SPEED, MIN_SPEED)
            err_speed_mult = error_speed_factor(eff_error, start_px=8.0,
                                                full_px=35.0, min_factor=0.30)
            auth_speed = 0.6 if self.authority.losing else 1.0
            anom_speed = 0.5 if abs(self.asym_smooth) > 15 else 1.0
            speed_mult = min(auth_speed, anom_speed)
            target_speed = max(target_speed * err_speed_mult * speed_mult, 0.0)
            speed = 0.7 * self.speed + 0.3 * target_speed
            self.last_angle = angle
        else:
            angle = 0.0 if force_straight else self.last_angle
            curve_ratio = 0.0
            speed = 0.85 * self.speed

        if speed_cap is not None:
            speed = min(speed, float(speed_cap))

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
            row_pts=row_pts,
            far_outlier=self.gate.last_far_outlier,
            far_spread=self.gate.last_far_spread,
        )
        return float(angle), self.speed, info


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
        cv2.putText(debug,
                    f"blend={info['blended']:+.1f} lead={info['lead']:+.1f} "
                    f"asym={info['asym']:+.0f}",
                    (10, 36), font, 0.45, (255, 255, 255), 1)
        cv2.putText(debug,
                    f"near={info['near']:+.0f} far={info['far']:+.0f} "
                    f"spd={speed:.1f} a={angle:+.1f}",
                    (10, 54), font, 0.45, (255, 255, 255), 1)
        cv2.putText(debug,
                    f"P={pid.last_p:+.1f} I={pid.last_i:+.1f} "
                    f"D={pid.last_d:+.1f} lim={pid.last_output_limit:.0f}",
                    (10, 72), font, 0.45, (200, 200, 255), 1)
        y = 90
        if not info["valid"]:
            cv2.putText(debug, "MASK INVALID - HOLD", (10, y), font, 0.5, (0, 0, 255), 1); y += 18
        if info["far_outlier"]:
            cv2.putText(debug, "FAR OUTLIER", (10, y), font, 0.5, (255, 128, 0), 1); y += 18
        if info["occl"]:
            cv2.putText(debug, "OCCLUDED", (10, y), font, 0.5, (255, 0, 255), 1); y += 18
        if info["forced_emer"]:
            cv2.putText(debug, f"FORCED EMERGENCY (hold={info['occl_emer_hold']})",
                        (10, y), font, 0.5, (0, 100, 255), 1)
        elif info["emer_auth"]:
            cv2.putText(debug, "EMERGENCY AUTH", (10, y), font, 0.5, (0, 0, 255), 1)
    else:
        cv2.putText(debug, f"NAV CONTROL spd={speed:.1f} a={angle:+.1f}",
                    (10, 36), font, 0.45, (255, 255, 255), 1)
    cv2.imshow("Lane Debug", debug)


# ==============================================================
# MAIN LOOP
# ==============================================================
if __name__ == "__main__":
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log_file = open(LOG_PATH, "w", newline="", buffering=1)
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "t", "frame", "nav_state", "commit", "commit_age", "turn_dir", "nav_cnt",
        "bias_px", "base_w", "cur_w", "ratio", "mask_trigger",
        "topoL", "topoS", "topoR", "top_cov", "sign_classes",
        "new_lane_cnt", "crossed_zero", "exit_reason", "turn_err", "bottom_cx",
        "near", "far", "blended", "eff_error", "asym",
        "occl", "forced_emer", "emer_auth", "losing",
        "p", "i", "d", "raw", "out_lim", "valid", "coverage",
        "speed", "angle",
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

            # ---------- Perception cho navigator (maycayv9) ----------
            topo = get_topology(seg_mask, W, H)

            sign_classes = sign_detect(raw_image)
            if navigator.state not in ('turning', 'cooldown'):
                voter.update(sign_classes)
            committed = voter.get_committed()
            commit_age = voter.get_commit_age()

            lane_width_estimator.measure_width(seg_mask)
            base_w = lane_width_estimator.base_width
            cur_w = lane_width_estimator.current_width
            mask_trigger = intersection_detector.update(base_w, cur_w)

            # Turn error chỉ cần khi đang rẽ
            bias_pixel = 0
            turn_err = 0.0
            bottom_cx = -1
            if navigator.state == 'turning':
                if not was_turning:
                    turn_est.reset()
                was_turning = True
                bias_pixel = decide_bias(committed, topo, base_w,
                                         scale=navigator.bias_scale())
                turn_err, bottom_cx = turn_est.step(
                    seg_mask, bias_pixel, TURN_SPEED, MAX_SPEED,
                    lane_width_estimator.is_wide_change())
                navigator.heading_ok = (turn_est.far_off_raw is None
                                        or abs(turn_est.far_off_raw) < TURN_EXIT_FAR_THRESH)
            else:
                was_turning = False

            effective_trigger = (mask_trigger and navigator.state == 'slow_down')
            nav_speed, nav_angle, is_active = navigator.update(
                sign_classes, committed, topo, effective_trigger,
                turn_err, bottom_cx, W, base_w, cur_w
            )

            if navigator.just_entered_cooldown:
                print("[MAIN] enter cooldown — reset lane baseline + PID state")
                lane_width_estimator.reset_baseline()
                follower.reset_tracking()
            if navigator.state in ('stopping', 'turning', 'cooldown'):
                intersection_detector.reset()

            # ---------- CONTROL ----------
            info = {}
            if is_active is True:
                # stopping / turning: navigator lái
                speed = float(nav_speed)
                angle = float(nav_angle) if nav_angle is not None else 0.0
                follower.speed = speed
                AVControl(speed, angle)
                print(f"[NAV] state={navigator.state} cnt={navigator._cnt} "
                      f"commit={committed} turn_dir={navigator._turn_dir} "
                      f"bias={bias_pixel} err={turn_err:+.1f} "
                      f"ff={navigator.get_ff_used():+.1f} "
                      f"raw={navigator.get_angle_raw():+.2f} "
                      f"ang={angle:+.2f} bc={bottom_cx} W={W} "
                      f"new_lane={navigator.get_new_lane_cnt()} "
                      f"cross={navigator._crossed_zero} spd={speed}")
            else:
                # normal / slow_down / cooldown: PID v20 lái
                speed_cap = None
                force_straight = False
                near_only = False
                if is_active == 'slow':
                    speed_cap = SLOW_SPEED
                    near_only = lane_width_estimator.is_wide_change()
                elif navigator.state == 'cooldown':
                    speed_cap = COOLDOWN_SPEED_CAP
                    force_straight = navigator._cnt <= COOLDOWN_FORCE_STRAIGHT_FRAMES
                    near_only = navigator._cnt <= POST_TURN_NEAR_ONLY_FRAMES

                angle, speed, info = follower.step(
                    lane_mask, t_now, speed_cap=speed_cap,
                    force_straight=force_straight, near_only=near_only)
                AVControl(speed, angle)

                if force_straight:
                    tag = "COOLDOWN-FORCE"
                elif info["occl"] and info["forced_emer"]:
                    tag = "F"
                elif info["occl"]:
                    tag = "O"
                elif info["emer_auth"]:
                    tag = "E"
                elif info["losing"]:
                    tag = "L"
                else:
                    tag = navigator.state.upper()
                ratio = (cur_w / base_w) if (base_w and base_w > 0) else 0
                print(f"[{tag}] id=- topo=L{int(topo['left'])}S{int(topo['straight'])}"
                      f"R{int(topo['right'])} commit={committed} "
                      f"r={ratio:.2f} near={info['near']:+.0f} far={info['far']:+.0f} "
                      f"eff={info['eff_error']:+.1f} occl={int(info['occl'])} "
                      f"hold={info['occl_emer_hold']} "
                      f"spd={speed:.1f} ang={angle:+.2f} valid={int(info['valid'])}")

            # ---------- LOG ----------
            ratio_log = (cur_w / base_w) if (base_w and base_w > 0) else 0
            pid = follower.pid
            has = bool(info)
            log_writer.writerow([
                f"{t_now:.3f}", frame_idx, navigator.state,
                committed if committed else "", commit_age,
                navigator._turn_dir,
                navigator._cnt if navigator.state in ('turning', 'cooldown', 'stopping') else 0,
                bias_pixel, base_w if base_w else 0, cur_w if cur_w else 0,
                f"{ratio_log:.3f}", int(mask_trigger),
                int(topo['left']), int(topo['straight']), int(topo['right']),
                f"{topo.get('top_half_coverage', 1.0):.3f}",
                "|".join(sign_classes),
                navigator.get_new_lane_cnt() if navigator.state == 'turning' else 0,
                int(navigator._crossed_zero), navigator.get_exit_reason(),
                f"{turn_err:+.3f}", bottom_cx,
                f"{info['near']}" if has else "",
                f"{info['far']}" if has else "",
                f"{info['blended']:.3f}" if has else "",
                f"{info['eff_error']:.3f}" if has else "",
                f"{info['asym']:.3f}" if has else "",
                int(info['occl']) if has else "",
                int(info['forced_emer']) if has else "",
                int(info['emer_auth']) if has else "",
                int(info['losing']) if has else "",
                f"{pid.last_p:.4f}" if has else "",
                f"{pid.last_i:.4f}" if has else "",
                f"{pid.last_d:.4f}" if has else "",
                f"{pid.last_raw:.4f}" if has else "",
                f"{pid.last_output_limit:.1f}" if has else "",
                int(info['valid']) if has else "",
                f"{info['coverage']:.4f}" if has else "",
                f"{speed:.3f}", f"{angle:.3f}",
            ])
            if frame_idx % LOG_FLUSH_EVERY == 0:
                log_file.flush()

            # ---------- GUI ----------
            if ENABLE_GUI:
                nav_text = (f"NAV={navigator.state} commit={committed} "
                            f"mask_trig={int(mask_trigger)}")
                draw_debug(lane_mask, info, pid, angle, speed, nav_text)
                cv2.imshow('Raw', raw_image)
                cv2.imshow('Road Mask', seg_mask)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        print('Closing socket and windows...')
        log_file.close()
        AVControl(0, 0)
        CloseSocket()
        cv2.destroyAllWindows()

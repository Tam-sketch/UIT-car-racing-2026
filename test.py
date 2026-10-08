# ==============================================================
# UIT CAR RACING 2026 - VÒNG 2 (V4.6.1)
# Base: V4.5
# Changes vs V4.5:
#   [V4.6.1-1] get_topology / get_early_topology trả thêm
#              'top_half_empty' + 'top_half_coverage'
#   [V4.6.1-2] no_turn_left / no_turn_right:
#              - Nếu nửa trên ảnh (y=0..H/2) TRỐNG mask
#                → FORCE rẽ ngược hướng cấm.
#              - Nếu nửa trên có mask (thấy đường thẳng phía xa)
#                → đi thẳng.
#   [V4.6.1-3] _forced_opposite bypass topo gate khi trigger STOPPING.
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
ROAD_MODEL_PATH = "/workspace/best14.pt"
SIGN_MODEL_PATH = "/workspace/sign.pt"

ROAD_IMGSZ = 320
SIGN_IMGSZ = 640
SIGN_CONF  = 0.7

MIN_FRAMES = 4
MAX_GAP    = 3
BIAS_RATIO = 0.3

Y_LO_RATIO = 0.15
Y_HI_RATIO = 0.70
MIN_BRANCH_WIDTH = 12

# --- Speed ---
SLOW_SPEED = 10
TURN_SPEED = 12

# --- Turn angle ---
TURN_FF_BEFORE      = 20
TURN_FF_AFTER       = 10
TURN_PROP_K         = 0.15
TURN_MAX_ANGLE      = 25
TURN_CLAMP_SAME_DIR = True

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

MAX_TURN_FRAMES = 90

# --- Straight-line stabilization ---
STRAIGHT_ERROR_DEADBAND = 2.0
STRAIGHT_CURVE_RATIO = 0.06
STRAIGHT_ANGLE_LIMIT = 10.0

# --- V3.10 hybrid NORMAL steering ---
HYBRID_CURVE_RATIO = 0.10
HYBRID_ERROR_RATIO = 0.18
HYBRID_K_MIN = 0.06
HYBRID_K_MAX = 0.30
HYBRID_MAX_ANGLE = 25.0
HYBRID_CURVE_SPEED_FACTOR = 1.20
HYBRID_AGGRESSIVE_GAIN = 5.0

# --- Turn mask-loss recovery ---
TURN_MASK_LOST_TRIGGER_FRAMES = 20
TURN_MASK_RECOVER_FRAMES = 2

# --- Bias decay ---
BIAS_DECAY_FACTOR = 0.3

# --- Cooldown ---
COOLDOWN_FRAMES                = 30
COOLDOWN_SPEED_CAP             = 18
COOLDOWN_FORCE_STRAIGHT_FRAMES = 5

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

# --- Voter ---
SIGN_TIMEOUT_FRAMES = 300

# --- Log ---
LOG_PATH = "/workspace/my_code/pid_log_v3.csv"
LOG_FLUSH_EVERY = 10

ENABLE_GUI = os.environ.get("ENABLE_GUI", "1") != "0"

# --- V4.6.1: Top-half emptiness check ---
# Nếu vùng trên ảnh (y = 0 → TOP_HALF_RATIO*H) có mật độ mask
# < TOP_HALF_EMPTY_THRESH → coi như KHÔNG có đường thẳng phía trước.
TOP_HALF_RATIO = 0.5            # lấy nửa trên (y = 0 → 90 với H=180)
TOP_HALF_EMPTY_THRESH = 0.02    # < 2% pixel có mask → coi là trống


# ==============================================================
# LOAD MODELS
# ==============================================================
print("Loading models...")
road_model = YOLO(ROAD_MODEL_PATH)
sign_model = YOLO(SIGN_MODEL_PATH)
print(f"Road model classes: {road_model.names}")
print(f"Sign model classes: {sign_model.names}")


# ==============================================================
# CSV LOGGER
# ==============================================================
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
log_file = open(LOG_PATH, "w", newline="", buffering=1)
log_writer = csv.writer(log_file)
log_writer.writerow([
    "t", "frame", "nav_state", "raw_state",
    "commit", "commit_age", "turn_dir", "turn_frames",
    "bias_px", "bias_scale", "base_w", "cur_w", "ratio",
    "err", "angle", "speed",
    "bottom_cx", "top_cx", "lane_heading_deg",
    "ff_used", "angle_raw",
    "new_lane_cnt",
    "curve_ratio", "k",
    "pid_p", "pid_i", "pid_d", "pid_raw",
    "sign_classes", "mask_trigger",
    "crossed_zero", "stable_cnt", "initial_sign",
    "exit_reason",
])
print(f"[log] writing to {LOG_PATH}")


# ==============================================================
# PID CONTROLLER — NORMAL LANE FOLLOWING ONLY
# ==============================================================
class SteeringPID:
    """Continuous PID used only in NORMAL lane following."""

    def __init__(self, kp=0.32, ki=0.003, kd=0.75,
                 integral_limit=3.0, derivative_alpha=0.55,
                 output_limit=20.0, rate_limit=8.0,
                 adaptive_d_boost=0.10, adaptive_d_cap=2.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.derivative_alpha = derivative_alpha
        self.output_limit = output_limit
        self.rate_limit = rate_limit
        self.adaptive_d_boost = adaptive_d_boost
        self.adaptive_d_cap = adaptive_d_cap
        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_derivative = 0.0
        self.prev_output = 0.0
        self.first_call = True
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
        self.last_p = self.last_i = self.last_d = self.last_raw = 0.0

    def step(self, error):
        error = float(error)
        p_term = self.kp * error

        self.integral += error
        self.integral = float(np.clip(self.integral,
                                      -self.integral_limit,
                                      self.integral_limit))
        i_term = self.ki * self.integral

        if self.first_call:
            raw_derivative = 0.0
        else:
            raw_derivative = error - self.prev_error

        self.filtered_derivative = (
            self.derivative_alpha * raw_derivative
            + (1.0 - self.derivative_alpha) * self.filtered_derivative
        )

        velocity_mag = abs(self.filtered_derivative)
        d_boost = 1.0 + self.adaptive_d_boost * min(velocity_mag, 20.0)
        d_boost = min(d_boost, self.adaptive_d_cap)
        d_term = self.kd * d_boost * self.filtered_derivative

        raw_output = p_term + i_term + d_term
        output = self.output_limit * np.tanh(raw_output / self.output_limit)

        delta = np.clip(output - self.prev_output,
                        -self.rate_limit, self.rate_limit)
        output = self.prev_output + delta

        self.prev_error = error
        self.prev_output = float(output)
        self.first_call = False
        self.last_p = p_term
        self.last_i = i_term
        self.last_d = d_term
        self.last_raw = raw_output
        return float(output)


# ==============================================================
# STATE
# ==============================================================
class LastSpeed:
    value = 30


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

        self._turn_mask_lost_cnt = 0
        self._turn_mask_recover_cnt = 0
        self._turn_mask_wait = False

        self._topo_match_cnt = 0

        # V4.6.1: cờ báo sign ép rẽ ngược hướng cấm
        self._forced_opposite = False

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
               blended_error, lane_heading_deg,
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

            # V4.6.1: forced opposite + hình học đủ rộng → cho phép fire luôn
            if self._forced_opposite and safe_geometry and potential_dir != 0:
                topo_trigger = True

            if mask_trigger:
                mask_topo_match = False

                if potential_dir < 0 and topo['left']:
                    mask_topo_match = True
                elif potential_dir > 0 and topo['right']:
                    mask_topo_match = True
                elif potential_dir == 0:
                    mask_topo_match = True

                # V4.6.1: forced opposite → tin sign, bypass topo gate
                if not mask_topo_match and self._forced_opposite and potential_dir != 0:
                    print(
                        f"[NAV] forced-opposite — bypass topo gate "
                        f"(dir={potential_dir}, "
                        f"top_cov="
                        f"{topo_for_decision.get('top_half_coverage', 1.0):.3f})"
                    )
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
                    self._forced_opposite = False

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
                    f"forced={int(self._forced_opposite)}, "
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
                if offset < NEW_LANE_CENTER_THRESH and lane_stable:
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
                self._forced_opposite = False
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
        V4.6.1:
          - no_turn_left  + nửa trên ảnh CÓ mask  → 0 (đi thẳng)
          - no_turn_left  + nửa trên ảnh TRỐNG    → +1 (FORCE turn_right)
          - no_turn_right + nửa trên ảnh CÓ mask  → 0 (đi thẳng)
          - no_turn_right + nửa trên ảnh TRỐNG    → -1 (FORCE turn_left)
        """
        self._forced_opposite = False

        if sign is None:
            return 0
        if sign == 'turn_left':
            return -1
        if sign == 'turn_right':
            return +1
        if sign == 'straight':
            return 0

        top_half_empty = bool(topo.get('top_half_empty', False))
        top_half_cov   = float(topo.get('top_half_coverage', 1.0))

        if sign == 'no_turn_left':
            if not top_half_empty:
                return 0
            self._forced_opposite = True
            print(
                f"[NAV] no_turn_left + top-half empty "
                f"(top_cov={top_half_cov:.3f}) → FORCE turn_right"
            )
            return +1

        if sign == 'no_turn_right':
            if not top_half_empty:
                return 0
            self._forced_opposite = True
            print(
                f"[NAV] no_turn_right + top-half empty "
                f"(top_cov={top_half_cov:.3f}) → FORCE turn_left"
            )
            return -1

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


# Khởi tạo toàn cục
voter = SignVoter(min_frames=MIN_FRAMES, max_gap=MAX_GAP)
navigator = SlowStopTurnNavigator(voter=voter)
intersection_detector = IntersectionDetector()
intersection_id = 0
prev_state = 0
last_speed = LastSpeed()
lane_width_estimator = None


# ==============================================================
# PERCEPTION
# ==============================================================
def get_yolo_road_mask(raw_image):
    results = road_model.predict(
        source=raw_image, imgsz=ROAD_IMGSZ, verbose=False
    )
    if not results or results[0].masks is None:
        return np.zeros(raw_image.shape[:2], dtype=np.uint8)
    masks = results[0].masks.data.cpu().numpy()
    combined = (np.sum(masks, axis=0) > 0).astype(np.uint8) * 255
    return combined


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


def topology_to_state(topo):
    state = 0
    if topo['left']:
        state |= 1
    if topo['right']:
        state |= 2
    return state


# ==============================================================
# DECISION → bias
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

    def measure_width(self, gray_image, green_line_points, step=5, fraction=2):
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
# STEERING HEURISTIC
# ==============================================================
def calculate_steering_angle(segment_image, speed=30,
                             k_min=0.06, k_max=0.3, max_speed=40,
                             lane_width_estimator=None,
                             bias_pixel=0,
                             show_debug=False,
                             update_filter=True):
    if len(segment_image.shape) == 3:
        gray = cv2.cvtColor(segment_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = segment_image.copy()

    height, width = gray.shape
    green_line_points = []

    for i in range(height - 1, -1, -1):
        row = gray[i, :]
        non_zero_cols = np.where(row > 0)[0]
        if len(non_zero_cols) > 0:
            cx_row = int(np.mean(non_zero_cols))
            green_line_points.append((cx_row, i))

    lane_cx = width // 2
    curve_error = 0
    near_error_raw = 0.0
    far_error_raw = 0.0

    near_points = [(cx, y) for cx, y in green_line_points if y >= height // 2]
    far_points = [(cx, y) for cx, y in green_line_points if y < height // 3]

    bottom_cx_raw = -1
    top_cx = -1
    lane_heading_deg = 0.0

    if len(green_line_points) >= 2:
        bottom_cx_raw, bottom_y = green_line_points[0]
        top_cx, top_y = green_line_points[-1]
        dy = bottom_y - top_y
        dx = top_cx - bottom_cx_raw
        if dy > 0:
            lane_heading_deg = float(np.degrees(np.arctan2(dx, dy)))

    wide_change = False
    if lane_width_estimator is not None:
        cur_w, width_profile = lane_width_estimator.measure_width(gray, green_line_points)
        wide_change = lane_width_estimator.is_wide_change()

    if wide_change:
        far_points = []

    if green_line_points:
        weighted_sum = sum(cx * (height - y) for cx, y in green_line_points)
        total_weight = sum(height - y for _, y in green_line_points)

        if total_weight > 0:
            lane_cx = int(weighted_sum / total_weight)

        if len(green_line_points) >= 2:
            bottom_row_cx = green_line_points[0][0]
            top_row_cx = green_line_points[-1][0]
            curve_error = top_row_cx - bottom_row_cx

        target_cx = width // 2 + bias_pixel

        if near_points:
            near_error_raw = float(np.mean([cx for cx, y in near_points])) - target_cx

        if far_points:
            far_error_raw = float(np.mean([cx for cx, y in far_points])) - target_cx

    else:
        if hasattr(calculate_steering_angle, "_filter_state"):
            fs = calculate_steering_angle._filter_state
            fs["bottom_hist"].clear()
            fs["near_hist"].clear()
            fs["far_hist"].clear()
            fs["blend_hist"].clear()
            fs["bottom_value"] = None
            fs["near_value"] = None
            fs["far_value"] = None
            fs["blend_value"] = None

        return (0.0, 0.0, k_min, 0.5, 0.5, 0.0, -1, -1, 0.0)

    if not hasattr(calculate_steering_angle, "_filter_state"):
        calculate_steering_angle._filter_state = {
            "bottom_hist": [],
            "near_hist": [],
            "far_hist": [],
            "blend_hist": [],
            "bottom_value": None,
            "near_value": None,
            "far_value": None,
            "blend_value": None,
            "prev_bias": 0.0,
            "last_result": None,
        }

    fs = calculate_steering_angle._filter_state

    def reset_error_filter():
        fs["near_hist"].clear()
        fs["far_hist"].clear()
        fs["blend_hist"].clear()
        fs["near_value"] = None
        fs["far_value"] = None
        fs["blend_value"] = None

    def filter_value(hist, state_name, raw_value, alpha):
        raw_value = float(raw_value)
        hist.append(raw_value)
        if len(hist) > 3:
            hist.pop(0)

        median_value = float(np.median(hist))
        previous = fs[state_name]

        if previous is None:
            filtered = median_value
        else:
            if median_value * previous < 0 and abs(median_value) > 10.0:
                alpha_use = max(alpha, 0.85)
            else:
                alpha_use = alpha

            filtered = (1.0 - alpha_use) * previous + alpha_use * median_value

        fs[state_name] = filtered
        return filtered

    half_width = max(width // 2, 1)

    raw_curve_ratio = min(abs(curve_error) / half_width, 1.0)
    raw_error_ratio = min(
        (abs(near_error_raw) + abs(far_error_raw)) / width,
        1.0
    )

    curve_level = np.clip(raw_curve_ratio / 0.12, 0.0, 1.0)
    error_level = np.clip(raw_error_ratio / 0.20, 0.0, 1.0)
    dynamic_level = max(curve_level, error_level)

    alpha_bottom = 0.55 + 0.40 * dynamic_level
    alpha_error = 0.65 + 0.35 * dynamic_level
    alpha_blended = 0.70 + 0.30 * dynamic_level

    bypass_error_filter = (
        raw_curve_ratio >= 0.10
        or raw_error_ratio >= 0.18
    )

    if abs(float(bias_pixel) - fs["prev_bias"]) > 8.0:
        reset_error_filter()

    fs["prev_bias"] = float(bias_pixel)

    if update_filter:
        bottom_cx_filtered = filter_value(
            fs["bottom_hist"],
            "bottom_value",
            bottom_cx_raw,
            alpha_bottom
        )
    else:
        bottom_cx_filtered = (
            fs["bottom_value"]
            if fs["bottom_value"] is not None
            else float(bottom_cx_raw)
        )

    bottom_cx = int(round(bottom_cx_filtered))

    if bypass_error_filter:
        near_error = near_error_raw
        far_error = far_error_raw if far_points else 0.0

        if update_filter:
            fs["near_hist"] = [float(near_error_raw)]
            fs["near_value"] = float(near_error_raw)

            if far_points:
                fs["far_hist"] = [float(far_error_raw)]
                fs["far_value"] = float(far_error_raw)
            else:
                fs["far_hist"].clear()
                fs["far_value"] = None
    else:
        if update_filter:
            near_error = filter_value(
                fs["near_hist"],
                "near_value",
                near_error_raw,
                alpha_error
            )

            if far_points:
                far_error = filter_value(
                    fs["far_hist"],
                    "far_value",
                    far_error_raw,
                    alpha_error
                )
            else:
                fs["far_hist"].clear()
                fs["far_value"] = None
                far_error = 0.0
        else:
            near_error = (
                fs["near_value"]
                if fs["near_value"] is not None
                else near_error_raw
            )

            far_error = (
                fs["far_value"]
                if far_points and fs["far_value"] is not None
                else (far_error_raw if far_points else 0.0)
            )

    abs_error = abs(near_error) + abs(far_error)

    straight_candidate = (
        raw_curve_ratio < STRAIGHT_CURVE_RATIO
        and raw_error_ratio < 0.12
    )

    if straight_candidate:
        w_near = 0.85
        w_far = 0.15
    elif abs_error < 10:
        w_near = 0.7
        w_far = 0.3
    else:
        w_far = 0.5 + 0.25 * (speed / max_speed)
        w_far = np.clip(w_far, 0.35, 0.75)
        w_near = 1.0 - w_far - 0.05
        w_near = np.clip(w_near, 0.25, 0.65)

        if near_error * far_error < 0:
            w_far *= 0.7
            w_near = 1.0 - w_far - 0.05
            w_near = np.clip(w_near, 0.25, 0.65)

    blended_error_raw = w_near * near_error + w_far * far_error

    if bypass_error_filter:
        blended_error = blended_error_raw
        if update_filter:
            fs["blend_hist"] = [float(blended_error_raw)]
            fs["blend_value"] = float(blended_error_raw)
    else:
        if update_filter:
            blended_error = filter_value(
                fs["blend_hist"],
                "blend_value",
                blended_error_raw,
                alpha_blended
            )
        else:
            blended_error = (
                fs["blend_value"]
                if fs["blend_value"] is not None
                else blended_error_raw
            )

    if abs(blended_error) < STRAIGHT_ERROR_DEADBAND:
        blended_error = 0.0
        fs["blend_value"] = 0.0

    error_ratio = min(abs(blended_error) / half_width, 1.0)

    curve_ratio = min(abs(curve_error) / half_width, 1.0)

    k = k_min + (k_max - k_min) * (curve_ratio ** 0.5)
    k *= (1.0 + 1.0 * error_ratio)
    base_angle = blended_error * k

    if curve_ratio < STRAIGHT_CURVE_RATIO and error_ratio < 0.12:
        speed_factor = (
            1.0
            + 0.55 * (speed / max_speed)
            * (1 - 0.5 * (error_ratio ** 2))
        )
        aggressive_factor = 1.0
    else:
        speed_factor = (
            1.0
            + 1.2 * (speed / max_speed)
            * (1 - 0.5 * (error_ratio ** 2))
        )
        aggressive_factor = (
            1.0 + 5.0 * (error_ratio ** 2)
            if error_ratio > 0.1
            else 1.0
        )

    normal_angle = base_angle * speed_factor * aggressive_factor

    if curve_ratio < STRAIGHT_CURVE_RATIO:
        normal_angle = np.clip(normal_angle, -STRAIGHT_ANGLE_LIMIT, STRAIGHT_ANGLE_LIMIT)
    else:
        normal_angle = np.clip(normal_angle, -15, 15)

    if blended_error >= 35:
        angle = 25
    elif blended_error <= -35:
        angle = -25
    elif 25 <= blended_error < 35:
        scale = (blended_error - 25) / 20
        angle = 18 + 12 * scale
    elif -35 < blended_error <= -25:
        scale = (-blended_error - 25) / 20
        angle = -(18 + 12 * scale)
    elif 15 <= blended_error < 25:
        scale = (blended_error - 20) / 20
        angle = 10 + 10 * scale
    elif -25 < blended_error <= -15:
        scale = (-blended_error - 20) / 20
        angle = -(10 + 10 * scale)
    else:
        angle = normal_angle

    angle = np.clip(angle, -25, 25)

    if show_debug and ENABLE_GUI:
        debug = np.zeros((height, width, 3), dtype=np.uint8)
        debug[gray > 0] = (90, 90, 90)

        for cx, y in green_line_points:
            debug[y, cx] = (0, 255, 0)

        cv2.line(debug, (lane_cx, 0), (lane_cx, height), (0, 0, 255), 2)

        target_cx_draw = width // 2 + bias_pixel
        cv2.line(
            debug,
            (target_cx_draw, 0),
            (target_cx_draw, height),
            (255, 0, 255),
            1
        )

        if far_points:
            far_pts = np.array(
                [[cx, y] for cx, y in far_points],
                np.int32
            ).reshape((-1, 1, 2))
            cv2.polylines(debug, [far_pts], False, (255, 0, 0), 2)

        if near_points:
            near_cx = int(np.mean([cx for cx, y in near_points]))
            cv2.line(
                debug,
                (near_cx, height // 2),
                (near_cx, height),
                (0, 255, 255),
                2
            )

        if bottom_cx >= 0:
            cv2.circle(
                debug,
                (bottom_cx, height - 5),
                5,
                (0, 200, 255),
                -1
            )
            cv2.putText(
                debug,
                f"bottom={bottom_cx} raw={bottom_cx_raw} W/2={width//2}",
                (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 255),
                1
            )

        if lane_width_estimator and lane_width_estimator.current_width:
            base_w = lane_width_estimator.base_width or 0
            cur_w = lane_width_estimator.current_width
            ratio = cur_w / base_w if base_w > 0 else 0

            cv2.putText(
                debug,
                f"BaseW: {base_w:.0f} CurW: {cur_w:.0f} "
                f"r: {ratio:.2f} Bias: {bias_pixel}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            if wide_change:
                cv2.putText(
                    debug,
                    "WIDE ROAD",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1
                )

        cv2.putText(
            debug,
            f"Heading: {lane_heading_deg:+.1f} deg",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 200, 0),
            1
        )

        cv2.putText(
            debug,
            f"Near: {near_error:+.1f} Far: {far_error:+.1f}",
            (10, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.putText(
            debug,
            f"Blend: {blended_error:+.1f} Curve: {curve_ratio:.2f} "
            f"Angle: {angle:+.1f}",
            (10, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.putText(
            debug,
            "RAW-TURN" if bypass_error_filter else "FILTER-STRAIGHT",
            (10, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.imshow("Lane Debug", debug)

    return (
        angle,
        blended_error,
        k,
        w_near,
        w_far,
        curve_ratio,
        bottom_cx,
        top_cx,
        lane_heading_deg
    )


# ==============================================================
# V3.10 HYBRID NORMAL STEERING
# ==============================================================
def calculate_hybrid_normal_angle(
        blended_error,
        curve_ratio,
        speed,
        max_speed,
        width,
        k_min=HYBRID_K_MIN,
        k_max=HYBRID_K_MAX):
    half_width = max(width // 2, 1)
    error_ratio = min(abs(float(blended_error)) / half_width, 1.0)

    adaptive_mode = (
        curve_ratio >= HYBRID_CURVE_RATIO
        or error_ratio >= HYBRID_ERROR_RATIO
    )

    if not adaptive_mode:
        return None, error_ratio, 0.0, False

    k = k_min + (k_max - k_min) * (curve_ratio ** 0.5)
    k *= (1.0 + error_ratio)
    base_angle = float(blended_error) * k

    speed_ratio = speed / max(max_speed, 1.0)
    speed_factor = (
        1.0
        + HYBRID_CURVE_SPEED_FACTOR * speed_ratio
        * (1.0 - 0.5 * (error_ratio ** 2))
    )

    aggressive_factor = (
        1.0 + HYBRID_AGGRESSIVE_GAIN * (error_ratio ** 2)
        if error_ratio > 0.10
        else 1.0
    )

    angle = base_angle * speed_factor * aggressive_factor

    if blended_error >= 35:
        angle = 25.0
    elif blended_error <= -35:
        angle = -25.0
    elif 25 <= blended_error < 35:
        scale = (blended_error - 25.0) / 20.0
        angle = 18.0 + 12.0 * scale
    elif -35 < blended_error <= -25:
        scale = (-blended_error - 25.0) / 20.0
        angle = -(18.0 + 12.0 * scale)
    elif 15 <= blended_error < 25:
        scale = (blended_error - 20.0) / 20.0
        angle = 10.0 + 10.0 * scale
    elif -25 < blended_error <= -15:
        scale = (-blended_error - 20.0) / 20.0
        angle = -(10.0 + 10.0 * scale)

    return float(np.clip(angle, -HYBRID_MAX_ANGLE, HYBRID_MAX_ANGLE)), error_ratio, k, True


# ==============================================================
# MAIN LOOP
# ==============================================================
if __name__ == "__main__":
    max_speed = 25
    min_speed = 22

    lane_width_estimator = LaneWidthEstimator()
    voter = SignVoter(min_frames=MIN_FRAMES, max_gap=MAX_GAP)
    navigator = SlowStopTurnNavigator(voter=voter)
    intersection_detector = IntersectionDetector()
    steering_pid = SteeringPID()

    intersection_id = 0
    prev_state = 0
    last_speed = LastSpeed()
    prev_nav_state = 'normal'
    frame_idx = 0

    try:
        while True:
            frame_idx += 1
            state_raw = GetStatus()
            raw_image = GetRaw()

            seg_mask = get_yolo_road_mask(raw_image)
            H, W = seg_mask.shape

            topo = get_topology(seg_mask, W, H)
            raw_state = topology_to_state(topo)

            if raw_state == 0:
                current_state = 0
            else:
                if prev_state == 0:
                    intersection_id += 1
                current_state = raw_state
            prev_state = current_state

            sign_classes = sign_detect(raw_image)

            if navigator.state == 'turning':
                pass
            else:
                voter.update(sign_classes)
            committed = voter.get_committed()
            commit_age = voter.get_commit_age()

            lane_width_estimator.measure_width(seg_mask, [])
            base_w = lane_width_estimator.base_width
            cur_w = lane_width_estimator.current_width
            mask_trigger = intersection_detector.update(base_w, cur_w)

            bias_scale = navigator.bias_scale() if navigator.state == 'turning' else 1.0
            bias_pixel_pre = decide_bias(committed, topo, base_w, scale=bias_scale)

            (_, blended_error_pre, _, _, _, _,
             bottom_cx_pre, top_cx_pre, heading_pre) = calculate_steering_angle(
                seg_mask,
                max_speed=max_speed,
                speed=last_speed.value,
                lane_width_estimator=lane_width_estimator,
                bias_pixel=bias_pixel_pre,
                show_debug=False
            )

            effective_trigger = (mask_trigger and navigator.state == 'slow_down')

            nav_speed, nav_angle, is_active = navigator.update(
                sign_classes, committed, topo, effective_trigger,
                blended_error_pre, heading_pre,
                bottom_cx_pre, W, base_w, cur_w
            )

            if (navigator.state == 'cooldown'
                    and prev_nav_state != 'cooldown'):
                print("[MAIN] enter cooldown — reset lane baseline")
                lane_width_estimator.reset_baseline()
            prev_nav_state = navigator.state

            if navigator.state in ('stopping', 'turning', 'cooldown'):
                intersection_detector.reset()

            # ==================================================
            # CONTROL
            # ==================================================
            if navigator.state != 'normal':
                steering_pid.reset()

            bias_pixel = 0
            blended_error = 0.0
            angle = 0.0
            speed = last_speed.value
            bottom_cx = -1
            top_cx = -1
            lane_heading_deg = 0.0
            ff_used = 0.0
            angle_raw = 0.0

            if is_active is True:
                final_speed = nav_speed
                final_angle = nav_angle if nav_angle is not None else 0
                speed = final_speed
                angle = final_angle
                last_speed.value = final_speed
                bias_pixel = bias_pixel_pre
                blended_error = blended_error_pre
                bottom_cx = bottom_cx_pre
                top_cx = top_cx_pre
                lane_heading_deg = heading_pre
                ff_used = navigator.get_ff_used()
                angle_raw = navigator.get_angle_raw()
                AVControl(final_speed, final_angle)

                print(f"[NAV] state={navigator.state} cnt={navigator._cnt} "
                      f"commit={committed} turn_dir={navigator._turn_dir} "
                      f"bias={bias_pixel} "
                      f"err={blended_error_pre:+.1f} "
                      f"ff={ff_used:+.1f} raw={angle_raw:+.2f} "
                      f"ang={final_angle:+.2f} "
                      f"bc={bottom_cx_pre} W={W} "
                      f"new_lane={navigator.get_new_lane_cnt()} "
                      f"cross={navigator._crossed_zero} "
                      f"spd={final_speed}")

                if ENABLE_GUI:
                    calculate_steering_angle(
                        seg_mask,
                        speed=final_speed,
                        max_speed=max_speed,
                        lane_width_estimator=lane_width_estimator,
                        bias_pixel=bias_pixel_pre,
                        show_debug=True,
                        update_filter=False
                    )

            elif is_active == 'slow':
                (angle, blended_error, _k, _wn, _wf, _cr,
                 bottom_cx, top_cx, lane_heading_deg) = calculate_steering_angle(
                    seg_mask,
                    max_speed=max_speed,
                    speed=SLOW_SPEED,
                    lane_width_estimator=lane_width_estimator,
                    bias_pixel=0,
                    show_debug=True
                )
                speed = SLOW_SPEED
                last_speed.value = SLOW_SPEED
                AVControl(SLOW_SPEED, angle)
                print(f"[SLOW] commit={committed} spd={SLOW_SPEED} "
                      f"ang={angle:+.2f} err={blended_error:.1f} "
                      f"bc={bottom_cx} "
                      f"topo=L{int(topo['left'])}S{int(topo['straight'])}R{int(topo['right'])} "
                      f"top_cov={topo.get('top_half_coverage', 1.0):.3f} "
                      f"top_empty={int(topo.get('top_half_empty', False))} "
                      f"mask_trig={int(mask_trigger)}")

            else:
                (angle, blended_error, _k, _wn, _wf, curve_ratio,
                 bottom_cx, top_cx, lane_heading_deg) = calculate_steering_angle(
                    seg_mask,
                    max_speed=max_speed,
                    speed=last_speed.value,
                    lane_width_estimator=lane_width_estimator,
                    bias_pixel=0,
                    show_debug=True
                )

                force_straight = (
                    navigator.state == 'cooldown'
                    and navigator._cnt <= COOLDOWN_FORCE_STRAIGHT_FRAMES
                )
                if force_straight:
                    angle = 0.0

                target_speed = max(SLOW_SPEED, max_speed * (1 - 0.7 * curve_ratio))
                if navigator.state == 'cooldown':
                    target_speed = min(target_speed, COOLDOWN_SPEED_CAP)
                speed = 0.6 * last_speed.value + 0.4 * target_speed
                last_speed.value = speed

                if navigator.state == 'normal':
                    hybrid_angle, error_ratio, hybrid_k, hybrid_active = (
                        calculate_hybrid_normal_angle(
                            blended_error,
                            curve_ratio,
                            speed,
                            max_speed,
                            W
                        )
                    )

                    if hybrid_active:
                        steering_pid.reset()
                        angle = hybrid_angle
                    else:
                        angle = steering_pid.step(blended_error)
                else:
                    steering_pid.reset()
                    error_ratio = min(
                        abs(blended_error) / max(W // 2, 1), 1.0
                    )
                    hybrid_k = 0.0
                    hybrid_active = False

                AVControl(speed, angle)

                ratio = (cur_w / base_w) if (base_w and base_w > 0) else 0
                tag = "COOLDOWN-FORCE" if force_straight else navigator.state.upper()
                if navigator.state == 'normal':
                    if hybrid_active:
                        pid_text = (
                            f" HYBRID=CURVE"
                            f" ER={error_ratio:.2f}"
                            f" K={hybrid_k:.3f}"
                        )
                    else:
                        pid_text = (
                            f" HYBRID=PID"
                            f" P={steering_pid.last_p:+.2f}"
                            f" I={steering_pid.last_i:+.2f}"
                            f" D={steering_pid.last_d:+.2f}"
                        )
                else:
                    pid_text = ""

                print(
                    f"[{tag}] raw={raw_state} id={intersection_id} | "
                    f"topo=L{int(topo['left'])}S{int(topo['straight'])}R{int(topo['right'])} | "
                    f"commit={committed} | "
                    f"BaseW={base_w or 0:.0f} CurW={cur_w or 0:.0f} r={ratio:.2f} | "
                    f"err={blended_error:+.1f} ang={angle:+.2f} spd={speed:.1f} "
                    f"bc={bottom_cx}{pid_text}"
                )

            # CSV LOG
            ratio_log = (cur_w / base_w) if (base_w and base_w > 0) else 0
            log_writer.writerow([
                f"{time.time():.3f}",
                frame_idx,
                navigator.state,
                raw_state,
                committed if committed else "",
                commit_age,
                navigator._turn_dir,
                navigator._cnt if navigator.state in ('turning', 'cooldown', 'stopping') else 0,
                bias_pixel,
                f"{bias_scale:.2f}",
                base_w if base_w else 0,
                cur_w if cur_w else 0,
                f"{ratio_log:.3f}",
                f"{blended_error:+.3f}",
                f"{angle:+.3f}",
                f"{speed:.2f}",
                bottom_cx,
                top_cx,
                f"{lane_heading_deg:+.2f}",
                f"{ff_used:+.2f}",
                f"{angle_raw:+.3f}",
                navigator.get_new_lane_cnt() if navigator.state == 'turning' else 0,
                f"{curve_ratio:.3f}" if 'curve_ratio' in dir() else "0.000",
                f"{_k:.4f}" if is_active == 'slow' or (is_active is not True) else "0.0000",
                f"{steering_pid.last_p:+.4f}",
                f"{steering_pid.last_i:+.4f}",
                f"{steering_pid.last_d:+.4f}",
                f"{steering_pid.last_raw:+.4f}",
                "|".join(sign_classes),
                int(mask_trigger),
                int(navigator._crossed_zero),
                navigator._stable_cnt if navigator.state == 'turning' else 0,
                navigator._initial_sign,
                navigator.get_exit_reason(),
            ])

            if frame_idx % LOG_FLUSH_EVERY == 0:
                log_file.flush()

            if ENABLE_GUI:
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

// ============================================================
// config/RobotConfig.h
//
// Central config for the UCR-2026 car application.
//
// STRUCTURE:
//   SECTION A — active in the current build (main.cpp / ucrv6)
//   SECTION B — turn/sign integration (now WIRED IN main.cpp)
//   SECTION C — removed (failed gate iterations from Log 13)
// ============================================================
#pragma once

namespace ucr {
namespace config {

// ============================================================
// SECTION A — ACTIVE IN CURRENT BUILD
// ============================================================

// --------------------------------------------------------
// Model paths
// --------------------------------------------------------
inline constexpr char ROAD_ENGINE_PATH[] =
    "/home/car4/Downloads/road/kga2.engine";
inline constexpr char SIGN_ENGINE_PATH[] =
    "/home/car4/Downloads/sign/huy_sign.engine";

// --------------------------------------------------------
// Hardware limits
// --------------------------------------------------------
inline constexpr float MAX_SPEED_CMS = 20.0f;
inline constexpr float MIN_SPEED_CMS = 10.0f;

// --------------------------------------------------------
// Servo calibration (measured empirically, Log 12)
//
//   servo_cmd > 0 (left):   real =  0.96 * servo_cmd − 6.2
//   servo_cmd < 0 (right):  real =  0.71 * servo_cmd + 2.5
//
// Conventions:
//   - PID output (out.angle):     positive = RIGHT
//   - Real angle:                 positive = LEFT
//   - Servo command (SetAngle):   positive = LEFT
// --------------------------------------------------------
inline constexpr float SERVO_LEFT_SLOPE    = 0.96f;
inline constexpr float SERVO_LEFT_OFFSET   = -6.2f;
inline constexpr float SERVO_RIGHT_SLOPE   = 0.71f;
inline constexpr float SERVO_RIGHT_OFFSET  = 2.5f;

inline constexpr float MAX_REAL_ANGLE_DEG = 30.0f;
inline constexpr int   MAX_SERVO_CMD      = 55;

inline float RealAngleToServo(float real_deg) {
    if (real_deg > 0.0f)
        return (real_deg - SERVO_LEFT_OFFSET) / SERVO_LEFT_SLOPE;
    if (real_deg < 0.0f)
        return (real_deg - SERVO_RIGHT_OFFSET) / SERVO_RIGHT_SLOPE;
    return 0.0f;
}

inline float ServoToRealAngle(float servo_cmd) {
    if (servo_cmd > 0.0f)
        return SERVO_LEFT_SLOPE * servo_cmd + SERVO_LEFT_OFFSET;
    if (servo_cmd < 0.0f)
        return SERVO_RIGHT_SLOPE * servo_cmd + SERVO_RIGHT_OFFSET;
    return 0.0f;
}

// --------------------------------------------------------
// Control loop
// --------------------------------------------------------
inline constexpr int   CONTROL_PERIOD_MS = 100;   // 10 Hz
inline constexpr float MAX_ACCEL_CMS2    = 20.0f;

// --------------------------------------------------------
// Road model
// --------------------------------------------------------
inline constexpr int   ROAD_DA_CHANNEL  = 1;
inline constexpr float ROAD_CONF_MARGIN = 0.02f;

// --------------------------------------------------------
// Lane follower tuning
// --------------------------------------------------------
inline constexpr float  CURVE_SCALE        = 150.0f;
inline constexpr double POST_PID_GAIN      = 2.2;   // see E52 in README
inline constexpr int    MIN_ROWS           = 5;
inline constexpr double COVERAGE_THRESHOLD = 0.05;

// --------------------------------------------------------
// UI
// --------------------------------------------------------
inline constexpr bool SHOW_UI = true;

// ============================================================
// SECTION B — TURN / SIGN INTEGRATION (wired in main.cpp)
// ============================================================

// --------------------------------------------------------
// Sign model
// --------------------------------------------------------
inline constexpr float SIGN_CONF_THRESH = 0.5f;
inline constexpr float SIGN_IOU_THRESH  = 0.45f;
inline constexpr int   SIGN_INPUT_SIZE  = 416;

namespace sign_class {
    inline constexpr int CONSTRUCTION_AHEAD = 0;
    inline constexpr int NO_LEFT            = 1;
    inline constexpr int NO_RIGHT           = 2;
    inline constexpr int PARKING            = 3;
    inline constexpr int ROUNDABOUT         = 4;
    inline constexpr int TRAFFIC_LIGHT      = 5;
    inline constexpr int TURN_LEFT          = 6;
    inline constexpr int TURN_RIGHT         = 7;
}
inline constexpr const char* SignClassName(int id) {
    switch (id) {
        case 0: return "construct";
        case 1: return "no_left";
        case 2: return "no_right";
        case 3: return "parking";
        case 4: return "roundabout";
        case 5: return "traffic_light";
        case 6: return "turn_left";
        case 7: return "turn_right";
        default: return "-";
    }
}

// --------------------------------------------------------
// Sign detect + voter timers (units noted per line)
// --------------------------------------------------------
inline constexpr int SIGN_DETECT_EVERY             = 2;    // control-tick units
inline constexpr int SIGN_VOTER_ACTIVATE_FRAMES    = 2;    // detect-update units
inline constexpr int SIGN_VOTER_TIMEOUT_FRAMES     = 60;   // control-tick units
inline constexpr int SIGN_HOLD_FRAMES              = 18;   // control-tick units
inline constexpr int SIGN_SNAPSHOT_HOLD_FRAMES     = 3;    // control-tick units
inline constexpr int TOPO_CONFIRM_FRAMES           = 3;    // detect-update units
inline constexpr int RESET_DEBOUNCE_FRAMES         = 90;   // control-tick units

// --------------------------------------------------------
// Turn SM — force-turn and phase parameters
// --------------------------------------------------------
inline constexpr float FORCE_ANGLE               = 22.0f;
inline constexpr float HEADING_RESET_DEG         = 56.0f;
inline constexpr float FORCE_APPROACH_SPEED      = 8.0f;
inline constexpr float FORCE_TURNING_SPEED       = 5.0f;
inline constexpr float FORCE_EXIT_SPEED          = 12.0f;
inline constexpr float FORCE_TURN_PHASE_DEG      = 15.0f;
inline constexpr float FORCE_EXIT_PHASE_DEG      = 50.0f;
inline constexpr float SLOW_SPEED_CMS            = 10.0f;
inline constexpr int   MAX_FORCE_TICKS           = 50;   // safety timeout ~5s

// --------------------------------------------------------
// Opening detection (V21 port — replaces area-spike gate)
// --------------------------------------------------------
inline constexpr int   OPEN_THRESH             = 15;    // px deviation threshold
inline constexpr int   OPEN_MIN_HEIGHT         = 10;    // min vertical extent (px)
inline constexpr int   OPEN_SMOOTH_WIN         = 5;     // moving-avg window for edges
inline constexpr int   OPEN_BASE_WIN           = 50;    // rolling-median window for baselines
inline constexpr float TURN_OPENING_PROXIMITY  = 0.55f; // y_mid > this*height → arm
inline constexpr int   TURN_ARM_CONFIRM        = 2;     // consecutive ticks to confirm

// --------------------------------------------------------
// Bezier path planning (V21 port — drives steering during turn)
// --------------------------------------------------------
inline constexpr int   BEZIER_CORNER_RADIUS    = 20;
inline constexpr int   BEZIER_MIN_TURN_DIST    = 4;
inline constexpr int   BEZIER_EXTEND_LEN       = 80;
inline constexpr float BEZIER_LOOKAHEAD_FRAC   = 0.85f;
inline constexpr float BEZIER_STEER_GAIN       = 0.30f;

// --------------------------------------------------------
// Heading (BNO055) transform
// --------------------------------------------------------
inline constexpr float HEADING_SIGN       = 1.0f;
inline constexpr float HEADING_OFFSET_DEG = 0.0f;

// --------------------------------------------------------
// PID rate limiter applied at the final handoff (v20 ↔ turn)
// --------------------------------------------------------
inline constexpr float PID_RATE_LIMIT_DEG = 10.0f;   // per control tick

// ============================================================
// SECTION C — REMOVED (do not re-add)
//
//   TURN_CONDITION_Y_LO_RATIO  (0.69)  — bottom-band coverage gate
//   TURN_CONDITION_COVERAGE    (0.78)  — bottom-band coverage gate
//   TURN_OPENING_MARGIN        (0.08)  — far-band vs near-band gate
//   TURN_NEAR_MIN_COVERAGE     (0.55)  — far-band vs near-band gate
//   TURN_AREA_RATIO            (1.30)  — area-spike gate
//   TURN_GATE_PERSIST          (2)     — area-spike gate
// ============================================================

}  // namespace config
}  // namespace ucr
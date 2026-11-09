// ============================================================
// config/RobotConfig.h
//
// Central config for the UCR-2026 car application.
//
// STRUCTURE:
//   SECTION A — active in the current build (main.cpp / ucrv5 binary)
//               Everything in this section is read by main.cpp through
//               v20::LaneFollower, RoadSegmenter, or directly.
//
//   SECTION B — reserved for turn/sign integration
//               Read by nothing currently. Wire these into SignDetector,
//               SignVoter, and TurnStateMachine::TurnConfig when the
//               turn/sign stack is re-integrated into main.cpp.
//
//   SECTION C — removed
//               Failed gate iterations from Log 13. Listed for reference
//               so they don't get re-invented.
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

// --------------------------------------------------------
// Hardware limits
// --------------------------------------------------------
// MAX_SPEED_CMS was raised 15→20 in Log 12 after confirming the round-3
// track allows ~10 cm/s turning without sliding. Do not exceed 25 until
// the track shows a need.
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
//
// Left/right gain differ by ~35%. Mechanical deadband is asymmetric
// (≈6.5° servo on the left, ≈3.5° on the right). See README
// "Servo calibration" and Log 12 E52/E53 for the two latent issues.
// --------------------------------------------------------
inline constexpr float SERVO_LEFT_SLOPE    = 0.96f;
inline constexpr float SERVO_LEFT_OFFSET   = -6.2f;
inline constexpr float SERVO_RIGHT_SLOPE   = 0.71f;
inline constexpr float SERVO_RIGHT_OFFSET  = 2.5f;

// Symmetric real-angle bound. At 30° real:
//   left  servo ≈ +37.7
//   right servo ≈ −45.8
// Both safely inside the ±60° servo range. Raise to 33 only if a track
// demands more right-turn capability AND E52 is resolved.
inline constexpr float MAX_REAL_ANGLE_DEG = 30.0f;

// Hard safety clamp on the raw servo command. Independent of the
// real-angle bound — catches calibration mistakes and saturation drift.
inline constexpr int MAX_SERVO_CMD = 55;

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
// PID step period. Every rate limit / EMA inside LaneFollowerV20 is
// per-step; changing this value changes their real-world meaning.
inline constexpr int   CONTROL_PERIOD_MS = 100;   // 10 Hz
inline constexpr float MAX_ACCEL_CMS2   = 20.0f;  // soft-start ramp only

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
// SECTION B — RESERVED FOR TURN/SIGN INTEGRATION
//
// Not read by the current main.cpp. On re-integration:
//   - SIGN_ENGINE_PATH     → SignDetector::Init
//   - SIGN_CONF/IOU/INPUT  → SignDetector::Config
//   - SIGN_DETECT_EVERY    → when Detect() is called
//   - SIGN_VOTER_*         → SignVoter constructor
//   - TOPO_CONFIRM_FRAMES  → TopoLatch constructor
//   - RESET_DEBOUNCE_FRAMES→ TurnStateMachine::TurnConfig
//   - FORCE_*/SLOW_*       → TurnStateMachine::TurnConfig
//   - TURN_AREA_RATIO/GATE_PERSIST → TurnStateMachine::TurnConfig
//   - HEADING_*            → BNO055 → TurnStateMachine::Update()
// ============================================================

// --------------------------------------------------------
// Sign model
// --------------------------------------------------------
inline constexpr char SIGN_ENGINE_PATH[] =
    "/home/car4/Downloads/sign/huy_sign.engine";

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
// Sign detect + voter timers
//
// SIGN_DETECT_EVERY = 2: tested at 4 (no observable difference at ~6 Hz
// loop rate); 2 is kept.
//
// Timer units are noted per constant. If FPS changes significantly,
// recompute ticks = seconds × actual_Hz.
// --------------------------------------------------------
inline constexpr int SIGN_DETECT_EVERY             = 2;    // control-tick units
inline constexpr int SIGN_VOTER_ACTIVATE_FRAMES    = 2;    // detect-update units (~1.3 s at 1.5 Hz)
inline constexpr int SIGN_VOTER_TIMEOUT_FRAMES     = 60;   // control-tick units (~10 s at 6 Hz)
inline constexpr int SIGN_HOLD_FRAMES              = 18;   // control-tick units (~3 s at 6 Hz)
inline constexpr int SIGN_SNAPSHOT_HOLD_FRAMES     = 3;    // control-tick units (~0.5 s at 6 Hz)
inline constexpr int TOPO_CONFIRM_FRAMES           = 3;    // detect-update units
inline constexpr int RESET_DEBOUNCE_FRAMES         = 90;   // control-tick units (~15 s at 6 Hz)

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

// --------------------------------------------------------
// Turn SM — arming gate (area-spike, Log 13 iteration 3)
//
// Tested in Log 13: never fired on the round-3 map, because the road
// segmentation model does not mask far enough ahead to expose the
// junction widening. Kept here as a reference; replaced by a working
// gate in a future session.
// --------------------------------------------------------
inline constexpr float TURN_AREA_RATIO   = 1.30f;
inline constexpr int   TURN_GATE_PERSIST = 2;

// --------------------------------------------------------
// Heading (BNO055) transform
//
// BNO055 eulerDeg.x is the fused heading. HEADING_SIGN / _OFFSET
// transform it into TurnStateMachine's convention (+ = right).
// Defaults are identity. Verify empirically on-car.
// --------------------------------------------------------
inline constexpr float HEADING_SIGN       = 1.0f;
inline constexpr float HEADING_OFFSET_DEG = 0.0f;

// ============================================================
// SECTION C — REMOVED (do not re-add)
//
// Failed gate iterations from Log 13. Deleted from both this file and
// TurnStateMachine::TurnConfig. Listed here so they don't get re-invented.
//
//   TURN_CONDITION_Y_LO_RATIO  (0.69)  — bottom-band coverage gate
//   TURN_CONDITION_COVERAGE    (0.78)  — bottom-band coverage gate
//   TURN_OPENING_MARGIN        (0.08)  — far-band vs near-band gate
//   TURN_NEAR_MIN_COVERAGE     (0.55)  — far-band vs near-band gate
//   OPEN_THRESH / OPEN_MIN_HEIGHT / OPEN_SMOOTH_WIN / OPEN_BASE_WIN /
//   TURN_OPENING_PROXIMITY            — V21 opening gate draft
//                                        (only ever lived in the
//                                        never-compiled LaneFollowerV21.h)
// ============================================================

}  // namespace config
}  // namespace ucr


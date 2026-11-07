// ============================================================
// config/RobotConfig.h
//
// Central config for the UCR-2026 car application.
// All tunable constants live here. Controller internals
// (LaneFollowerV20::Config) are still configured at the call
// site so this file stays focused on hardware/system values.
// ============================================================
#pragma once

namespace ucr {
namespace config {

// --------------------------------------------------------
// Model paths
// --------------------------------------------------------
inline constexpr char ROAD_ENGINE_PATH[] = "/home/car4/Downloads/road/kga2.engine";
inline constexpr char SIGN_ENGINE_PATH[] = "/home/car4/Downloads/sign/huy_sign.engine";

// --------------------------------------------------------
// Hardware limits
// --------------------------------------------------------
// Raised from 15 cm/s. User confirmed 10 cm/s turns without sliding,
// so 15 → 20 is safe headroom. Do not exceed 25 until the track
// shows a need.
inline constexpr float MAX_SPEED_CMS = 20.0f;
inline constexpr float MIN_SPEED_CMS = 10.0f;

// --------------------------------------------------------
// Servo calibration (measured empirically)
//
// Real angle = SLOPE * servo_cmd + OFFSET, per branch:
//   servo_cmd > 0  (left):   real =  0.96 * cmd - 6.2
//   servo_cmd < 0  (right):  real =  0.71 * cmd + 2.5
//
// Conventions used by this project:
//   - PID output (out.angle): positive = RIGHT turn
//   - Servo command (SetAngle): positive = LEFT turn
//   - Real angle (physical): positive = LEFT turn (matches servo)
//
// The helper functions below convert between "desired real angle"
// and "servo command", inverting the measured curves.
// --------------------------------------------------------
inline constexpr float SERVO_LEFT_SLOPE    = 0.96f;
inline constexpr float SERVO_LEFT_OFFSET   = -6.2f;
inline constexpr float SERVO_RIGHT_SLOPE   = 0.71f;
inline constexpr float SERVO_RIGHT_OFFSET  = 2.5f;

// Symmetric real-angle bound. Servo commands are computed to hit
// exactly this on both sides, so the car turns equally left/right.
// At 30° real: left servo ≈ 37.7, right servo ≈ −45.8. Both are
// safely inside the ±60° servo range. Raise to 33 if you need
// to match the previous max right-turn capability.
inline constexpr float MAX_REAL_ANGLE_DEG = 30.0f;

// Hard safety clamp on the raw servo command. Independent of the
// real-angle bound — catches calibration errors and saturated maps.
inline constexpr int MAX_SERVO_CMD = 55;

// Convert: desired real angle (deg, + = left) → servo command.
inline float RealAngleToServo(float real_deg) {
    if (real_deg > 0.0f)
        return (real_deg - SERVO_LEFT_OFFSET) / SERVO_LEFT_SLOPE;
    if (real_deg < 0.0f)
        return (real_deg - SERVO_RIGHT_OFFSET) / SERVO_RIGHT_SLOPE;
    return 0.0f;
}

// Inverse: servo command → predicted real angle (deg, + = left).
// Use for UI diagnostics and post-run analysis.
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
// PID step period. All LaneFollowerV20 rate limits and EMA
// alphas are per-step, so this value determines their
// real-world meaning. 100 ms = 10 Hz.
inline constexpr int   CONTROL_PERIOD_MS = 100;

// Speed ramp applied on motor commands (soft-start).
// Decel is not ramped — the driver handles it.
inline constexpr float MAX_ACCEL_CMS2 = 20.0f;

// --------------------------------------------------------
// Road model
// --------------------------------------------------------
inline constexpr int   ROAD_DA_CHANNEL  = 1;
inline constexpr float ROAD_CONF_MARGIN = 0.02f;

// --------------------------------------------------------
// Lane follower tuning (from test_run.cpp)
// --------------------------------------------------------
inline constexpr float  CURVE_SCALE        = 150.0f;
inline constexpr double POST_PID_GAIN      = 2.2;
inline constexpr int    MIN_ROWS           = 5;
inline constexpr double COVERAGE_THRESHOLD = 0.05;

// ---- Sign detector ----

static constexpr float SIGN_CONF_THRESH        = 0.60f;
static constexpr float SIGN_IOU_THRESH         = 0.45f;
static constexpr int   SIGN_INPUT_SIZE         = 416;
static constexpr int   SIGN_DETECT_EVERY       = 2;    // every 2N control ticks

// ---- Sign voter (v28 values) ----
static constexpr int SIGN_VOTER_MIN_FRAMES      = 4;
static constexpr int SIGN_VOTER_MAX_GAP         = 3;
static constexpr int SIGN_VOTER_TIMEOUT         = 300;
static constexpr int SIGN_VOTER_BLACKLIST_FRAMES = 300;

// ---- Turn state machine (v28 values; UNVERIFIED for TwinLiteNet resolution) ----
static constexpr float TURN_TARGET_YAW            = 72.0f;
static constexpr float TURN_SHARP_ANGLE           = 22.0f;
static constexpr float TURN_BRAKING_SPEED         = 8.0f;
static constexpr float TURN_TURNING_SPEED         = 8.0f;
static constexpr float TURN_BRAKE_MULT            = 1.60f;
static constexpr float TURN_TURN_MULT             = 2.00f;
static constexpr float TURN_BASELINE_FLOOR        = 2280.0f;  // tuned on YOLO-320x192 output
static constexpr float TURN_BASELINE_ALPHA        = 0.97f;
static constexpr float TURN_TAPER_START           = 0.85f;
static constexpr float TURN_BAIL_FRAC_OF_BASELINE = 1.05f;
static constexpr float TURN_PEAK_DROP_PX          = 200.0f;

// ---- Heading source (BNO055) ----
// eulerDeg.x is the BNO055's fused yaw. Test file (`./main` with no args) prints it raw.
// Set HEADING_SIGN = -1.0f if a clockwise bench turn decreases the value.
static constexpr float HEADING_SIGN       = 1.0f;
static constexpr float HEADING_OFFSET_DEG = 0.0f;

// --------------------------------------------------------
// UI
// --------------------------------------------------------
inline constexpr bool SHOW_UI = true;

}  // namespace config
}  // namespace ucr
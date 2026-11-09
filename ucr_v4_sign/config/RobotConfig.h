#pragma once
#include <string>

namespace ucr { namespace config {

// ============================================================
// SYSTEM
// ============================================================
constexpr int   CONTROL_PERIOD_MS   = 100;      // 10 Hz
constexpr float MAX_SPEED_CMS       = 15.0f;
constexpr float MIN_SPEED_CMS       = 5.0f;
constexpr float MAX_ACCEL_CMS2      = 20.0f;
constexpr float MAX_STEER_DEG       = 50.0f;
constexpr bool  INVERT_STEER        = false;    

// ============================================================
// MODEL PATHS
// ============================================================
constexpr const char* ROAD_ENGINE_PATH = "/home/car4/Downloads/road/kga2.engine";
constexpr const char* SIGN_ENGINE_PATH = "/home/car4/Downloads/sign/huy_sign.engine";

// ============================================================
// ROAD FOLLOWER (v20)
// ============================================================
constexpr int   MIN_ROWS             = 5;
constexpr float COVERAGE_THRESHOLD   = 0.05f;
constexpr float CURVE_SCALE          = 150.0f;

// ============================================================
// STEERING CALIBRATION (Từ dữ liệu hồi quy)
// ============================================================
constexpr float STEER_LEFT_M  = 0.96f;
constexpr float STEER_LEFT_B  = -6.2f;
constexpr float STEER_RIGHT_M = 0.71f;
constexpr float STEER_RIGHT_B = 2.5f;

// ============================================================
// SIGN & MISSION
// ============================================================
constexpr int   VOTER_ACTIVATE_FRAMES      = 5;
constexpr int   VOTER_TIMEOUT_FRAMES       = 100;
constexpr float SIGN_CONFIDENCE_THRESHOLD  = 0.25f;
constexpr float SIGN_NMS_THRESHOLD         = 0.45f;

// Force turn
constexpr float FORCE_ANGLE                = 25.0f;
constexpr float FORCE_APPROACH_SPEED       = 8.0f;
constexpr float FORCE_TURNING_SPEED        = 5.0f;
constexpr float FORCE_EXIT_SPEED           = 12.0f;
constexpr float FORCE_TURN_PHASE_DEG       = 15.0f;
constexpr float FORCE_EXIT_PHASE_DEG       = 50.0f;
constexpr float HEADING_RESET_DEG          = 56.0f;

// Điều kiện kích hoạt force turn
constexpr float TURN_CONDITION_Y_LO_RATIO  = 0.69f;
constexpr float TURN_CONDITION_COVERAGE    = 0.78f;

} } // namespace
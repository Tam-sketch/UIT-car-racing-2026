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
constexpr bool  INVERT_STEER        = false;    // Đổi thành true nếu servo lắp ngược

// ============================================================
// MODEL PATHS
// ============================================================
constexpr const char* ROAD_ENGINE_PATH = "/home/car4/Downloads/road/kga2.engine";
constexpr const char* SIGN_ENGINE_PATH = "/home/car4/Downloads/sign/huy_sign.engine";

// ============================================================
// STEERING CALIBRATION (Hồi quy)
// ============================================================
constexpr float STEER_LEFT_M  = 0.96f;
constexpr float STEER_LEFT_B  = -6.2f;
constexpr float STEER_RIGHT_M = 0.71f;
constexpr float STEER_RIGHT_B = 2.5f;

} } // namespace
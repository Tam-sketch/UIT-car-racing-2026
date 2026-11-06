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
inline constexpr char ROAD_ENGINE_PATH[] =
    "/home/car4/Downloads/road/kga2.engine";
// When sign model is ready:
// inline constexpr char SIGN_ENGINE_PATH[] = "...";

// --------------------------------------------------------
// Hardware limits
// --------------------------------------------------------
inline constexpr float MAX_SPEED_CMS = 15.0f;
inline constexpr float MIN_SPEED_CMS = 5.0f;

// NOTE: 50° has NOT been physically verified. The thisinh
// repo's keyboard clamps at ±20°. Before trusting 50°,
// bench-test the servo at 5/10/15/20/30/40/50° for stalls
// and mechanical binding. Until then, 20 is the safe value.
inline constexpr int   MAX_STEER_DEG = 50;

// Servo driver: positive angle → turn left (opposite of PID convention).
inline constexpr bool  INVERT_STEER  = true;

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

// --------------------------------------------------------
// UI
// --------------------------------------------------------
inline constexpr bool SHOW_UI = true;

}  // namespace config
}  // namespace ucr
// ============================================================
// NavTypes.h — Chỉ chứa enum/struct cho navigation
// Thay thế NavigationBrain.h (chỉ giữ type, không giữ class)
// ============================================================
#pragma once

#include <string>

enum class NavAction {
    FOLLOW_LANE = 0,
    FORCE_LEFT,
    FORCE_RIGHT,
    STOP_PARKING
};

struct NavDecision {
    NavAction   action         = NavAction::FOLLOW_LANE;
    float       target_speed   = 22.0f;
    bool        emergency_stop = false;
    std::string active_sign    = "NONE";
};
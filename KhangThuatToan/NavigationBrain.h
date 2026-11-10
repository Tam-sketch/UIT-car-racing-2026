// ============================================================
// NavigationBrain.h
// ============================================================
#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <iostream>
#include "SignDetector.h"

enum class NavAction {
    FOLLOW_LANE = 0,
    FORCE_LEFT,
    FORCE_RIGHT,
    STOP_PARKING
};

struct NavDecision {
    NavAction action = NavAction::FOLLOW_LANE;
    float steer_bias_px = 0.0f;
    float target_speed = 14.0f;
    bool emergency_stop = false;
    std::string active_sign = "NONE";
};

class NavigationBrain {
public:
    NavigationBrain() = default;
    void ClearLatchedSign() {
        latched_sign_ = "NONE";
        latch_cooldown_frames_ = 0;
        latched_avoid_bias_ = 0.0f;
    }

    void UpdateSign(const std::vector<SignObject>& signs) {
        for (const auto& s : signs) {
            if (s.confidence >= 0.70f && s.box.area() >= 200) {
                if (s.name == "traffic_light") continue;
                if (latched_sign_ != s.name) {
                    latched_avoid_bias_ = 0.0f;
                }
                latched_sign_ = s.name;
                latch_cooldown_frames_ = LATCH_DURATION_FRAMES;
                std::cout << "[NAV BRAIN] >> DETECTED & LATCHED SIGN: " << s.name << "\n";
                break;
            }
        }
    }

    std::string GetTurnDirection() const {
        // Nhóm rẽ trái
        if (latched_sign_ == "turn_left" || 
            latched_sign_ == "construction_ahead" || 
            latched_sign_ == "no_right") {
            return "LEFT";
        }
        // Nhóm rẽ phải
        if (latched_sign_ == "turn_right" || 
            latched_sign_ == "no_left") {
            return "RIGHT";
        }
        return "NONE";
    }

    NavDecision Process(const cv::Mat& roadMask, float currentSpeed) {
        NavDecision dec;
        dec.target_speed = currentSpeed;
        dec.active_sign = latched_sign_;

        if (latch_cooldown_frames_ > 0) {
            latch_cooldown_frames_--;
        } else {
            latched_sign_ = "NONE";
            dec.active_sign = "NONE";
        }

        if (dec.active_sign == "parking") {
            dec.action = NavAction::STOP_PARKING;
            dec.target_speed = 0.0f;
            dec.emergency_stop = true;
            return dec;
        }

        std::string turn_direction = GetTurnDirection();

        if (turn_direction == "LEFT") {
            dec.action = NavAction::FORCE_LEFT;
            dec.target_speed = 10.0f;
        } else if (turn_direction == "RIGHT") {
            dec.action = NavAction::FORCE_RIGHT;
            dec.target_speed = 10.0f;
        } else {
            dec.action = NavAction::FOLLOW_LANE;
            dec.steer_bias_px = 0.0f;
        }

        return dec;
    }

private:
    std::string latched_sign_ = "NONE";
    float latched_avoid_bias_ = 0.0f;
    int latch_cooldown_frames_ = 0;
    static constexpr int LATCH_DURATION_FRAMES = 50; 
};

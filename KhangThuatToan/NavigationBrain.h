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
    float target_speed = 22.0f;
    bool emergency_stop = false;
    std::string active_sign = "NONE";
};

class NavigationBrain {
public:
    NavigationBrain() = default;
    void ClearLatchedSign() {
        latched_sign_ = "NONE";
        latch_cooldown_frames_ = 0;
        candidate_sign_ = "NONE";
        candidate_count_ = 0;
        candidate_miss_ = 0;
    }

    void UpdateSign(const std::vector<SignObject>& signs) {
        std::string best_sign = "NONE";
        float best_conf = 0.0f;

        for (const auto& s : signs) {
            if (s.confidence >= 0.70f && s.box.area() >= 200) {
                if (s.name == "traffic_light") continue;
                if (block_frames_ > 0 && s.name == blocked_sign_) continue; // biển vừa xử lý xong -> bỏ qua
                if (s.confidence > best_conf) {
                    best_conf = s.confidence;
                    best_sign = s.name;
                }
            }
        }

        if (best_sign != "NONE") {
            if (best_sign == candidate_sign_) {
                candidate_count_++;
                candidate_miss_ = 0;
            } else {
                candidate_sign_ = best_sign;
                candidate_count_ = 1;
                candidate_miss_ = 0;
            }

            // Chốt biển sau 5 frame xác nhận (parking chốt ở 3 frame để phanh kịp)
            const int req_frames = (best_sign == "parking") ? 3 : SIGN_CONFIRM_FRAMES;
            if (candidate_count_ >= req_frames) {
                if (latched_sign_ != best_sign) {
                    std::cout << "[NAV BRAIN] >> CONFIRMED (" << candidate_count_
                              << " frames) & LATCHED SIGN: " << best_sign << "\n";
                }
                latched_sign_ = best_sign;
                latch_cooldown_frames_ = LATCH_DURATION_FRAMES;
            }
        } else {
            // Cho phép hụt tối đa 1 frame do nhòe chuyển động trước khi reset bộ đếm 5 frame
            if (candidate_count_ > 0) {
                candidate_miss_++;
                if (candidate_miss_ > 1) {
                    candidate_sign_ = "NONE";
                    candidate_count_ = 0;
                    candidate_miss_ = 0;
                }
            }
        }
    }

    void BlockSign(const std::string& name, int frames) {
        blocked_sign_ = name;
        block_frames_ = frames;
        if (latched_sign_ == name) ClearLatchedSign();
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

    NavDecision Process(float currentSpeed) {
        if (block_frames_ > 0) block_frames_--; 
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
        }

        return dec;
    }

    // Overload nhận thêm cv::Mat roadMask nếu file cpp nào đó truyền 2 tham số
    NavDecision Process(const cv::Mat& /*roadMask*/, float currentSpeed) {
        return Process(currentSpeed);
    }

private:
    std::string latched_sign_ = "NONE";
    int latch_cooldown_frames_ = 0;
    static constexpr int LATCH_DURATION_FRAMES = 50; 
    static constexpr int SIGN_CONFIRM_FRAMES   = 5;  // Đủ 5 frame mới chốt biển
    std::string candidate_sign_ = "NONE";
    int candidate_count_ = 0;
    int candidate_miss_  = 0;
    std::string blocked_sign_ = "NONE";
    int block_frames_ = 0;
};

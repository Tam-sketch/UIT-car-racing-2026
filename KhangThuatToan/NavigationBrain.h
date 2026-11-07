// ============================================================
// NavigationBrain.h — Bộ điều phối hành vi biển báo & tách nhánh
// ============================================================
#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include "SignDetector.h"

enum class NavAction {
    FOLLOW_LANE = 0,
    FORCE_LEFT,
    FORCE_RIGHT,
    FORCE_STRAIGHT,
    AVOID_CONSTRUCTION,
    HANDLE_ROUNDABOUT,
    STOP_PARKING
};

struct NavDecision {
    NavAction action = NavAction::FOLLOW_LANE;
    float steer_bias_px = 0.0f; // Bù lệch điểm ngắm sang trái (-) hoặc sang phải (+)
    float target_speed = 14.0f;
    bool emergency_stop = false;
    std::string active_sign = "NONE";
};

class NavigationBrain {
public:
    NavigationBrain() = default;

    // Cập nhật trạng thái biển báo (quét mỗi 5 frame)
    void UpdateSign(const std::vector<SignObject>& signs) {
        for (const auto& s : signs) {
            // Lọc điều kiện: Conf >= 0.7 và Diện tích > 800 px
            if (s.confidence >= 0.70f && s.box.area() >= 800) {
                // Bỏ qua traffic_light như yêu cầu
                if (s.name == "traffic_light") continue;

                // Nếu gặp biển mới khác biển cũ, reset chốt hướng rẽ
                if (latched_sign_ != s.name) {
                    latched_avoid_bias_ = 0.0f;
                }

                // Cập nhật biển báo mới và kích hoạt bộ chốt lệnh (Latch)
                latched_sign_ = s.name;
                latch_cooldown_frames_ = LATCH_DURATION_FRAMES;
                std::cout << "[NAV BRAIN] >> DETECTED & LATCHED SIGN: " << s.name 
                          << " (Conf: " << s.confidence << ", Area: " << s.box.area() << ")\n";
                break;
            }
        }
    }

    // Xử lý hành vi và tính toán góc bù làn dựa trên mặt đường thực tế
    NavDecision Process(const cv::Mat& roadMask, float currentSpeed, double roadWidthNear) {
        NavDecision dec;
        dec.target_speed = currentSpeed;

        // Giảm trừ bộ đếm chốt lệnh qua từng frame
        if (latch_cooldown_frames_ > 0) {
            latch_cooldown_frames_--;
            dec.active_sign = latched_sign_;
        } else {
            latched_sign_ = "NONE";
            dec.active_sign = "NONE";
            latched_avoid_bias_ = 0.0f;
        }

        // 1. GẶP PARKING: DỪNG XE TẠM THỜI
        if (dec.active_sign == "parking") {
            dec.action = NavAction::STOP_PARKING;
            dec.target_speed = 0.0f;
            dec.emergency_stop = true;
            dec.steer_bias_px = 0.0f;
            return dec;
        }

        // 2. PHÂN TÍCH NHÁNH ĐƯỜNG PHÍA TRƯỚC BẰNG VÙNG (ZONE ANALYSIS)
        bool has_left_branch = false;
        bool has_right_branch = false;
        bool has_straight_path = false;

        if (!roadMask.empty()) {
            AnalyzeJunctionsZone(roadMask, has_left_branch, has_right_branch, has_straight_path);
        }

        // 3. XỬ LÝ QUY TẮC TỪNG BIỂN BÁO
        if (dec.active_sign == "turn_left") {
            dec.action = NavAction::FORCE_LEFT;
            dec.steer_bias_px = -65.0f; 
            dec.target_speed = 11.5f;   
        }
        else if (dec.active_sign == "turn_right") {
            dec.action = NavAction::FORCE_RIGHT;
            dec.steer_bias_px = +65.0f; 
            dec.target_speed = 11.5f;
        }
        else if (dec.active_sign == "no_left") {
            if (has_straight_path) {
                dec.action = NavAction::FORCE_STRAIGHT;
                dec.steer_bias_px = 0.0f;
            } else {
                dec.action = NavAction::FORCE_RIGHT;
                dec.steer_bias_px = +55.0f;
                dec.target_speed = 12.0f;
            }
        }
        else if (dec.active_sign == "no_right") {
            if (has_straight_path) {
                dec.action = NavAction::FORCE_STRAIGHT;
                dec.steer_bias_px = 0.0f;
            } else {
                dec.action = NavAction::FORCE_LEFT;
                dec.steer_bias_px = -55.0f;
                dec.target_speed = 12.0f;
            }
        }
        else if (dec.active_sign == "construction_ahead") {
            dec.action = NavAction::AVOID_CONSTRUCTION;
            dec.target_speed = 11.0f;
            
            // Chốt hướng rẽ né công trường ở frame đầu tiên
            if (std::abs(latched_avoid_bias_) < 1.0f) {
                if (has_right_branch && !has_left_branch) {
                    latched_avoid_bias_ = +55.0f;
                } else if (has_left_branch && !has_right_branch) {
                    latched_avoid_bias_ = -55.0f;
                } else if (has_right_branch) {
                    // Nếu có cả 2 nhánh, ưu tiên rẽ phải
                    latched_avoid_bias_ = +55.0f;
                } else {
                    // Nếu phân tích vùng bị nhiễu không thấy rõ nhánh nào, 
                    // ép mặc định tránh trái
                    latched_avoid_bias_ = -55.0f;
                }
            }
            dec.steer_bias_px = latched_avoid_bias_;
        }
        else if (dec.active_sign == "roundabout") {
            dec.action = NavAction::HANDLE_ROUNDABOUT;
            dec.steer_bias_px = +35.0f;
            dec.target_speed = 11.5f;
        }
        else {
            dec.action = NavAction::FOLLOW_LANE;
            dec.steer_bias_px = 0.0f;
        }

        return dec;
    }

private:
    // Quét nhiều dòng liên tiếp (Zone) để khắc phục nhược điểm của 1 đường cắt đơn
    void AnalyzeJunctionsZone(const cv::Mat& mask, bool& has_left, bool& has_right, bool& has_straight) {
        int w = mask.cols;
        int h = mask.rows;
        
        int start_y = static_cast<int>(h * 0.40);
        int end_y   = static_cast<int>(h * 0.50);

        int left_white_count = 0;
        int center_white_count = 0;
        int right_white_count = 0;

        for (int y = start_y; y <= end_y; y += 2) {
            const uchar* row = mask.ptr<uchar>(y);
            for (int x = 0; x < w; ++x) {
                if (row[x] > 0) {
                    if (x < w * 0.35) left_white_count++;
                    else if (x > w * 0.65) right_white_count++;
                    else center_white_count++;
                }
            }
        }

        // Ngưỡng tính toán cần cao hơn vì đếm theo vùng (Zone)
        int threshold = (end_y - start_y) * 5; 
        
        has_left     = (left_white_count > threshold);
        has_right    = (right_white_count > threshold);
        has_straight = (center_white_count > threshold);
    }

    std::string latched_sign_ = "NONE";
    float latched_avoid_bias_ = 0.0f;
    int latch_cooldown_frames_ = 0;
    static constexpr int LATCH_DURATION_FRAMES = 50; 
};

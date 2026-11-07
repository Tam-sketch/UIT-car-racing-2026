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
        }

        // ----------------------------------------------------
        // 1. GẶP PARKING: DỪNG XE TẠM THỜI
        // ----------------------------------------------------
        if (dec.active_sign == "parking") {
            dec.action = NavAction::STOP_PARKING;
            dec.target_speed = 0.0f;
            dec.emergency_stop = true;
            dec.steer_bias_px = 0.0f;
            return dec;
        }

        // ----------------------------------------------------
        // 2. PHÂN TÍCH NHÁNH ĐƯỜNG PHÍA TRƯỚC (JUNCTION ANALYSIS)
        // ----------------------------------------------------
        bool has_left_branch = false;
        bool has_right_branch = false;
        bool has_straight_path = false;

        if (!roadMask.empty()) {
            AnalyzeJunctions(roadMask, has_left_branch, has_right_branch, has_straight_path);
        }

        // ----------------------------------------------------
        // 3. XỬ LÝ QUY TẮC TỪNG BIỂN BÁO
        // ----------------------------------------------------
        if (dec.active_sign == "turn_left") {
            dec.action = NavAction::FORCE_LEFT;
            dec.steer_bias_px = -65.0f; // Kéo tâm ngắm lệch mạnh sang trái
            dec.target_speed = 11.5f;   // Giảm tốc để bo cua an toàn
        }
        else if (dec.active_sign == "turn_right") {
            dec.action = NavAction::FORCE_RIGHT;
            dec.steer_bias_px = +65.0f; // Kéo tâm ngắm lệch mạnh sang phải
            dec.target_speed = 11.5f;
        }
        else if (dec.active_sign == "no_left") {
            // Không được rẽ trái: Ưu tiên đi thẳng, nếu không có đường thẳng thì rẽ phải
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
            // Không được rẽ phải: Ưu tiên đi thẳng, nếu không có đường thẳng thì rẽ trái
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
            // Có công trường: Tuyệt đối tránh đường chính/thẳng, lách sang nhánh phụ
            dec.action = NavAction::AVOID_CONSTRUCTION;
            dec.target_speed = 11.0f;
            if (has_right_branch) {
                dec.steer_bias_px = +60.0f; // Ưu tiên rẽ phải tránh chướng ngại
            } else if (has_left_branch) {
                dec.steer_bias_px = -60.0f; // Hoặc rẽ trái
            } else {
                dec.steer_bias_px = +45.0f; // Ép biên né chướng ngại
            }
        }
        else if (dec.active_sign == "roundabout") {
            // Bùng binh: Xe theo quy chuẩn nhập làn vòng xuyến (ôm nhẹ mép ngoài/bên phải)
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
    // Kiểm tra các nhánh rẽ ở khoảng 40% - 60% chiều cao khung hình mask
    void AnalyzeJunctions(const cv::Mat& mask, bool& has_left, bool& has_right, bool& has_straight) {
        int w = mask.cols;
        int h = mask.rows;
        int scan_y = h * 0.45; // Quét ở tầm nhìn xa vừa phải

        const uchar* row = mask.ptr<uchar>(scan_y);
        int left_white_count = 0;
        int center_white_count = 0;
        int right_white_count = 0;

        for (int x = 0; x < w; ++x) {
            if (row[x] > 0) {
                if (x < w * 0.35) left_white_count++;
                else if (x > w * 0.65) right_white_count++;
                else center_white_count++;
            }
        }

        has_left     = (left_white_count > 15);
        has_right    = (right_white_count > 15);
        has_straight = (center_white_count > 15);
    }

    std::string latched_sign_ = "NONE";
    int latch_cooldown_frames_ = 0;
    static constexpr int LATCH_DURATION_FRAMES = 50; // Duy trì hành vi ~7-8 giây sau khi thấy biển
};
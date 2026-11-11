// ============================================================
// NavigationBrain.h
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
            if (s.confidence >= 0.70f && s.box.area() >= 280) {
                if (s.name == "traffic_light" || s.name == "walking_person") continue;
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

    // ------------------------------------------------------------
    // NGƯỜI ĐI BỘ + ĐÈN GIAO THÔNG (gọi mỗi frame, độc lập với chốt biển báo)
    // - Pedestrian: không xét area, thấy đủ 3 frame -> dừng ngay;
    //   còn thấy box là còn dừng; mất liên tiếp 5 frame -> tự đi tiếp.
    //   Không ghi đè vào latched_sign_ nên không cần reset biển và không làm mất biển đang nhớ.
    // - Traffic Light: Đỏ/Vàng -> dừng khi lại gần đèn (để kịp chốt biển kế bên đèn);
    //   Xanh -> đi tiếp (và thực hiện lệnh rẽ của biển kế bên nếu có).
    // ------------------------------------------------------------
    void UpdateHazards(const cv::Mat& frame, const std::vector<SignObject>& objs, bool turn_already_armed = false) {
        // ---- 1. NGƯỜI ĐI BỘ (PEDESTRIAN): không quan tâm area ----
        bool ped_now = false;
        for (const auto& o : objs) {
            if (o.name == "walking_person" && o.confidence >= PED_CONF) {
                ped_now = true;
                break;
            }
        }

        if (ped_now) {
            ped_seen_++;
            ped_miss_ = 0;
            if (!ped_stop_ && ped_seen_ >= PED_CONFIRM_FRAMES) {
                ped_stop_ = true;
                std::cout << "[NAV BRAIN] >> PEDESTRIAN CONFIRMED (3 frames) -> STOP IMMEDIATELY!\n";
            }
        } else {
            if (ped_stop_) {
                // Đang dừng vì người đi bộ: phải mất box đủ 5 frame liên tiếp mới được đi tiếp
                ped_miss_++;
                if (ped_miss_ >= PED_CLEAR_FRAMES) {
                    ped_stop_ = false;
                    ped_seen_ = 0;
                    ped_miss_ = 0;
                    std::cout << "[NAV BRAIN] >> PEDESTRIAN CLEARED (5 frames) -> RESUME GO!\n";
                }
            } else {
                // Chưa chốt đủ 3 frame mà mất box -> reset bộ đếm tiến tới 3 frame
                ped_seen_ = 0;
                ped_miss_ = 0;
            }
        }

        // ---- 2. ĐÈN GIAO THÔNG (TRAFFIC LIGHT) ----
        const SignObject* tl = nullptr;
        for (const auto& o : objs) {
            if (o.name == "traffic_light" && o.confidence >= TL_CONF &&
                (!tl || o.box.area() > tl->box.area())) {
                tl = &o;
            }
        }

        if (!tl) {
            tl_miss_++;
            tl_red_cnt_ = 0;
            tl_green_cnt_ = 0;
            tl_approaching_red_ = false;
            if (tl_stop_ && tl_miss_ >= TL_LOST_RELEASE_FRAMES) {
                tl_stop_ = false;
                std::cout << "[NAV BRAIN] >> TRAFFIC LIGHT OUT OF VIEW -> GO\n";
            }
            if (tl_miss_ >= 5) tl_color_ = "NONE";
            return;
        }

        tl_miss_ = 0;
        std::string detected_color = ClassifyLight(frame, tl->box);
        if (detected_color != "NONE") {
            tl_color_ = detected_color;
        }

        // Vị trí dừng: tiến lại gần đèn một chút để camera kịp chốt biển báo đặt kế bên đèn.
        // Nếu đã chốt được biển kế bên (latched_sign_ != "NONE" hoặc turn_already_armed) thì dừng ở khoảng cách vừa phải,
        // nếu chưa chốt được biển thì cho xe tiến sát hơn một chút để lấy trọn biển báo.
        const int area = tl->box.area();
        const int h_box = tl->box.height;
        const bool has_sign_ready = (latched_sign_ != "NONE") || turn_already_armed;
        const bool close_enough = has_sign_ready
            ? (area >= TL_STOP_AREA_WITH_SIGN || h_box >= TL_STOP_HEIGHT_WITH_SIGN)
            : (area >= TL_STOP_AREA_MAX       || h_box >= TL_STOP_HEIGHT_MAX);

        if (tl_color_ == "RED" || tl_color_ == "YELLOW") {
            tl_red_cnt_++;
            tl_green_cnt_ = 0;
            tl_approaching_red_ = (tl_red_cnt_ >= TL_CONFIRM_FRAMES) && !close_enough;
            if (!tl_stop_ && close_enough && tl_red_cnt_ >= TL_CONFIRM_FRAMES) {
                tl_stop_ = true;
                tl_approaching_red_ = false;
                std::cout << "[NAV BRAIN] >> TRAFFIC LIGHT " << tl_color_
                          << " (area=" << area << ", sign=" << latched_sign_ << ") -> STOP!\n";
            }
        } else if (tl_color_ == "GREEN") {
            tl_green_cnt_++;
            tl_red_cnt_ = 0;
            tl_approaching_red_ = false;
            if (tl_stop_ && tl_green_cnt_ >= TL_CONFIRM_FRAMES) {
                tl_stop_ = false;
                std::cout << "[NAV BRAIN] >> TRAFFIC LIGHT GREEN (sign=" << latched_sign_ << ") -> GO!\n";
            }
        }
    }

    bool HazardStop() const { return ped_stop_ || tl_stop_; }
    bool PedestrianStop() const { return ped_stop_; }
    bool TrafficLightStop() const { return tl_stop_; }
    bool ApproachingRedLight() const { return tl_approaching_red_; }
    const std::string& TrafficLightColor() const { return tl_color_; }

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

        // Khi đang dừng hoặc đang tiến chậm lại gần đèn đỏ / người đi bộ:
        // đóng băng thời gian giữ biển (latch_cooldown_frames_) để biển kế bên đèn không bao giờ bị hết hạn
        if ((HazardStop() || ApproachingRedLight()) && latched_sign_ != "NONE") {
            latch_cooldown_frames_ = LATCH_DURATION_FRAMES;
        } else if (latch_cooldown_frames_ > 0) {
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

        // Khi thấy đèn Đỏ/Vàng từ xa nhưng chưa đủ gần để dừng: giảm tốc xuống 10 cm/s để tiến lại gần đèn và chốt biển kế bên
        if (ApproachingRedLight()) {
            dec.target_speed = std::min(dec.target_speed, 10.0f);
        }

        // Dừng do người đi bộ / đèn đỏ-vàng: vẫn giữ nguyên dec.action & dec.active_sign theo biển báo
        if (HazardStop()) {
            dec.target_speed = 0.0f;
            dec.emergency_stop = true;
        }
        return dec;
    }

    // Overload nhận thêm cv::Mat roadMask nếu file cpp nào đó truyền 2 tham số
    NavDecision Process(const cv::Mat& /*roadMask*/, float currentSpeed) {
        return Process(currentSpeed);
    }

private:
    // Phân loại màu đèn giao thông bằng HSV (hỗ trợ cả tâm đèn LED sáng gắt lẫn viền màu)
    static std::string ClassifyLight(const cv::Mat& frame, const cv::Rect& box) {
        if (frame.empty()) return "NONE";
        cv::Rect r = box & cv::Rect(0, 0, frame.cols, frame.rows);
        if (r.area() < 20) return "NONE";
        cv::Mat hsv;
        cv::cvtColor(frame(r), hsv, cv::COLOR_BGR2HSV);
        int red = 0, yellow = 0, green = 0;
        for (int y = 0; y < hsv.rows; y++) {
            const cv::Vec3b* p = hsv.ptr<cv::Vec3b>(y);
            for (int x = 0; x < hsv.cols; x++) {
                int h = p[x][0], s = p[x][1], v = p[x][2];
                if (!((s >= 70 && v >= 120) || (s >= 40 && v >= 185))) continue;
                if (h <= 12 || h >= 160) red++;
                else if (h >= 13 && h <= 38) yellow++;
                else if (h >= 42 && h <= 98) green++;
            }
        }
        const int minPix = std::max(5, r.area() / 120);
        int best = std::max({red, yellow, green});
        if (best < minPix) return "NONE";
        if (best == red) return "RED";
        if (best == yellow) return "YELLOW";
        return "GREEN";
    }

    // Người đi bộ (chốt 3 frame không cần area, mất 5 frame mới đi tiếp)
    static constexpr float PED_CONF           = 0.55f;
    static constexpr int   PED_CONFIRM_FRAMES = 3;
    static constexpr int   PED_CLEAR_FRAMES   = 5;
    bool ped_stop_ = false;
    int  ped_seen_ = 0, ped_miss_ = 0;

    // Đèn giao thông (tiến lại gần đèn để nhận biển báo kế bên rồi mới dừng)
    static constexpr float TL_CONF                  = 0.55f;
    static constexpr int   TL_CONFIRM_FRAMES        = 2;
    static constexpr int   TL_STOP_AREA_WITH_SIGN   = 950;   // Khi đã chốt được biển kế bên đèn
    static constexpr int   TL_STOP_HEIGHT_WITH_SIGN = 40;
    static constexpr int   TL_STOP_AREA_MAX         = 1350;  // Khi chưa chốt biển, tiến gần hơn chút nữa mới dừng
    static constexpr int   TL_STOP_HEIGHT_MAX       = 50;
    static constexpr int   TL_LOST_RELEASE_FRAMES   = 20;
    bool tl_stop_ = false;
    bool tl_approaching_red_ = false;
    int  tl_red_cnt_ = 0, tl_green_cnt_ = 0, tl_miss_ = 0;
    std::string tl_color_ = "NONE";

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

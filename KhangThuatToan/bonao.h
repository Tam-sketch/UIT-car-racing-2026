// ============================================================
// NavigationBrain.h
// ============================================================
//4
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

struct TrafficLightMetrics {
    std::string raw_color = "NONE";
    std::string confirmed_color = "NONE";
    int top_brightness = 0;   // Chóa trên (ĐỎ)
    int mid_brightness = 0;   // Chóa giữa (VÀNG)
    int bot_brightness = 0;   // Chóa dưới (XANH)
    int diff_brightness = 0;  // Độ chênh lệch giữa chóa sáng nhất và chóa nhì
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
            tl_none_cnt_  = 0;
            tl_approaching_red_ = false;
            if (tl_stop_ && tl_miss_ >= TL_LOST_RELEASE_FRAMES) {
                tl_stop_ = false;
                std::cout << "[NAV BRAIN] >> TRAFFIC LIGHT OUT OF VIEW -> GO\n";
            }
            if (tl_miss_ >= 5) {
                tl_color_ = "NONE";
                tl_metrics_ = TrafficLightMetrics{};
            }
            return;
        }

        tl_miss_ = 0;
        int top_b = 0, mid_b = 0, bot_b = 0, diff = 0;
        std::string detected_color = ClassifyLight(frame, tl->box, top_b, mid_b, bot_b, diff);
        tl_metrics_.raw_color       = detected_color;
        tl_metrics_.top_brightness  = top_b;
        tl_metrics_.mid_brightness  = mid_b;
        tl_metrics_.bot_brightness  = bot_b;
        tl_metrics_.diff_brightness = diff;

        // Máy trạng thái lọc xác nhận màu đèn qua các frame liên tiếp
        if (detected_color == "RED" || detected_color == "YELLOW") {
            tl_red_cnt_++;
            tl_green_cnt_ = 0;
            tl_none_cnt_  = 0;
            if (tl_red_cnt_ >= TL_CONFIRM_FRAMES) {
                tl_color_ = detected_color;
            }
        } else if (detected_color == "GREEN") {
            tl_green_cnt_++;
            tl_red_cnt_  = 0;
            tl_none_cnt_ = 0;
            if (tl_green_cnt_ >= TL_CONFIRM_FRAMES) {
                tl_color_ = "GREEN";
            }
        } else { // "NONE" (cả 3 vỏ sáng/tối ngang nhau)
            tl_none_cnt_++;
            if (tl_none_cnt_ >= 4) {
                tl_red_cnt_   = 0;
                tl_green_cnt_ = 0;
                // Nếu đang dừng đèn đỏ thì không tự ý nhả nếu đèn chỉ bị rung nhẹ chớp tắt
                if (!tl_stop_) {
                    tl_color_ = "NONE";
                }
            }
        }
        tl_metrics_.confirmed_color = tl_color_;

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
            tl_approaching_red_ = !close_enough;
            if (!tl_stop_ && close_enough && tl_red_cnt_ >= TL_CONFIRM_FRAMES) {
                tl_stop_ = true;
                tl_approaching_red_ = false;
                std::cout << "[NAV BRAIN] >> TRAFFIC LIGHT " << tl_color_
                          << " [Top:" << top_b << " Mid:" << mid_b << " Bot:" << bot_b << " Diff:" << diff << "]"
                          << " (area=" << area << ", sign=" << latched_sign_ << ") -> STOP!\n";
            }
        } else if (tl_color_ == "GREEN") {
            tl_approaching_red_ = false;
            if (tl_stop_ && tl_green_cnt_ >= TL_CONFIRM_FRAMES) {
                tl_stop_ = false;
                std::cout << "[NAV BRAIN] >> TRAFFIC LIGHT GREEN "
                          << " [Top:" << top_b << " Mid:" << mid_b << " Bot:" << bot_b << " Diff:" << diff << "]"
                          << " (sign=" << latched_sign_ << ") -> GO!\n";
            }
        }
    }

    bool HazardStop() const { return ped_stop_ || tl_stop_; }
    bool PedestrianStop() const { return ped_stop_; }
    bool TrafficLightStop() const { return tl_stop_; }
    bool ApproachingRedLight() const { return tl_approaching_red_; }
    const std::string& TrafficLightColor() const { return tl_color_; }
    const TrafficLightMetrics& GetTLMetrics() const { return tl_metrics_; }

    void SetLatchedSign(const std::string& name) {
        latched_sign_ = name;
        latch_cooldown_frames_ = LATCH_DURATION_FRAMES;
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

        // Khi đang dừng hoặc đang tiến chậm lại gần đèn đỏ / người đi bộ / bãi đỗ xe:
        // đóng băng thời gian giữ biển (latch_cooldown_frames_) để biển không bao giờ bị hết hạn
        if ((HazardStop() || ApproachingRedLight() || latched_sign_ == "parking") && latched_sign_ != "NONE") {
            latch_cooldown_frames_ = LATCH_DURATION_FRAMES;
        } else if (latch_cooldown_frames_ > 0) {
            latch_cooldown_frames_--;
        } else {
            latched_sign_ = "NONE";
            dec.active_sign = "NONE";
        }

        if (dec.active_sign == "parking") {
            dec.action = NavAction::FOLLOW_LANE;
            dec.target_speed = 10.0f; // Tốc độ tiếp cận bãi đỗ
            dec.emergency_stop = false;
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
    // =========================================================================
    // PHÂN LOẠI MÀU ĐÈN GIAO THÔNG (KẾT HỢP ĐỘ SÁNG + MÀU SẮC THỰC HSV & BGR):
    // Khắc phục triệt để lỗi ánh sáng mặt trời/đèn trần chiếu từ trên đỉnh làm chói trắng tầng Đỏ:
    // - Chỉ khi chóa có màu sắc thực thụ (Đỏ: H<=14/H>=165 & S>=45 & R>G,B; Xanh: H 35-95 & S>=35 & G>R)
    //   mới được tính điểm màu (Color Score). Ánh sáng chói trắng (S thấp, kênh RGB ngang nhau)
    //   sẽ có điểm màu = 0 và bị loại bỏ hoàn toàn, không thể nhận nhầm thành Đỏ!
    // =========================================================================
    static std::string ClassifyLight(const cv::Mat& frame, const cv::Rect& box,
                                     int& out_top_b, int& out_mid_b, int& out_bot_b, int& out_diff) {
        out_top_b = 0; out_mid_b = 0; out_bot_b = 0; out_diff = 0;
        if (frame.empty()) return "NONE";
        cv::Rect r = box & cv::Rect(0, 0, frame.cols, frame.rows);
        if (r.width < 6 || r.height < 12 || r.area() < 60) return "NONE";

        cv::Mat roi = frame(r);
        cv::Mat hsv;
        cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);

        const int H = r.height;
        const int W = r.width;

        // Cắt bớt 15% viền trái/phải để tập trung đúng vào chóa đèn tròn bên trong
        const int x_start = static_cast<int>(W * 0.15);
        const int x_end   = std::max(x_start + 1, static_cast<int>(W * 0.85));

        auto analyze_slot = [&](int y_pct_start, int y_pct_end, int slot_type,
                                int& avg_brightness, int& color_pixels, float& color_score) {
            int y0 = std::clamp(static_cast<int>(H * y_pct_start / 100.0), 0, H - 1);
            int y1 = std::clamp(static_cast<int>(H * y_pct_end / 100.0), y0 + 1, H);

            std::vector<int> b_vals;
            b_vals.reserve((y1 - y0) * (x_end - x_start));
            color_pixels = 0;
            color_score  = 0.0f;

            for (int y = y0; y < y1; ++y) {
                const cv::Vec3b* bgr_ptr = roi.ptr<cv::Vec3b>(y);
                const cv::Vec3b* hsv_ptr = hsv.ptr<cv::Vec3b>(y);
                for (int x = x_start; x < x_end; ++x) {
                    int b   = bgr_ptr[x][0];
                    int g   = bgr_ptr[x][1];
                    int r_v = bgr_ptr[x][2];

                    int h = hsv_ptr[x][0];
                    int s = hsv_ptr[x][1];
                    int v = hsv_ptr[x][2];

                    b_vals.push_back(v);

                    if (slot_type == 0) {
                        // TẦNG ĐỎ (TOP):
                        // Phải có màu đỏ thực thụ: H đỏ, Saturation >= 45 (chống chói trắng), và R vượt trội G, B
                        bool is_red = (((h <= 14 || h >= 165) && s >= 45 && v >= 65) ||
                                       (r_v >= 110 && r_v > g * 1.30 && r_v > b * 1.30));
                        if (is_red) {
                            color_pixels++;
                            color_score += static_cast<float>(r_v - std::max(g, b));
                        }
                    } else if (slot_type == 1) {
                        // TẦNG VÀNG (MID):
                        bool is_yellow = (((h >= 15 && h <= 34) && s >= 45 && v >= 75) ||
                                          (r_v >= 120 && g >= 110 && b < std::min(r_v, g) * 0.70));
                        if (is_yellow) {
                            color_pixels++;
                            color_score += static_cast<float>((r_v + g) / 2 - b);
                        }
                    } else if (slot_type == 2) {
                        // TẦNG XANH (BOT):
                        // Màu xanh lá / xanh ngọc: H trong [35, 95], Saturation >= 35, và G vượt trội R
                        bool is_green = (((h >= 35 && h <= 95) && s >= 35 && v >= 65) ||
                                         (g >= 95 && g > r_v * 1.15 && (g > b * 0.88 || s >= 35)));
                        if (is_green) {
                            color_pixels++;
                            color_score += static_cast<float>(g - r_v);
                        }
                    }
                }
            }

            if (b_vals.empty()) {
                avg_brightness = 0;
                return;
            }
            std::sort(b_vals.begin(), b_vals.end(), std::greater<int>());
            const size_t top_k = std::max<size_t>(1, b_vals.size() / 5);
            long sum = 0;
            for (size_t i = 0; i < top_k; ++i) sum += b_vals[i];
            avg_brightness = static_cast<int>(sum / top_k);
        };

        int top_b = 0, mid_b = 0, bot_b = 0;
        int red_px = 0, yel_px = 0, grn_px = 0;
        float red_score = 0.0f, yel_score = 0.0f, grn_score = 0.0f;

        analyze_slot(4,  33, 0, top_b, red_px, red_score); // Tầng 1: ĐỎ
        analyze_slot(35, 65, 1, mid_b, yel_px, yel_score); // Tầng 2: VÀNG
        analyze_slot(67, 96, 2, bot_b, grn_px, grn_score); // Tầng 3: XANH

        out_top_b = top_b;
        out_mid_b = mid_b;
        out_bot_b = bot_b;
        out_diff  = std::abs(top_b - bot_b);

        // QUYẾT ĐỊNH MÀU SẮC ĐÈN:
        // Ưu tiên điểm màu thực tế (Color Score) trước để loại bỏ hoàn toàn ánh sáng trắng chói từ trần/mặt trời!
        const int MIN_COLOR_PX = 3;
        const float MIN_COLOR_SCORE = 60.0f;

        // 1. Nếu tầng Xanh có màu xanh thực thụ vượt trội
        if (grn_px >= MIN_COLOR_PX && grn_score >= MIN_COLOR_SCORE && grn_score > red_score * 1.25f) {
            return "GREEN";
        }

        // 2. Nếu tầng Đỏ có màu đỏ thực thụ vượt trội
        if (red_px >= MIN_COLOR_PX && red_score >= MIN_COLOR_SCORE && red_score > grn_score * 1.25f) {
            return "RED";
        }

        // 3. Nếu tầng Vàng có màu vàng thực thụ
        if (yel_px >= MIN_COLOR_PX && yel_score >= MIN_COLOR_SCORE && yel_score > grn_score && yel_score > red_score) {
            return "YELLOW";
        }

        // Nếu điểm màu chưa đủ mạnh, chỉ xét độ sáng chênh lệch khi tầng tương ứng có ít nhất một chút màu (không phải chói trắng thuần)
        int b_max = std::max({top_b, mid_b, bot_b});
        int b_second = 0;
        if (b_max == top_b)       b_second = std::max(mid_b, bot_b);
        else if (b_max == mid_b)  b_second = std::max(top_b, bot_b);
        else                      b_second = std::max(top_b, mid_b);

        const int b_diff = b_max - b_second;
        if (b_diff >= TL_MIN_CONTRAST_DIFF && b_max >= TL_MIN_BRIGHTNESS) {
            if (b_max == bot_b && grn_px >= 1) return "GREEN";
            if (b_max == top_b && red_px >= 1) return "RED";
            if (b_max == mid_b && yel_px >= 1) return "YELLOW";
        }

        return "NONE";
    }

    // Người đi bộ (chốt 3 frame không cần area, mất 5 frame mới đi tiếp)
    static constexpr float PED_CONF           = 0.55f;
    static constexpr int   PED_CONFIRM_FRAMES = 3;
    static constexpr int   PED_CLEAR_FRAMES   = 5;
    bool ped_stop_ = false;
    int  ped_seen_ = 0, ped_miss_ = 0;

    // Đèn giao thông (tiến lại gần đèn để nhận biển báo kế bên rồi mới dừng)
    static constexpr float TL_CONF                  = 0.45f;
    static constexpr int   TL_CONFIRM_FRAMES        = 2;
    static constexpr int   TL_MIN_BRIGHTNESS        = 95;   // Độ sáng tối thiểu của chóa đèn khi bật
    static constexpr int   TL_MIN_CONTRAST_DIFF     = 18;   // Chênh lệch tối thiểu giữa chóa sáng nhất và chóa nhì
    static constexpr int   TL_STOP_AREA_WITH_SIGN   = 950;  // Khi đã chốt được biển kế bên đèn
    static constexpr int   TL_STOP_HEIGHT_WITH_SIGN = 40;
    static constexpr int   TL_STOP_AREA_MAX         = 1350; // Khi chưa chốt biển, tiến gần hơn chút nữa mới dừng
    static constexpr int   TL_STOP_HEIGHT_MAX       = 50;
    static constexpr int   TL_LOST_RELEASE_FRAMES   = 20;
    bool tl_stop_ = false;
    bool tl_approaching_red_ = false;
    int  tl_red_cnt_ = 0, tl_green_cnt_ = 0, tl_none_cnt_ = 0, tl_miss_ = 0;
    std::string tl_color_ = "NONE";
    TrafficLightMetrics tl_metrics_;

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

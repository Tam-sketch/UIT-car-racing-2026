#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <deque>
#include <iostream>

namespace v20 {

// ============================================================
// CONFIG
// ============================================================
struct Config {
    // Mask validity
    int   min_rows            = 5;
    float coverage_threshold  = 0.05f;
    
    // Speed
    float max_speed           = 15.0f;
    float min_speed           = 5.0f;
    float curve_scale         = 150.0f;
    float post_pid_gain       = 1.0f; // Set = 1.0 nếu dùng phương trình hồi quy bên ngoài

    // Mask processing
    int   mask_width          = 320;

    // PID params
    float kp                  = 0.32f;
    float ki                  = 0.0f;
    float kd                  = 0.6f;
    float integral_limit      = 2.0f;
    float derivative_alpha    = 0.55f;
    float output_limit        = 22.0f;
    float rate_limit          = 10.0f;
    float deadband            = 1.5f;
    
    // Emergency params
    float emg_output_limit    = 25.0f;
    float emg_rate_limit      = 20.0f;
    float emg_error_px        = 30.0f;
    int   emg_streak_req      = 2;
    int   emg_hold_after_occl = 5;

    // Occlusion params
    float hard_step_px        = 30.0f;
    float err_step_px         = 18.0f;
    float cov_drop            = 0.22f;
    float jagged_px           = 15.0f;
    int   occl_hangover       = 4;
    float far_outlier_px      = 40.0f;
    float far_outlier_near_px = 20.0f;
    float far_spread_px       = 40.0f;
    int   far_window          = 5;

    // Asymmetry params
    float asym_strength       = 0.20f;
    float asym_threshold      = 8.0f;
    float asym_ema            = 0.7f;
    float asym_cap            = 12.0f;
    float asym_decay          = 0.80f;

    // Lead params
    float far_delta_clip      = 25.0f;
    float d_far_clip          = 6.0f;
    float d_far_ema_alpha     = 0.4f;
    float lead_rate_gain      = 0.8f;
};

// ============================================================
// OCCLUSION GATE
// ============================================================
class OcclusionGate {
public:
    OcclusionGate(const Config& c) : c_(c) {}
    
    void reset() {
        cov_ema_ = -1.0f;
        prev_error_ = 0.0f;
        initialized_ = false;
        hangover_left_ = 0;
        far_hist_.clear();
        last_far_outlier_ = false;
        last_far_spread_ = 0.0f;
        last_hard_step_ = false;
    }

    bool step(float error, float coverage, const std::vector<cv::Point>& row_centers, 
              bool valid, float far_e, float near_e) {
        far_hist_.push_back(far_e);
        if (far_hist_.size() > c_.far_window) far_hist_.pop_front();

        if (far_hist_.size() >= 2) {
            auto minmax = std::minmax_element(far_hist_.begin(), far_hist_.end());
            last_far_spread_ = *minmax.second - *minmax.first;
        } else {
            last_far_spread_ = 0.0f;
        }

        if (!initialized_) {
            cov_ema_ = coverage;
            prev_error_ = error;
            initialized_ = true;
            last_hard_step_ = false;
            hangover_left_ = 0;
            last_far_outlier_ = false;
            return false;
        }

        cov_ema_ = 0.9f * cov_ema_ + 0.1f * coverage;

        if (!valid) {
            prev_error_ = error;
            last_hard_step_ = false;
            hangover_left_ = c_.occl_hangover;
            return true;
        }

        float err_step = std::abs(error - prev_error_);
        float cov_drop = 0.0f;
        if (cov_ema_ > 0.05f) {
            cov_drop = std::max(0.0f, (cov_ema_ - coverage) / cov_ema_);
        }

        // Jaggedness
        float jagged = 0.0f;
        if (row_centers.size() >= 6) {
            float sum_d2 = 0.0f;
            for (size_t i = 2; i < row_centers.size(); ++i) {
                float d2 = (row_centers[i].x - row_centers[i-1].x) - 
                           (row_centers[i-1].x - row_centers[i-2].x);
                sum_d2 += std::abs(d2);
            }
            jagged = sum_d2 / (row_centers.size() - 2);
        }

        bool far_outlier = (std::abs(far_e) > c_.far_outlier_px && 
                            std::abs(near_e) < c_.far_outlier_near_px);
        last_far_outlier_ = far_outlier;

        bool far_spread_trig = (far_hist_.size() >= static_cast<size_t>(c_.far_window) && 
                                last_far_spread_ > c_.far_spread_px);

        int signals = 0;
        if (err_step > c_.err_step_px) signals++;
        if (cov_drop > c_.cov_drop) signals++;
        if (jagged > c_.jagged_px) signals++;
        if (far_outlier) signals++;
        if (far_spread_trig) signals++;

        prev_error_ = error;
        bool hard = err_step > c_.hard_step_px;
        last_hard_step_ = hard;

        bool raw_occl = hard || (signals >= 2);

        if (raw_occl) {
            hangover_left_ = c_.occl_hangover;
            return true;
        }
        if (hangover_left_ > 0) {
            hangover_left_--;
            return true;
        }
        return false;
    }

    bool last_hard_step() const { return last_hard_step_; }
    bool last_far_outlier() const { return last_far_outlier_; }
    float last_far_spread() const { return last_far_spread_; }

private:
    Config c_;
    float cov_ema_ = -1.0f;
    float prev_error_ = 0.0f;
    bool initialized_ = false;
    int hangover_left_ = 0;
    std::deque<float> far_hist_;
    bool last_far_outlier_ = false;
    float last_far_spread_ = 0.0f;
    bool last_hard_step_ = false;
};

// ============================================================
// AUTHORITY MANAGER
// ============================================================
class AuthorityManager {
public:
    AuthorityManager(float out_limit) : out_limit_(out_limit) {}
    
    void reset() {
        sat_streak_ = 0;
        losing_ = false;
    }

    void step(float error, float prev_error, float pid_output) {
        float d_err = error - prev_error;
        float error_pred = error + 0.35f * d_err; // tau = 0.35

        if (std::abs(pid_output) > 0.9f * out_limit_) {
            sat_streak_++;
        } else {
            sat_streak_ = 0;
        }

        if (!losing_) {
            if (sat_streak_ >= 3 && std::abs(error_pred) > std::abs(error) && std::abs(error) > 30.0f) {
                losing_ = true;
            }
        } else {
            if (std::abs(error) < 20.0f || sat_streak_ < 3) {
                losing_ = false;
            }
        }
    }

    bool losing() const { return losing_; }

private:
    float out_limit_;
    int sat_streak_ = 0;
    bool losing_ = false;
};

// ============================================================
// STEERING PID
// ============================================================
class SteeringPID {
public:
    SteeringPID(const Config& c) : c_(c), authority_(c.output_limit) {}

    void reset() {
        integral_ = 0.0f;
        prev_error_ = 0.0f;
        filt_deriv_ = 0.0f;
        prev_out_ = 0.0f;
        first_call_ = true;
        authority_.reset();
        emg_auth_ = false;
        emg_streak_ = 0;
        forced_emer_ = false;
    }

    void softReset() {
        integral_ = 0.0f;
        prev_error_ = 0.0f;
        filt_deriv_ = 0.0f;
        first_call_ = true;
    }

    void setForcedEmergency(bool b) { forced_emer_ = b; }
    bool emergency_authority() const { return emg_auth_; }
    bool authority_losing() const { return authority_.losing(); }

    float step(float error) {
        float kp_eff = c_.kp * (authority_.losing() ? 1.5f : 1.0f);
        float kd_eff = c_.kd * (authority_.losing() ? 0.7f : 1.0f);

        float out_lim, rate_lim;
        if (forced_emer_) {
            emg_streak_ = c_.emg_streak_req;
            out_lim = c_.emg_output_limit;
            rate_lim = c_.emg_rate_limit;
            emg_auth_ = true;
        } else {
            if (std::abs(error) > c_.emg_error_px) emg_streak_++;
            else emg_streak_ = 0;

            if (emg_streak_ >= c_.emg_streak_req) {
                out_lim = c_.emg_output_limit;
                rate_lim = c_.emg_rate_limit;
                emg_auth_ = true;
            } else {
                out_lim = c_.output_limit;
                rate_lim = c_.rate_limit;
                emg_auth_ = false;
            }
        }

        float e = 0.0f;
        if (std::abs(error) > c_.deadband) {
            e = error - (error > 0 ? c_.deadband : -c_.deadband);
        }

        float p_term = kp_eff * e;
        integral_ += e;
        integral_ = std::clamp(integral_, -c_.integral_limit, c_.integral_limit);
        float i_term = c_.ki * integral_;

        float raw_d = first_call_ ? 0.0f : (error - prev_error_);
        filt_deriv_ = c_.derivative_alpha * raw_d + (1.0f - c_.derivative_alpha) * filt_deriv_;
        float d_term = kd_eff * filt_deriv_;

        float raw_out = p_term + i_term + d_term;
        float pid_out = out_lim * std::tanh(raw_out / out_lim);

        if (authority_.losing()) integral_ = 0.0f;
        authority_.step(error, prev_error_, pid_out);

        float delta = std::clamp(pid_out - prev_out_, -rate_lim, rate_lim);
        float out = prev_out_ + delta;
        out = std::clamp(out, -25.0f, 25.0f);

        prev_error_ = error;
        prev_out_ = out;
        first_call_ = false;

        return out;
    }

private:
    Config c_;
    AuthorityManager authority_;
    float integral_ = 0.0f;
    float prev_error_ = 0.0f;
    float filt_deriv_ = 0.0f;
    float prev_out_ = 0.0f;
    bool first_call_ = true;
    bool emg_auth_ = false;
    int emg_streak_ = 0;
    bool forced_emer_ = false;
};

// ============================================================
// LANE FOLLOWER (V20 Core)
// ============================================================
class LaneFollower {
public:
    struct Output {
        float speed;
        float angle;
        bool  valid;
        float near_e;
        float far_e;
        float eff_error;
        bool  occl;
        bool  forced_emer;
        bool  emer_auth;
        bool  losing;
        float coverage;
        float kept_frac;
        bool  branch_candidate;
    };

    explicit LaneFollower(const Config& c = Config()) : c_(c), pid_(c), gate_(c) {
        resetTracking();
    }

    void resetTracking() {
        pid_.reset(); 
        gate_.reset();
        speed_ = c_.min_speed; 
        lastAngle_ = 0; 
        curveSmooth_ = 0; 
        prevValid_ = false;
        prevFarE_ = 0; 
        dFarFilt_ = 0; 
        asymSmooth_ = 0; 
        emerHold_ = 0;
        occlStreak_ = 0;
        cleanStreak_ = 0;
    }

    void softReset() { 
        pid_.softReset(); 
    }

    double speed() const { return speed_; }
    void setSpeed(double s) { speed_ = s; }
    const Config& config() const { return c_; }

    Output step(const cv::Mat& roadMask) {
        Output o;
        o.speed = speed_;
        o.angle = lastAngle_;
        o.valid = false;
        o.near_e = 0; o.far_e = 0; o.eff_error = 0;
        o.occl = false; o.forced_emer = false; o.emer_auth = false; o.losing = false;
        o.coverage = 0; o.kept_frac = 0; o.branch_candidate = false;

        if (roadMask.empty() || roadMask.type() != CV_8UC1) {
            // Invalid mask -> decay speed
            speed_ *= 0.85;
            o.speed = speed_;
            return o;
        }

        // 1. Mask processing (Keep largest component)
        cv::Mat binary, labels, stats, centroids;
        cv::threshold(roadMask, binary, 0, 255, cv::THRESH_BINARY);
        
        int num_labels = cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8, CV_32S);
        cv::Mat largest = cv::Mat::zeros(binary.size(), CV_8UC1);
        float total_px = static_cast<float>(cv::countNonZero(binary));
        float kept_px = 0;

        if (num_labels > 1) {
            int max_area = 0; int max_idx = 0;
            for (int i = 1; i < num_labels; ++i) {
                int area = stats.at<int>(i, cv::CC_STAT_AREA);
                if (area > max_area) { max_area = area; max_idx = i; }
            }
            largest = (labels == max_idx);
            kept_px = static_cast<float>(max_area);
        } else {
            largest = binary;
            kept_px = total_px;
        }

        o.kept_frac = total_px > 0 ? (kept_px / total_px) : 0.0f;
        o.branch_candidate = (o.kept_frac < 0.5f && kept_px > 200); // Threshold 0.5 làm ví dụ

        cv::Mat gray = largest;

        // 2. Centerline
        float near_e, far_e, curve_e;
        std::vector<cv::Point> row_pts;
        bool valid;
        float coverage;
        extractCenterline(gray, curveSmooth_, near_e, far_e, curve_e, row_pts, valid, coverage);
        
        o.valid = valid;
        o.coverage = coverage;
        o.near_e = near_e;
        o.far_e = far_e;

        // 3. Blending
        float far_delta = std::clamp(far_e - near_e, -c_.far_delta_clip, c_.far_delta_clip);
        float d_far_raw = (valid && prevValid_) ? (far_e - prevFarE_) : 0.0f;
        float d_far = std::clamp(d_far_raw, -c_.d_far_clip, c_.d_far_clip);
        dFarFilt_ = c_.d_far_ema_alpha * d_far + (1.0f - c_.d_far_ema_alpha) * dFarFilt_;
        
        float lead = 0.5f * far_delta + c_.lead_rate_gain * dFarFilt_;
        float blended = near_e + lead;
        prevFarE_ = far_e;

        // 4. Occlusion & Asymmetry
        bool occl = gate_.step(blended, coverage, row_pts, valid, far_e, near_e);
        o.occl = occl;

        if (occl) {
            occlStreak_++;
            cleanStreak_ = 0;
            emerHold_ = c_.emg_hold_after_occl;
        } else {
            cleanStreak_++;
            if (occlStreak_ > 0) occlStreak_--;
            if (emerHold_ > 0) emerHold_--;
        }

        float asym_applied = 0.0f;
        if (!occl && valid) {
            float asym_raw = detectAsymmetry(gray);
            asymSmooth_ = c_.asym_ema * asymSmooth_ + (1.0f - c_.asym_ema) * asym_raw;
            if (std::abs(near_e) < 5.0f && std::abs(asymSmooth_) < c_.asym_cap) {
                asymSmooth_ *= c_.asym_decay;
            }
            asymSmooth_ = std::clamp(asymSmooth_, -c_.asym_cap, c_.asym_cap);
        }

        if (std::abs(asymSmooth_) >= c_.asym_threshold) {
            asym_applied = c_.asym_strength * asymSmooth_;
        }

        float eff_error = blended + asym_applied;
        o.eff_error = eff_error;

        // 5. Control
        bool forced_emer = (emerHold_ > 0);
        o.forced_emer = forced_emer;
        pid_.setForcedEmergency(forced_emer);

        if (valid) {
            if (!prevValid_) {
                pid_.softReset();
                gate_.reset();
            }
            float angle = pid_.step(eff_error) * c_.post_pid_gain;
            lastAngle_ = angle;
            o.angle = angle;

            // Speed planning
            float curve_ratio = std::min(std::abs(curveSmooth_) / c_.curve_scale, 1.0f);
            float target_speed = c_.max_speed - (c_.max_speed - c_.min_speed) * curve_ratio;
            
            float err_f = 1.0f;
            if (std::abs(eff_error) > 8.0f) {
                err_f = std::clamp(1.0f - (std::abs(eff_error) - 8.0f)/(35.0f - 8.0f) * 0.7f, 0.3f, 1.0f);
            }

            float auth_f = pid_.authority_losing() ? 0.6f : 1.0f;
            float anom_f = (std::abs(asymSmooth_) > 15.0f) ? 0.5f : 1.0f;
            
            target_speed = target_speed * err_f * std::min(auth_f, anom_f);
            speed_ = 0.7f * speed_ + 0.3f * target_speed;
            o.speed = speed_;
        } else {
            o.angle = lastAngle_;
            o.speed = speed_ * 0.85f;
            speed_ = o.speed;
        }

        o.emer_auth = pid_.emergency_authority();
        o.losing = pid_.authority_losing();
        prevValid_ = valid;
        
        return o;
    }

private:
    void extractCenterline(const cv::Mat& gray, float prev_curve, 
                           float& near_e, float& far_e, float& curve_e, 
                           std::vector<cv::Point>& row_pts, bool& valid, float& coverage) {
        int h = gray.rows, w = gray.cols;
        row_pts.clear();

        for (int y = h - 1; y >= 0; --y) {
            const uint8_t* row = gray.ptr<uint8_t>(y);
            int start = -1, end = -1, max_len = 0;
            int curr_start = -1;
            for (int x = 0; x < w; ++x) {
                if (row[x] > 0) {
                    if (curr_start == -1) curr_start = x;
                    if (x - curr_start + 1 > max_len) {
                        max_len = x - curr_start + 1;
                        start = curr_start;
                        end = x;
                    }
                } else {
                    curr_start = -1;
                }
            }
            if (max_len > 0) {
                row_pts.push_back(cv::Point((start + end) / 2, y));
            }
        }

        coverage = cv::countNonZero(gray) / static_cast<float>(h * w);
        if (row_pts.size() < static_cast<size_t>(c_.min_rows) || coverage < c_.coverage_threshold) {
            valid = false;
            near_e = 0; far_e = 0; curve_e = 0;
            curveSmooth_ = 0;
            return;
        }
        valid = true;

        // Percentiles
        std::vector<int> ys;
        for (auto& p : row_pts) ys.push_back(p.y);
        std::sort(ys.begin(), ys.end());
        int y_bot = ys[static_cast<int>(ys.size() * 0.98f)];
        int y_top = ys[static_cast<int>(ys.size() * 0.02f)];
        int span = std::max(y_bot - y_top, 1);
        int y_split_near = y_bot - static_cast<int>(span * 0.45f);
        int y_split_far  = y_top + static_cast<int>(span * 0.35f);

        std::vector<int> near_xs, far_xs;
        for (auto& p : row_pts) {
            if (p.y >= y_split_near) near_xs.push_back(p.x);
            if (p.y <= y_split_far)  far_xs.push_back(p.x);
        }

        auto median = [](std::vector<int>& v) {
            if (v.empty()) return -1;
            std::sort(v.begin(), v.end());
            return v[v.size() / 2];
        };

        int near_cx = median(near_xs);
        int far_cx  = median(far_xs);

        near_e = (near_cx != -1) ? (near_cx - w / 2) : 0;
        far_e  = (far_cx != -1)  ? (far_cx - w / 2)  : 0;

        int top_cx = median(far_xs);
        int bot_cx = median(near_xs);
        curve_e = (top_cx != -1 && bot_cx != -1) ? (top_cx - bot_cx) : 0.0f;
        curveSmooth_ = 0.90f * prev_curve + 0.10f * curve_e;
    }

    float detectAsymmetry(const cv::Mat& gray) {
        int h = gray.rows, w = gray.cols;
        std::vector<int> bottom_cxs;
        for (int y = static_cast<int>(h * 0.80f); y < h; y += 4) {
            const uint8_t* row = gray.ptr<uint8_t>(y);
            int first = -1, last = -1;
            for (int x = 0; x < w; ++x) {
                if (row[x] > 0) { if (first == -1) first = x; last = x; }
            }
            if (first != -1 && last != first) bottom_cxs.push_back((first + last) / 2);
        }
        if (bottom_cxs.empty()) return 0.0f;

        std::sort(bottom_cxs.begin(), bottom_cxs.end());
        int ref_cx = bottom_cxs[bottom_cxs.size() / 2];

        std::vector<float> lws, rws;
        int y0 = static_cast<int>(h * 0.18f), y1 = static_cast<int>(h * 0.60f);
        for (int y = y0; y < y1; y += 4) {
            const uint8_t* row = gray.ptr<uint8_t>(y);
            int first = -1, last = -1;
            for (int x = 0; x < w; ++x) {
                if (row[x] > 0) { if (first == -1) first = x; last = x; }
            }
            if (first != -1 && last != first) {
                float lw = ref_cx - first;
                float rw = last - ref_cx;
                if (lw > 0 && rw > 0) { lws.push_back(lw); rws.push_back(rw); }
            }
        }
        if (lws.empty()) return 0.0f;

        std::sort(lws.begin(), lws.end());
        std::sort(rws.begin(), rws.end());
        float lw = lws[lws.size() / 2];
        float rw = rws[rws.size() / 2];
        float total = lw + rw;
        if (total < 40.0f) return 0.0f;
        return ((rw - lw) / total) * 45.0f;
    }

    Config c_;
    SteeringPID pid_;
    OcclusionGate gate_;
    
    float speed_;
    float lastAngle_;
    float curveSmooth_;
    bool  prevValid_;
    float prevFarE_;
    float dFarFilt_;
    float asymSmooth_;
    int   emerHold_;
    int   occlStreak_;
    int   cleanStreak_;
};

} // namespace v20
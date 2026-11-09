#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <deque>
#include "config/RobotConfig.h"

namespace v20 {

class LaneFollower {
public:
    struct Output {
        float speed;
        float angle;
        bool valid;
    };

    LaneFollower() { resetTracking(); }

    void resetTracking() {
        speed_ = ucr::config::MIN_SPEED_CMS;
        lastAngle_ = 0.0f;
        curveSmooth_ = 0.0f;
        prevValid_ = false;
        integral_ = 0.0f;
        prevError_ = 0.0f;
    }

    void softReset() {
        integral_ = 0.0f;
        prevError_ = 0.0f;
    }

    Output step(const cv::Mat& roadMask) {
        Output o;
        o.speed = speed_;
        o.angle = lastAngle_;
        o.valid = false;

        if (roadMask.empty()) {
            speed_ *= 0.85f;
            o.speed = speed_;
            return o;
        }

        // 1. Tìm centerline
        int h = roadMask.rows, w = roadMask.cols;
        std::vector<cv::Point> row_pts;
        for (int y = h - 1; y >= 0; --y) {
            const uint8_t* row = roadMask.ptr<uint8_t>(y);
            int start = -1, end = -1, max_len = 0, curr_start = -1;
            for (int x = 0; x < w; ++x) {
                if (row[x] > 0) {
                    if (curr_start == -1) curr_start = x;
                    if (x - curr_start + 1 > max_len) { max_len = x - curr_start + 1; start = curr_start; end = x; }
                } else curr_start = -1;
            }
            if (max_len > 0) row_pts.push_back(cv::Point((start + end) / 2, y));
        }

        float coverage = cv::countNonZero(roadMask) / static_cast<float>(h * w);
        if (row_pts.size() < ucr::config::MIN_ROWS || coverage < ucr::config::COVERAGE_THRESHOLD) {
            speed_ *= 0.85f;
            o.speed = speed_;
            return o;
        }
        
        o.valid = true;

        // 2. Tính near/far error
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
        auto median = [](std::vector<int>& v) { if(v.empty()) return -1; std::sort(v.begin(), v.end()); return v[v.size()/2]; };

        int near_cx = median(near_xs);
        int far_cx  = median(far_xs);
        float near_e = (near_cx != -1) ? (near_cx - w / 2) : 0;
        float far_e  = (far_cx != -1)  ? (far_cx - w / 2)  : 0;

        int top_cx = median(far_xs);
        int bot_cx = median(near_xs);
        float curve_e = (top_cx != -1 && bot_cx != -1) ? (top_cx - bot_cx) : 0.0f;
        curveSmooth_ = 0.90f * curveSmooth_ + 0.10f * curve_e;

        // 3. PID
        float eff_error = near_e + 0.5f * (far_e - near_e);
        
        // Deadband
        if (std::abs(eff_error) < 1.5f) eff_error = 0.0f;
        
        integral_ += eff_error;
        integral_ = std::clamp(integral_, -2.0f, 2.0f);
        float deriv = eff_error - prevError_;
        float raw_angle = 0.32f * eff_error + 0.6f * deriv;
        
        float pid_out = 22.0f * std::tanh(raw_angle / 22.0f);
        float delta = std::clamp(pid_out - lastAngle_, -10.0f, 10.0f); // rate limit
        lastAngle_ += delta;
        lastAngle_ = std::clamp(lastAngle_, -25.0f, 25.0f);
        prevError_ = eff_error;

        o.angle = lastAngle_;

        // 4. Speed Planning
        float curve_ratio = std::min(std::abs(curveSmooth_) / ucr::config::CURVE_SCALE, 1.0f);
        float target_speed = ucr::config::MAX_SPEED_CMS - (ucr::config::MAX_SPEED_CMS - ucr::config::MIN_SPEED_CMS) * curve_ratio;
        float err_f = std::clamp(1.0f - (std::abs(eff_error) - 8.0f) / 27.0f * 0.7f, 0.3f, 1.0f);
        
        target_speed = std::max(target_speed * err_f, 0.0f);
        speed_ = 0.7f * speed_ + 0.3f * target_speed;
        o.speed = speed_;

        prevValid_ = true;
        return o;
    }

private:
    float speed_;
    float lastAngle_;
    float curveSmooth_;
    bool  prevValid_;
    float integral_;
    float prevError_;
};

} // namespace v20
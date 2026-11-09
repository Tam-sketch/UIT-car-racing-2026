#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <utility>
#include <vector>
#include <string>

namespace v21 {

struct Point { int x, y; };

struct SignState {
    bool active = false;
    int side = 0;      // -1: Left, 1: Right, 2: Both, 0: None
    int class_id = -1; // 6: turn_left, 7: turn_right, 0: construction...
};

struct Opening {
    int side;
    int y_start, y_end, y_mid;
    int x_extreme, x_baseline;
};

struct TurnMemory {
    bool active = false;
    int side = 0;
    int turn_width = 0;
    int last_y_mid = 9999;
};

struct Config {
    int  mask_width   = 320;
    bool keep_largest = true;

    // Opening Detection
    int    open_thresh = 15;
    int    min_height  = 10;
    int    smooth_win  = 5;
    int    base_win    = 50;
    int    pull_back   = 3;
    int    corner_radius = 15;
    int    min_turn_dist = 4;
    int    extend_len    = 80;

    // Steering & PID
    double post_pid_gain = 1.0; // Set = 1.0 để dùng hàm hồi quy bên ngoài
    double max_speed = 15.0;    
    double min_speed = 5.0;    

    double asym_strength = 0.20, asym_threshold = 8.0, asym_ema = 0.7;
    double asym_cap = 12.0, asym_decay = 0.80;
    double far_delta_clip = 25.0, d_far_clip = 6.0, d_far_ema_alpha = 0.4, lead_rate_gain = 0.8;

    double hard_step_px = 30.0, err_step_px = 18.0, cov_drop = 0.22, jagged_px = 15.0;
    int    occl_hangover = 4;
    double far_outlier_px = 40.0, far_outlier_near_px = 20.0, far_spread_px = 40.0;
    int    far_window = 5;
    double cov_ema_alpha = 0.9;

    int emergency_streak_req = 2;
    int emg_hold_after_occl  = 5;

    double kp = 0.32, ki = 0.0, kd = 0.6;
    double integral_limit = 2.0, derivative_alpha = 0.55;
    double output_limit = 22.0, rate_limit = 10.0, deadband = 1.5;
    double emergency_output_limit = 25.0, emergency_rate_limit = 20.0, emergency_error_px = 30.0;

    double auth_tau = 0.35, auth_sat_frac = 0.9;
    int    auth_sat_frames = 3;
    double auth_lose_enter_px = 30.0, auth_lose_exit_px = 20.0;

    int    min_rows = 5;
    double coverage_threshold = 0.05;
    double curve_scale = 150.0;
};

namespace detail {
    inline double medianInts(std::vector<int> v) {
        std::sort(v.begin(), v.end());
        size_t n = v.size();
        return (n % 2) ? (double)v[n / 2] : ((double)v[n / 2 - 1] + (double)v[n / 2]) / 2.0;
    }
    inline double sgn(double x) { return (x > 0) - (x < 0); }

    inline std::vector<float> movingAvg(const std::vector<float>& arr, int win) {
        int h = arr.size();
        std::vector<float> out(h, -1.0f);
        int half = win / 2;
        for (int i = 0; i < h; ++i) {
            float sum = 0; int count = 0;
            for (int j = std::max(0, i - half); j <= std::min(h - 1, i + half); ++j) {
                if (arr[j] >= 0) { sum += arr[j]; count++; }
            }
            if (count > 0) out[i] = sum / count;
        }
        return out;
    }

    inline std::vector<float> rollingMedian(const std::vector<float>& arr, int win) {
        int h = arr.size();
        std::vector<float> out(h, -1.0f);
        int half = win / 2;
        for (int i = 0; i < h; ++i) {
            std::vector<float> vals;
            for (int j = std::max(0, i - half); j <= std::min(h - 1, i + half); ++j) {
                if (arr[j] >= 0) vals.push_back(arr[j]);
            }
            if (!vals.empty()) {
                std::sort(vals.begin(), vals.end());
                out[i] = vals[vals.size() / 2];
            }
        }
        return out;
    }

    inline std::vector<Point> quadraticBezier(Point P0, Point P1, Point P2, int n = 10) {
        std::vector<Point> pts;
        for (int i = 0; i <= n; ++i) {
            float t = (float)i / n;
            float x = (1-t)*(1-t)*P0.x + 2*t*(1-t)*P1.x + t*t*P2.x;
            float y = (1-t)*(1-t)*P0.y + 2*t*(1-t)*P1.y + t*t*P2.y;
            pts.push_back({(int)x, (int)y});
        }
        return pts;
    }
}

class OcclusionGate {
public:
    explicit OcclusionGate(const Config& c = Config()) : c_(c) { reset(); }
    void reset() { covEma_ = 0; prevErr_ = 0; init_ = false; hangover_ = 0; farHist_.clear(); farSpread_ = 0; lastHard_ = false; lastFarOutlier_ = false; }
    bool step(double error, double coverage, const std::vector<std::pair<int, int>>& rows, bool valid, double farE, double nearE) {
        farHist_.push_back(farE);
        if ((int)farHist_.size() > c_.far_window) farHist_.pop_front();
        if (farHist_.size() >= 2) { auto mm = std::minmax_element(farHist_.begin(), farHist_.end()); farSpread_ = *mm.second - *mm.first; } 
        else farSpread_ = 0;

        if (!init_) { covEma_ = coverage; prevErr_ = error; init_ = true; hangover_ = 0; lastHard_ = false; lastFarOutlier_ = false; return false; }
        covEma_ = c_.cov_ema_alpha * covEma_ + (1.0 - c_.cov_ema_alpha) * coverage;

        if (!valid) { prevErr_ = error; hangover_ = c_.occl_hangover; lastHard_ = false; return true; }

        double errStep = std::abs(error - prevErr_);
        double covDrop = (covEma_ > 0.05) ? std::max(0.0, (covEma_ - coverage) / covEma_) : 0.0;
        double jagged = 0;
        if (rows.size() >= 6) {
            double sum = 0; int n = 0;
            for (size_t i = 1; i + 1 < rows.size(); i++) {
                sum += std::abs(rows[i + 1].first - 2 * rows[i].first + rows[i - 1].first); n++;
            }
            jagged = n ? sum / n : 0;
        }
        bool farOutlier = std::abs(farE) > c_.far_outlier_px && std::abs(nearE) < c_.far_outlier_near_px;
        lastFarOutlier_ = farOutlier;
        bool farSpreadTrig = (int)farHist_.size() >= c_.far_window && farSpread_ > c_.far_spread_px;

        int sig = 0;
        if (errStep > c_.err_step_px) sig++;
        if (covDrop > c_.cov_drop) sig++;
        if (jagged > c_.jagged_px) sig++;
        if (farOutlier) sig++;
        if (farSpreadTrig) sig++;

        prevErr_ = error;
        bool hard = errStep > c_.hard_step_px;
        lastHard_ = hard;
        if (hard || sig >= 2) { hangover_ = c_.occl_hangover; return true; }
        if (hangover_ > 0) { hangover_--; return true; }
        return false;
    }
    bool lastFarOutlier() const { return lastFarOutlier_; }
    bool lastHardStep() const { return lastHard_; }
    double farSpread() const { return farSpread_; }
private:
    Config c_;
    double covEma_, prevErr_, farSpread_;
    bool init_, lastHard_, lastFarOutlier_;
    int hangover_;
    std::deque<double> farHist_;
};

class AuthorityManager {
public:
    explicit AuthorityManager(const Config& c = Config()) : c_(c) { reset(); }
    void reset() { satStreak_ = 0; losing_ = false; }
    void step(double error, double prevError, double pidOut) {
        double dErr = error - prevError;
        double pred = error + c_.auth_tau * dErr;
        if (std::abs(pidOut) > c_.auth_sat_frac * c_.output_limit) satStreak_++; else satStreak_ = 0;
        if (!losing_) {
            if (satStreak_ >= c_.auth_sat_frames && std::abs(pred) > std::abs(error) && std::abs(error) > c_.auth_lose_enter_px) losing_ = true;
        } else {
            if (std::abs(error) < c_.auth_lose_exit_px || satStreak_ < c_.auth_sat_frames) losing_ = false;
        }
    }
    bool losing() const { return losing_; }
private:
    Config c_;
    int satStreak_;
    bool losing_;
};

class SteeringPID {
public:
    explicit SteeringPID(const Config& c = Config()) : c_(c), auth_(c) { reset(); }
    void reset() { integral_ = prevErr_ = filtD_ = prevOut_ = 0; first_ = true; auth_.reset(); emgAuth_ = false; emgStreak_ = 0; forced_ = false; }
    void softReset() { integral_ = prevErr_ = filtD_ = 0; first_ = true; }
    void setForcedEmergency(bool f) { forced_ = f; }
    bool losing() const { return auth_.losing(); }
    bool emergencyAuthority() const { return emgAuth_; }

    double step(double error) {
        double kpEff = c_.kp * (auth_.losing() ? 1.5 : 1.0);
        double kdEff = c_.kd * (auth_.losing() ? 0.7 : 1.0);
        double outLim, rateLim;
        if (forced_) { emgStreak_ = c_.emergency_streak_req; outLim = c_.emergency_output_limit; rateLim = c_.emergency_rate_limit; emgAuth_ = true; }
        else {
            if (std::abs(error) > c_.emergency_error_px) emgStreak_++; else emgStreak_ = 0;
            if (emgStreak_ >= c_.emergency_streak_req) { outLim = c_.emergency_output_limit; rateLim = c_.emergency_rate_limit; emgAuth_ = true; }
            else { outLim = c_.output_limit; rateLim = c_.rate_limit; emgAuth_ = false; }
        }
        double sg = detail::sgn(error);
        double e = (std::abs(error) < c_.deadband) ? sg * (std::abs(error) - c_.deadband) : error - sg * c_.deadband;
        double p = kpEff * e;
        integral_ = std::clamp(integral_ + e, -c_.integral_limit, c_.integral_limit);
        double i = c_.ki * integral_;
        double rawD = first_ ? 0.0 : (error - prevErr_);
        filtD_ = c_.derivative_alpha * rawD + (1.0 - c_.derivative_alpha) * filtD_;
        double d = kdEff * filtD_;
        double raw = p + i + d;
        double pidOut = outLim * std::tanh(raw / outLim);
        if (auth_.losing()) integral_ = 0;
        auth_.step(error, prevErr_, pidOut);
        double delta = std::clamp(pidOut - prevOut_, -rateLim, rateLim);
        double out = std::clamp(prevOut_ + delta, -25.0, 25.0);
        prevErr_ = error; prevOut_ = out; first_ = false;
        return out;
    }
private:
    Config c_;
    AuthorityManager auth_;
    double integral_, prevErr_, filtD_, prevOut_;
    bool first_, emgAuth_, forced_;
    int emgStreak_;
};

class LaneFollower {
public:
    struct Output {
        double angle = 0, speed = 0;              
        bool valid = false, occl = false, forced_emer = false, emer_auth = false, losing = false;
        double near_e = 0, far_e = 0, eff_error = 0, coverage = 0;
        int proc_width = 0, proc_height = 0;
        std::vector<Point> final_path; 
        std::string state_str = "CENTERLINE";
    };

    explicit LaneFollower(const Config& c = Config()) : c_(c), pid_(c), gate_(c) {
        speed_ = c_.min_speed;
        resetTracking();
    }

    void resetTracking() {
        pid_.reset(); gate_.reset();
        lastAngle_ = 0; curveSmooth_ = 0; prevValid_ = false;
        prevFarE_ = 0; dFarFilt_ = 0; asymSmooth_ = 0; emerHold_ = 0;
        turn_memory_ = TurnMemory();
    }
    void softReset() { pid_.softReset(); }

    Output step(const cv::Mat& rawMask, const SignState& sign) {
        Output o;
        if (rawMask.empty() || rawMask.type() != CV_8UC1) {
            o.state_str = "INVALID MASK";
            return o;
        }

        cv::Mat cleanMask = preprocess(rawMask);
        o.proc_width = cleanMask.cols;
        o.proc_height = cleanMask.rows;

        std::vector<Point> centerline = computeCenterline(cleanMask);
        auto openings = detectOpenings(cleanMask, sign);
        updateTurnMemory(sign, openings);

        std::vector<Point> final_path = centerline;
        o.state_str = "CENTERLINE";
       
        bool has_valid_opening = false;
        Opening target_opening;
       
        if (!openings.empty()) {
            target_opening = openings[0];
            for (const auto& op : openings) if (op.y_mid > target_opening.y_mid) target_opening = op;
            has_valid_opening = true;
        }

        bool should_turn = false;
        if (sign.active) {
            if (has_valid_opening) {
                if (sign.class_id == 6 || sign.class_id == 7 || sign.class_id == 0) should_turn = true;
                else if (sign.class_id == 1 || sign.class_id == 2) {
                    if (!centerline.empty() && centerline.back().y > target_opening.y_mid - 10) should_turn = true;
                }
            } else if (turn_memory_.active) {
                Point top_pt = centerline.empty() ? Point{rawMask.cols/2, rawMask.rows/2} : centerline.back();
                if (top_pt.y > turn_memory_.last_y_mid + 50) {
                    target_opening.side = turn_memory_.side;
                    target_opening.y_mid = top_pt.y;
                    target_opening.x_baseline = top_pt.x;
                    target_opening.x_extreme = top_pt.x + (turn_memory_.side == -1 ? -turn_memory_.turn_width : turn_memory_.turn_width);
                    should_turn = true;
                    o.state_str = "GHOST TURN " + std::string(turn_memory_.side == -1 ? "LEFT" : "RIGHT");
                }
            }
        }

        if (should_turn) {
            final_path = buildPathWithTurn(centerline, cleanMask, target_opening);
            if (o.state_str == "CENTERLINE") o.state_str = "TURN " + std::string(target_opening.side == -1 ? "LEFT" : "RIGHT");
        }

        o.final_path = final_path;

        double lookahead_error = 0.0;
        if (final_path.size() > 5) {
            int look_idx = final_path.size() * 2 / 3; 
            lookahead_error = final_path[look_idx].x - (cleanMask.cols / 2.0);
        } else if (!final_path.empty()) {
            lookahead_error = final_path.back().x - (cleanMask.cols / 2.0);
        }

        bool occl = gate_.step(lookahead_error, 0.5, {}, true, 0, 0); 
        if (occl) emerHold_ = c_.emg_hold_after_occl;
        else if (emerHold_ > 0) emerHold_--;

        bool forcedEmer = emerHold_ > 0;
        pid_.setForcedEmergency(forcedEmer);

        double angle_pre_gain = pid_.step(lookahead_error);
        o.angle = angle_pre_gain * c_.post_pid_gain;

        double target_speed = c_.max_speed;
        if (o.state_str.find("TURN") != std::string::npos) target_speed = c_.min_speed + 3.0;
        
        speed_ = 0.7 * speed_ + 0.3 * target_speed;

        o.speed = speed_;
        o.valid = !final_path.empty();
        o.eff_error = lookahead_error;
        o.occl = occl;
        o.forced_emer = forcedEmer;
        o.emer_auth = pid_.emergencyAuthority();
        o.losing = pid_.losing();
        return o;
    }

private:
    Config c_;
    SteeringPID pid_;
    OcclusionGate gate_;
    TurnMemory turn_memory_;
    double speed_, lastAngle_, curveSmooth_, prevFarE_, dFarFilt_, asymSmooth_;
    bool prevValid_;
    int emerHold_;

    cv::Mat preprocess(const cv::Mat& in) {
        cv::Mat out;
        if (c_.mask_width > 0 && in.cols != c_.mask_width) cv::resize(in, out, cv::Size(c_.mask_width, in.rows * c_.mask_width / in.cols), 0, 0, cv::INTER_NEAREST);
        else out = in.clone();
        
        cv::Mat binary = (out > 0);
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(35, 5));
        cv::morphologyEx(binary, out, cv::MORPH_CLOSE, kernel);
        return out;
    }

    std::vector<Point> computeCenterline(const cv::Mat& mask) {
        std::vector<Point> traj;
        int h = mask.rows, w = mask.cols;
        int cx = w / 2, cy = h - 1;

        for (int y = h - 1; y > h / 2; --y) {
            std::vector<int> xs;
            const uchar* row = mask.ptr<uchar>(y);
            for (int x = 0; x < w; ++x) if (row[x]) xs.push_back(x);
            if (!xs.empty()) { cx = (xs.front() + xs.back()) / 2; cy = y; break; }
        }
        traj.push_back({cx, cy});

        for (int y = cy - 3; y > h / 3; y -= 3) {
            const uchar* row = mask.ptr<uchar>(y);
            int best_seg_start = -1, best_seg_end = -1, min_dist = 9999, start = -1;
            for (int x = 0; x <= w; ++x) {
                bool on = (x < w) && row[x];
                if (on && start < 0) start = x;
                if (!on && start >= 0) {
                    int dist = (cx >= start && cx <= x - 1) ? 0 : std::min(std::abs(start - cx), std::abs(x - 1 - cx));
                    if (dist < min_dist) { min_dist = dist; best_seg_start = start; best_seg_end = x - 1; }
                    start = -1;
                }
            }
            if (best_seg_start < 0) break;
            int mid = (best_seg_start + best_seg_end) / 2;
            if (std::abs(mid - cx) > 50) mid = cx + (mid > cx ? 50 : -50);
            cx = mid;
            traj.push_back({cx, y});
        }
        return traj;
    }

    std::vector<Opening> detectOpenings(const cv::Mat& mask, const SignState& sign) {
        std::vector<Opening> openings;
        if (!sign.active || sign.side == 0) return openings;

        int h = mask.rows, w = mask.cols;
        std::vector<float> xl(h, -1), xr(h, -1);
       
        for (int y = 0; y < h; ++y) {
            const uchar* row = mask.ptr<uchar>(y);
            int first = -1, last = -1;
            for (int x = 0; x < w; ++x) if (row[x]) { if (first < 0) first = x; last = x; }
            if (first >= 0) { xl[y] = first; xr[y] = last; }
        }

        auto xl_s = detail::movingAvg(xl, c_.smooth_win);
        auto xr_s = detail::movingAvg(xr, c_.smooth_win);
        auto bl = detail::rollingMedian(xl_s, c_.base_win);
        auto br = detail::rollingMedian(xr_s, c_.base_win);

        auto findPeaks = [&](const std::vector<float>& dev, const std::vector<float>& edges, const std::vector<float>& base, int side) {
            bool in_open = false; int y_start = 0;
            for (int y = 0; y < h; ++y) {
                if (dev[y] < 0) {
                    if (in_open && y - y_start >= c_.min_height) {
                        int y_mid = (y_start + y - 1) / 2;
                        openings.push_back({side, y_start, y - 1, y_mid, (int)edges[y_mid], (int)base[y_mid]});
                    }
                    in_open = false; continue;
                }
                if (dev[y] > c_.open_thresh && !in_open) { in_open = true; y_start = y; }
                else if (dev[y] <= c_.open_thresh && in_open) {
                    if (y - y_start >= c_.min_height) {
                        int y_mid = (y_start + y - 1) / 2;
                        openings.push_back({side, y_start, y - 1, y_mid, (int)edges[y_mid], (int)base[y_mid]});
                    }
                    in_open = false;
                }
            }
        };

        std::vector<float> dev_l(h, -1), dev_r(h, -1);
        for (int y = 0; y < h; ++y) {
            if (xl_s[y] >= 0 && bl[y] >= 0) dev_l[y] = bl[y] - xl_s[y];
            if (xr_s[y] >= 0 && br[y] >= 0) dev_r[y] = xr_s[y] - br[y];
        }

        if (sign.side == -1 || sign.side == 2) findPeaks(dev_l, xl_s, bl, -1);
        if (sign.side == 1  || sign.side == 2) findPeaks(dev_r, xr_s, br, 1);

        return openings;
    }

    void updateTurnMemory(const SignState& sign, const std::vector<Opening>& openings) {
        if (!sign.active) { turn_memory_.active = false; return; }
        if (!openings.empty()) {
            Opening best = openings[0];
            for (const auto& op : openings) if (op.y_mid > best.y_mid) best = op;
            turn_memory_.active = true;
            turn_memory_.side = best.side;
            turn_memory_.turn_width = std::abs(best.x_extreme - best.x_baseline);
            turn_memory_.last_y_mid = std::min(turn_memory_.last_y_mid, best.y_mid);
        }
    }

    std::vector<Point> buildPathWithTurn(const std::vector<Point>& centerline, const cv::Mat& mask, const Opening& opening) {
        if (centerline.empty()) return centerline;
        int x_turn = (opening.x_extreme + opening.x_baseline) / 2;
        int y_turn_raw = opening.y_mid;
        int idx = 0, min_d = 9999;
        for (size_t i = 0; i < centerline.size(); ++i) {
            int d = std::abs(centerline[i].y - y_turn_raw);
            if (d < min_d) { min_d = d; idx = i; }
        }
        int x_c = centerline[idx].x, y_c = centerline[idx].y;
        int dist = std::abs(x_turn - x_c);
        if (dist < c_.min_turn_dist) return centerline;

        int sign = (x_turn < x_c) ? -1 : 1;
        int d_bez = std::min(dist, c_.corner_radius * 2);
        int A_y = y_c + d_bez, A_idx = idx;
        for (int i = idx; i >= 0; --i) if (centerline[i].y >= A_y) { A_idx = i; break; }

        Point A = centerline[A_idx];
        Point B = {x_c + sign * d_bez, y_c};
        Point ctrl = {x_c, y_c};
        auto bez = detail::quadraticBezier(A, ctrl, B, 10);

        std::vector<Point> path(centerline.begin(), centerline.begin() + A_idx + 1);
        path.insert(path.end(), bez.begin(), bez.end());
       
        int step = 5, x_curr = x_turn, y_curr = y_c;
        for (int i = 0; i < c_.extend_len / step; ++i) {
            x_curr += sign * step;
            if (x_curr < 0 || x_curr >= mask.cols) break;
            if (mask.at<uchar>(y_curr, x_curr) == 0) break;
            path.push_back({x_curr, y_curr});
        }
        return path;
    }
};

} // namespace v21
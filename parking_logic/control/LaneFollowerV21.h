// ============================================================
// control/LaneFollowerV21.h
// ============================================================
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <string>
#include <utility>
#include <vector>

namespace v21 {

struct SignState {
    bool active = false;
    int  side   = 0;
    int  class_id = -1;
};

struct Point { int x, y; };

struct Opening {
    int side;
    int y_start, y_end, y_mid;
    int x_extreme, x_baseline;
};

struct TurnMemory {
    bool active     = false;
    int  side       = 0;
    int  turn_width = 0;
    int  last_y_mid = 9999;
};

struct Config {
    int    mask_width = 320; bool keep_largest = true;
    double post_pid_gain = 1.0, max_speed = 15.0, min_speed = 5.0;
    double asym_strength = 0.20, asym_threshold = 8.0, asym_ema = 0.7;
    double asym_cap = 12.0, asym_decay = 0.80;
    double far_delta_clip = 25.0, d_far_clip = 6.0;
    double d_far_ema_alpha = 0.4, lead_rate_gain = 0.8;
    double hard_step_px = 30.0, err_step_px = 18.0;
    double cov_drop = 0.22, jagged_px = 15.0;
    int    occl_hangover = 4;
    double far_outlier_px = 40.0, far_outlier_near_px = 20.0, far_spread_px = 40.0;
    int    far_window = 5;
    double cov_ema_alpha = 0.9;
    int    emergency_streak_req = 2; int emg_hold_after_occl = 5;
    double kp = 0.32, ki = 0.0, kd = 0.6;
    double integral_limit = 2.0, derivative_alpha = 0.55;
    double output_limit = 22.0, rate_limit = 10.0, deadband = 1.5;
    double emergency_output_limit = 25.0, emergency_rate_limit = 20.0, emergency_error_px = 30.0;
    double auth_tau = 0.35, auth_sat_frac = 0.9; int auth_sat_frames = 3;
    double auth_lose_enter_px = 30.0, auth_lose_exit_px = 20.0;
    int    min_rows = 5;
    double coverage_threshold = 0.05, curve_scale = 150.0;
    int open_thresh = 15, min_height = 10, smooth_win = 5, base_win = 50;
    int pull_back = 3, corner_radius = 40, min_turn_dist = 4, extend_len = 80;
    int   turn_min_ticks = 25, turn_max_ticks = 100, turn_straight_near_px = 15;
    int   turn_straight_far_px = 20, turn_straight_ticks = 5, turn_synth_width_px = 60;
    float turn_complete_deg = 60.0f, turn_overshoot_deg = 100.0f;  
    double turn_lookahead_y_ratio = 0.30, turn_near_y_ratio = 0.85;   
    double turn_far_weight = 0.85; int turn_ramp_ticks = 8; double turn_error_cap = 60.0;
    double turn_b_lower_ratio = 0.30;
};

namespace detail {
inline double medianInts(std::vector<int> v) { std::sort(v.begin(), v.end()); size_t n = v.size(); return (n % 2) ? (double)v[n / 2] : ((double)v[n / 2 - 1] + (double)v[n / 2]) / 2.0; }
inline int percentileInt(const std::vector<int>& sorted, double q) { double pos = q * (double)(sorted.size() - 1); size_t lo = (size_t)pos; double fr = pos - (double)lo; return (int)((double)sorted[lo] + (lo + 1 < sorted.size() ? fr * (double)(sorted[lo + 1] - sorted[lo]) : 0.0)); }
inline double sgn(double x) { return (x > 0) - (x < 0); }
inline double planSpeed(double curveSmooth, double maxSp, double minSp, double scale) { double ratio = std::min(std::abs(curveSmooth) / scale, 1.0); return maxSp - (maxSp - minSp) * ratio; }
inline double errorSpeedFactor(double error, double startPx = 8.0, double fullPx = 35.0, double minFactor = 0.30) { double a = std::abs(error); if (a <= startPx) return 1.0; if (a >= fullPx) return minFactor; double t = (a - startPx) / (fullPx - startPx); return 1.0 - t * (1.0 - minFactor); }
inline cv::Mat preprocessMask(const cv::Mat& in, const Config& c, double& kept_frac_out) {
    kept_frac_out = 1.0; if (in.empty() || in.type() != CV_8UC1) return cv::Mat();
    cv::Mat out; if (c.mask_width > 0 && in.cols != c.mask_width) cv::resize(in, out, cv::Size(c.mask_width, in.rows * c.mask_width / in.cols), 0, 0, cv::INTER_NEAREST); else out = in.clone();
    cv::Mat binary = (out > 0); cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(35, 5)); cv::morphologyEx(binary, out, cv::MORPH_CLOSE, kernel);
    if (!c.keep_largest) return out;
    std::vector<int> labels((size_t)out.rows * out.cols, 0); std::vector<int> stack, areas(1, 0); int nLabels = 0;
    for (int y = 0; y < out.rows; y++) { const uchar* row = out.ptr<uchar>(y); for (int x = 0; x < out.cols; x++) { if (!row[x] || labels[(size_t)y * out.cols + x]) continue; ++nLabels; int area = 0; stack.clear(); stack.push_back(y * out.cols + x); labels[(size_t)y * out.cols + x] = nLabels; while (!stack.empty()) { int p = stack.back(); stack.pop_back(); area++; int py = p / out.cols, px = p % out.cols; for (int dy = -1; dy <= 1; dy++) { int ny = py + dy; if (ny < 0 || ny >= out.rows) continue; const uchar* nrow = out.ptr<uchar>(ny); for (int dx = -1; dx <= 1; dx++) { if (!dx && !dy) continue; int nx = px + dx; if (nx < 0 || nx >= out.cols) continue; size_t q = (size_t)ny * out.cols + nx; if (nrow[nx] && !labels[q]) { labels[q] = nLabels; stack.push_back((int)q); } } } } areas.push_back(area); } }
    if (nLabels <= 1) return out;
    int best = 1, total_area = 0; for (int i = 1; i <= nLabels; i++) total_area += areas[i]; for (int i = 2; i <= nLabels; i++) if (areas[i] > areas[best]) best = i;
    kept_frac_out = (total_area > 0) ? (double)areas[best] / (double)total_area : 1.0;
    for (int y = 0; y < out.rows; y++) { uchar* row = out.ptr<uchar>(y); for (int x = 0; x < out.cols; x++) if (labels[(size_t)y * out.cols + x] != best) row[x] = 0; }
    return out;
}
inline std::vector<float> movingAvg(const std::vector<float>& arr, int win) { int h = (int)arr.size(); std::vector<float> out(h, -1.0f); int half = win / 2; for (int i = 0; i < h; ++i) { float sum = 0; int count = 0; for (int j = std::max(0, i - half); j <= std::min(h - 1, i + half); ++j) if (arr[j] >= 0) { sum += arr[j]; count++; } if (count > 0) out[i] = sum / count; } return out; }
inline std::vector<float> rollingMedian(const std::vector<float>& arr, int win) { int h = (int)arr.size(); std::vector<float> out(h, -1.0f); int half = win / 2; for (int i = 0; i < h; ++i) { std::vector<float> vals; for (int j = std::max(0, i - half); j <= std::min(h - 1, i + half); ++j) if (arr[j] >= 0) vals.push_back(arr[j]); if (!vals.empty()) { std::sort(vals.begin(), vals.end()); out[i] = vals[vals.size() / 2]; } } return out; }
inline std::vector<Point> quadraticBezier(Point P0, Point P1, Point P2, int n = 10) { std::vector<Point> pts; for (int i = 0; i <= n; ++i) { float t = (float)i / n; float x = (1-t)*(1-t)*P0.x + 2*t*(1-t)*P1.x + t*t*P2.x; float y = (1-t)*(1-t)*P0.y + 2*t*(1-t)*P1.y + t*t*P2.y; pts.push_back({(int)x, (int)y}); } return pts; }
}  // namespace detail

class OcclusionGate {
public:
    explicit OcclusionGate(const Config& c = Config()) : c_(c) { reset(); }
    void reset() { covEma_ = 0; prevErr_ = 0; init_ = false; hangover_ = 0; farHist_.clear(); farSpread_ = 0; lastHard_ = false; lastFarOutlier_ = false; }
    bool step(double error, double coverage, const std::vector<std::pair<int, int>>& rows, bool valid, double farE, double nearE) {
        farHist_.push_back(farE); if ((int)farHist_.size() > c_.far_window) farHist_.pop_front();
        if (farHist_.size() >= 2) { auto mm = std::minmax_element(farHist_.begin(), farHist_.end()); farSpread_ = *mm.second - *mm.first; } else farSpread_ = 0;
        if (!init_) { covEma_ = coverage; prevErr_ = error; init_ = true; hangover_ = 0; lastHard_ = false; lastFarOutlier_ = false; return false; }
        covEma_ = c_.cov_ema_alpha * covEma_ + (1.0 - c_.cov_ema_alpha) * coverage;
        if (!valid) { prevErr_ = error; hangover_ = c_.occl_hangover; lastHard_ = false; return true; }
        double errStep = std::abs(error - prevErr_); double covDrop = (covEma_ > 0.05) ? std::max(0.0, (covEma_ - coverage) / covEma_) : 0.0;
        double jagged = 0; if (rows.size() >= 6) { double sum = 0; int n = 0; for (size_t i = 1; i + 1 < rows.size(); i++) { sum += std::abs(rows[i + 1].first - 2 * rows[i].first + rows[i - 1].first); n++; } jagged = n ? sum / n : 0; }
        bool farOutlier = std::abs(farE) > c_.far_outlier_px && std::abs(nearE) < c_.far_outlier_near_px; lastFarOutlier_ = farOutlier;
        bool farSpreadTrig = (int)farHist_.size() >= c_.far_window && farSpread_ > c_.far_spread_px;
        int sig = 0; if (errStep > c_.err_step_px) sig++; if (covDrop > c_.cov_drop) sig++; if (jagged > c_.jagged_px) sig++; if (farOutlier) sig++; if (farSpreadTrig) sig++;
        prevErr_ = error; bool hard = errStep > c_.hard_step_px; lastHard_ = hard;
        if (hard || sig >= 2) { hangover_ = c_.occl_hangover; return true; }
        if (hangover_ > 0) { hangover_--; return true; } return false;
    }
    bool lastFarOutlier() const { return lastFarOutlier_; } bool lastHardStep() const { return lastHard_; } double farSpread() const { return farSpread_; }
private:
    Config c_; double covEma_, prevErr_, farSpread_; bool init_, lastHard_, lastFarOutlier_; int hangover_; std::deque<double> farHist_;
};

class AuthorityManager {
public:
    explicit AuthorityManager(const Config& c = Config()) : c_(c) { reset(); }
    void reset() { satStreak_ = 0; losing_ = false; }
    void step(double error, double prevError, double pidOut) {
        double dErr = error - prevError; double pred = error + c_.auth_tau * dErr;
        if (std::abs(pidOut) > c_.auth_sat_frac * c_.output_limit) satStreak_++; else satStreak_ = 0;
        if (!losing_) { if (satStreak_ >= c_.auth_sat_frames && std::abs(pred) > std::abs(error) && std::abs(error) > c_.auth_lose_enter_px) losing_ = true; }
        else { if (std::abs(error) < c_.auth_lose_exit_px || satStreak_ < c_.auth_sat_frames) losing_ = false; }
    }
    bool losing() const { return losing_; }
private:
    Config c_; int satStreak_; bool losing_;
};

class SteeringPID {
public:
    explicit SteeringPID(const Config& c = Config()) : c_(c), auth_(c) { reset(); }
    void reset() { integral_ = prevErr_ = filtD_ = prevOut_ = 0; first_ = true; auth_.reset(); emgAuth_ = false; emgStreak_ = 0; forced_ = false; }
    void softReset() { integral_ = prevErr_ = filtD_ = 0; first_ = true; }
    void setForcedEmergency(bool f) { forced_ = f; }
    bool losing() const { return auth_.losing(); } bool emergencyAuthority() const { return emgAuth_; }
    double step(double error) {
        double kpEff = c_.kp * (auth_.losing() ? 1.5 : 1.0); double kdEff = c_.kd * (auth_.losing() ? 0.7 : 1.0);
        double outLim, rateLim;
        if (forced_) { emgStreak_ = c_.emergency_streak_req; outLim = c_.emergency_output_limit; rateLim = c_.emergency_rate_limit; emgAuth_ = true; }
        else {
            if (std::abs(error) > c_.emergency_error_px) emgStreak_++; else emgStreak_ = 0;
            if (emgStreak_ >= c_.emergency_streak_req) { outLim = c_.emergency_output_limit; rateLim = c_.emergency_rate_limit; emgAuth_ = true; }
            else { outLim = c_.output_limit; rateLim = c_.rate_limit; emgAuth_ = false; }
        }
        double sg = detail::sgn(error); double e = (std::abs(error) < c_.deadband) ? sg * (std::abs(error) - c_.deadband) : error - sg * c_.deadband;
        double p = kpEff * e; integral_ = std::clamp(integral_ + e, -c_.integral_limit, c_.integral_limit); double i = c_.ki * integral_;
        double rawD = first_ ? 0.0 : (error - prevErr_); filtD_ = c_.derivative_alpha * rawD + (1.0 - c_.derivative_alpha) * filtD_; double d = kdEff * filtD_;
        double raw = p + i + d; double pidOut = outLim * std::tanh(raw / outLim);
        if (auth_.losing()) integral_ = 0; auth_.step(error, prevErr_, pidOut);
        double delta = std::clamp(pidOut - prevOut_, -rateLim, rateLim); double out = std::clamp(prevOut_ + delta, -25.0, 25.0);
        prevErr_ = error; prevOut_ = out; first_ = false; return out;
    }
private:
    Config c_; AuthorityManager auth_; double integral_, prevErr_, filtD_, prevOut_; bool first_, emgAuth_, forced_; int emgStreak_;
};

class LaneFollower {
public:
    struct Output {
        double angle = 0, speed = 0;
        bool valid = false, occl = false, forced_emer = false;
        bool emer_auth = false, losing = false;
        double near_e = 0, far_e = 0, eff_error = 0, coverage = 0;
        double angle_pre_gain = 0, kept_frac = 1.0;
        std::vector<Point> final_path;
        int proc_width = 0, proc_height = 0;
        std::string state_str = "CENTERLINE";
        
        bool dbg_left_blocked = false;
        bool dbg_right_blocked = false;
        bool dbg_left_free = false;
        bool dbg_right_free = false;
        int  dbg_park_side = 0;
        float dbg_heading_base = 0.0f;
        float dbg_heading_target = 0.0f;
    };

    explicit LaneFollower(const Config& c = Config()) : c_(c), pid_(c), gate_(c) { speed_ = c_.min_speed; resetTracking(); }

    void resetTracking() {
        pid_.reset(); gate_.reset(); lastAngle_ = 0; curveSmooth_ = 0; prevValid_ = false;
        prevFarE_ = 0; dFarFilt_ = 0; asymSmooth_ = 0; emerHold_ = 0;
        turn_memory_ = TurnMemory(); turn_committed_ = false; turn_committed_side_ = 0; turn_tick_ = 0; turn_straight_streak_ = 0;
        park_state_ = ParkState::NONE; park_side_ = 0; heading_0_ = 0; park_tick_ = 0;
        left_blocked_cnt_ = 0; right_blocked_cnt_ = 0;
        left_free_cnt_ = 0; right_free_cnt_ = 0;
        dbg_left_blocked_ = false; dbg_right_blocked_ = false;
        dbg_left_free_ = false; dbg_right_free_ = false;
        park_sign_lost_ = false;
    }

    void softReset() { pid_.softReset(); }

    Output step(const cv::Mat& rawMask, const SignState& sign, float tof_left = -1, float tof_right = -1, float current_heading = 0.0f) {
        Output o;
        double kept_frac = 1.0;
        cv::Mat cleanMask = detail::preprocessMask(rawMask, c_, kept_frac);
        if (cleanMask.empty()) { o.state_str = "INVALID MASK"; return o; }
        o.proc_width = cleanMask.cols; o.proc_height = cleanMask.rows;

        std::vector<std::pair<int, int>> row_centers;
        auto cl = extractCenterline(cleanMask, curveSmooth_, c_, row_centers);
        curveSmooth_ = cl.curve_smooth;

        double farDelta = std::clamp(cl.far_error - cl.near_error, -c_.far_delta_clip, c_.far_delta_clip);
        double dFarRaw = (cl.valid && prevValid_) ? (cl.far_error - prevFarE_) : 0.0;
        double dFar = std::clamp(dFarRaw, -c_.d_far_clip, c_.d_far_clip);
        dFarFilt_ = c_.d_far_ema_alpha * dFar + (1.0 - c_.d_far_ema_alpha) * dFarFilt_;
        double lead = 0.5 * farDelta + c_.lead_rate_gain * dFarFilt_;
        double blended = cl.near_error + lead; prevFarE_ = cl.far_error;

        bool occl = gate_.step(blended, cl.coverage, row_centers, cl.valid, cl.far_error, cl.near_error);
        if (occl) emerHold_ = c_.emg_hold_after_occl; else if (emerHold_ > 0) emerHold_--;

        if (!occl && cl.valid) {
            double asymRaw = detectAsymmetry(cleanMask);
            asymSmooth_ = c_.asym_ema * asymSmooth_ + (1.0 - c_.asym_ema) * asymRaw;
            if (std::abs(cl.near_error) < 5.0 && std::abs(asymSmooth_) < c_.asym_cap) asymSmooth_ *= c_.asym_decay;
            asymSmooth_ = std::clamp(asymSmooth_, -c_.asym_cap, c_.asym_cap);
        }
        double asymApplied = (std::abs(asymSmooth_) < c_.asym_threshold) ? 0.0 : c_.asym_strength * asymSmooth_;
        double eff_error_v20 = blended + asymApplied;

        std::vector<Point> centerline_pts;
        for (auto& p : row_centers) centerline_pts.push_back({p.first, p.second});

        auto openings = detectOpenings(cleanMask, sign);
        updateTurnMemory(sign, openings);

        std::vector<Point> final_path = centerline_pts;
        o.state_str = "CENTERLINE";
        bool should_turn = false;
        Opening target_opening{};

        bool has_valid_opening = false;
        if (!openings.empty()) {
            target_opening = openings[0];
            for (const auto& op : openings) if (op.y_mid > target_opening.y_mid) target_opening = op;
            has_valid_opening = true;
        }

        // --- TÍCH HỢP PARKING OVERRIDE ---
        if (sign.active && sign.class_id == 3) {
            if (park_state_ == ParkState::NONE) {
                park_state_ = ParkState::APPROACH; park_tick_ = 0;
                left_blocked_cnt_ = 0; right_blocked_cnt_ = 0;
                left_free_cnt_ = 0; right_free_cnt_ = 0;
                park_sign_lost_ = false;
            } else if (park_state_ >= ParkState::SCAN_SLOT && park_state_ <= ParkState::DONE) {
                // RESET nếu đang trong quá trình đỗ (từ State 2-7) mà lại thấy biển
                park_state_ = ParkState::APPROACH; park_tick_ = 0;
                left_blocked_cnt_ = 0; right_blocked_cnt_ = 0;
                left_free_cnt_ = 0; right_free_cnt_ = 0;
                park_sign_lost_ = false;
            }
        }
        
        if (park_state_ != ParkState::NONE) {
            double park_angle = 0, park_speed = 0;
            std::string park_state_str = "PARK: INIT";
            processParking(tof_left, tof_right, current_heading, park_angle, park_speed, park_state_str);
            
            o.angle = park_angle; o.speed = park_speed; o.state_str = park_state_str;
            
            o.dbg_left_blocked = dbg_left_blocked_;
            o.dbg_right_blocked = dbg_right_blocked_;
            o.dbg_left_free = dbg_left_free_;
            o.dbg_right_free = dbg_right_free_;
            o.dbg_park_side = park_side_;
            o.dbg_heading_base = heading_0_;
            o.dbg_heading_target = dbg_target_heading_;

            if (park_state_ == ParkState::REVERSE_1 || 
                park_state_ == ParkState::REVERSE_2 || 
                park_state_ == ParkState::DONE) {
                return o; 
            }
        }

        if (sign.active) {
            if (has_valid_opening) {
                if (sign.class_id == 6 || sign.class_id == 7 || sign.class_id == 0) should_turn = true;
                else if (sign.class_id == 1 || sign.class_id == 2) {
                    if (!centerline_pts.empty() && centerline_pts.back().y > target_opening.y_mid - 10) should_turn = true;
                }
            } else if (turn_memory_.active) {
                Point top_pt = centerline_pts.empty() ? Point{cleanMask.cols/2, cleanMask.rows/2} : centerline_pts.back();
                if (top_pt.y > turn_memory_.last_y_mid + 50) {
                    target_opening.side = turn_memory_.side; target_opening.y_mid = top_pt.y;
                    target_opening.x_baseline = top_pt.x; target_opening.x_extreme = top_pt.x + (turn_memory_.side == -1 ? -turn_memory_.turn_width : turn_memory_.turn_width);
                    should_turn = true; o.state_str = "GHOST TURN " + std::string(turn_memory_.side == -1 ? "LEFT" : "RIGHT");
                }
            }
        }

        if (should_turn && !turn_committed_) { turn_committed_ = true; turn_committed_side_ = target_opening.side; turn_tick_ = 0; turn_straight_streak_ = 0; }
        if (turn_committed_) {
            ++turn_tick_;
            const bool road_is_straight = std::abs(cl.near_error) < c_.turn_straight_near_px && std::abs(cl.far_error - cl.near_error) < c_.turn_straight_far_px;
            if (road_is_straight) ++turn_straight_streak_; else turn_straight_streak_ = 0;
            const bool release = turn_tick_ >= c_.turn_max_ticks || (turn_tick_ >= c_.turn_min_ticks && turn_straight_streak_ >= c_.turn_straight_ticks);
            if (release) { turn_committed_ = false; turn_straight_streak_ = 0; turn_tick_ = 0; }
            else {
                should_turn = true;
                if (!has_valid_opening) {
                    Point top_pt = centerline_pts.empty() ? Point{cleanMask.cols / 2, cleanMask.rows / 2} : centerline_pts.back();
                    int synthW = (turn_memory_.turn_width > 20) ? turn_memory_.turn_width : c_.turn_synth_width_px;
                    target_opening.side = turn_committed_side_; target_opening.y_mid = top_pt.y;
                    target_opening.x_baseline = top_pt.x; target_opening.x_extreme = top_pt.x + (turn_committed_side_ == -1 ? -synthW : synthW);
                }
            }
        }

        if (should_turn) {
            final_path = buildPathWithTurn(centerline_pts, cleanMask, target_opening);
            if (o.state_str == "CENTERLINE") o.state_str = "TURN " + std::string(target_opening.side == -1 ? "LEFT" : "RIGHT");
        }

        double final_error = eff_error_v20;
        if (should_turn && final_path.size() > 5) {
            auto pathXAtY = [&](int target_y) -> double {
                int best_x = final_path.front().x; int best_dy = 1 << 30;
                for (const auto& p : final_path) { int dy = std::abs(p.y - target_y); if (dy < best_dy) { best_dy = dy; best_x = p.x; } }
                return best_x - cleanMask.cols / 2.0;
            };
            const int near_y = (int)(cleanMask.rows * c_.turn_near_y_ratio);
            const int far_y  = (int)(cleanMask.rows * c_.turn_lookahead_y_ratio);
            const double near_e_path = pathXAtY(near_y); const double far_e_path  = pathXAtY(far_y);
            const double bezier_error = near_e_path + c_.turn_far_weight * (far_e_path - near_e_path);
            const double ramp = std::min(1.0, static_cast<double>(turn_tick_) / c_.turn_ramp_ticks);
            final_error = (1.0 - ramp) * eff_error_v20 + ramp * bezier_error;
            final_error = std::clamp(final_error, -c_.turn_error_cap, +c_.turn_error_cap);
            speed_ = c_.min_speed + 3.0;
        } else {
            double target_speed = detail::planSpeed(curveSmooth_, c_.max_speed, c_.min_speed, c_.curve_scale);
            double errMult = detail::errorSpeedFactor(eff_error_v20); double authSpeed = pid_.losing() ? 0.6 : 1.0;
            target_speed = std::max(target_speed * errMult * authSpeed, 0.0); speed_ = 0.7 * speed_ + 0.3 * target_speed;
        }

        bool forcedEmer = emerHold_ > 0; pid_.setForcedEmergency(forcedEmer);
        double angle_pre_gain = 0.0;
        if (cl.valid) { if (!prevValid_) { pid_.softReset(); gate_.reset(); } angle_pre_gain = pid_.step(final_error); lastAngle_ = angle_pre_gain * c_.post_pid_gain; }
        else { angle_pre_gain = (c_.post_pid_gain > 0) ? lastAngle_ / c_.post_pid_gain : lastAngle_; speed_ = 0.85 * speed_; }
        prevValid_ = cl.valid;

        o.angle = lastAngle_; o.angle_pre_gain = angle_pre_gain; o.kept_frac = kept_frac; o.speed = speed_;
        o.valid = cl.valid; o.occl = occl; o.forced_emer = forcedEmer; o.emer_auth = pid_.emergencyAuthority(); o.losing = pid_.losing();
        o.near_e = cl.near_error; o.far_e = cl.far_error; o.eff_error = final_error; o.coverage = cl.coverage; o.final_path = final_path;
        return o;
    }

private:
    Config c_; SteeringPID pid_; OcclusionGate gate_; TurnMemory turn_memory_;
    double speed_, lastAngle_, curveSmooth_, prevFarE_, dFarFilt_, asymSmooth_; bool prevValid_; int emerHold_;
    bool turn_committed_ = false; int turn_committed_side_ = 0; int turn_tick_ = 0; int turn_straight_streak_ = 0;

    enum class ParkState { NONE, APPROACH, SCAN_SLOT, SLOT_FOUND, WAIT_STOP, REVERSE_1, REVERSE_2, DONE };
    ParkState park_state_ = ParkState::NONE;
    int park_side_ = 0; float heading_0_ = 0; int park_tick_ = 0;
    int left_blocked_cnt_ = 0; int right_blocked_cnt_ = 0;
    int left_free_cnt_ = 0; int right_free_cnt_ = 0;
    
    bool dbg_left_blocked_ = false;
    bool dbg_right_blocked_ = false;
    bool dbg_left_free_ = false;
    bool dbg_right_free_ = false;
    bool park_sign_lost_ = false;
    float dbg_target_heading_ = 0.0f;

    void processParking(float tof_left, float tof_right, float current_heading, double& out_angle, double& out_speed, std::string& state_str) {
        const float TOF_MIN_BLOCK = 50.0f;   
        const float TOF_MAX_BLOCK = 400.0f;  

        // Cập nhật bộ đếm Block/Free với Debounce 5 ticks (0.5s)
        auto update_tof_state = [&](float val, int& block_cnt, int& free_cnt, bool& is_blocked, bool& is_free) {
            if (val >= TOF_MIN_BLOCK && val <= TOF_MAX_BLOCK) {
                block_cnt++; free_cnt = 0;
            } else if (val > TOF_MAX_BLOCK || val == -1.0f) {
                free_cnt++; block_cnt = 0;
            } else {
                // val < 50 và != -1 -> Nhiễu, giữ nguyên bộ đếm
            }
            is_blocked = (block_cnt >= 5);
            is_free = (free_cnt >= 5);
        };

        update_tof_state(tof_left, left_blocked_cnt_, left_free_cnt_, dbg_left_blocked_, dbg_left_free_);
        update_tof_state(tof_right, right_blocked_cnt_, right_free_cnt_, dbg_right_blocked_, dbg_right_free_);

        if (park_state_ == ParkState::APPROACH) {
            park_tick_++;
            state_str = "PARK: APPROACH (STRAIGHT)";
            out_speed = 8.0f; 
            out_angle = 0.0f; // Khóa lái
            
            if (park_tick_ >= 20) { // 2 giây
                park_state_ = ParkState::SCAN_SLOT;
                park_tick_ = 0;
            }
        } 
        else if (park_state_ == ParkState::SCAN_SLOT) {
            state_str = "PARK: SCANNING SLOT";
            out_speed = 8.0f;
            out_angle = 0.0f;
            
            // 1 bên Block, 1 bên Free mới chuyển trạng thái
            if (dbg_left_blocked_ && dbg_right_free_) {
                park_side_ = 1; // Đỗ bên phải
                park_state_ = ParkState::SLOT_FOUND;
                state_str = "PARK: SLOT FOUND";
            } else if (dbg_left_free_ && dbg_right_blocked_) {
                park_side_ = -1; // Đỗ bên trái
                park_state_ = ParkState::SLOT_FOUND;
                state_str = "PARK: SLOT FOUND";
            }
            // Nếu 2 bên cùng Free hoặc cùng Block -> Không làm gì, giữ nguyên State 2
        } 
        else if (park_state_ == ParkState::SLOT_FOUND) {
            state_str = "PARK: MOVING FORWARD";
            out_speed = 8.0f;
            out_angle = 0.0f;
            
            if (dbg_left_blocked_ && dbg_right_blocked_) {
                heading_0_ = current_heading; 
                park_state_ = ParkState::WAIT_STOP; 
                park_tick_ = 0;
            }
        } 
        else if (park_state_ == ParkState::WAIT_STOP) {
            state_str = "PARK: STOPPED";
            out_speed = 0.0f; 
            out_angle = 0.0f;
            park_state_ = ParkState::REVERSE_1; 
            park_tick_ = 0;
        }
        else if (park_state_ == ParkState::REVERSE_1) {
            park_tick_++;
            state_str = "PARK: REVERSE TURN";
            
            // Đánh lái 45 độ
            out_angle = (park_side_ == -1) ? -45.0f : 45.0f; 
            
            if (park_tick_ <= 3) {
                out_speed = 0.0f; // Chờ Servo quay
            } else {
                out_speed = -8.0f; 
                float diff = std::abs(current_heading - heading_0_);
                dbg_target_heading_ = heading_0_ + (park_side_ == -1 ? -45.0f : 45.0f);
                if (diff >= 45.0f) { 
                    park_state_ = ParkState::REVERSE_2; 
                    park_tick_ = 0; 
                }
            }
        }
        else if (park_state_ == ParkState::REVERSE_2) {
            park_tick_++;
            state_str = "PARK: STRAIGHTEN";
            
            // Đánh lái 45 độ ngược lại
            out_angle = (park_side_ == -1) ? 45.0f : -45.0f; 
            
            if (park_tick_ <= 3) {
                out_speed = 0.0f; // Chờ Servo quay
            } else {
                out_speed = -8.0f;
                float diff = std::abs(current_heading - heading_0_);
                dbg_target_heading_ = heading_0_; 
                if (diff <= 3.0f) { 
                    park_state_ = ParkState::DONE; 
                }
            }
        }
        else if (park_state_ == ParkState::DONE) {
            state_str = "PARK: DONE"; 
            out_speed = 0.0f; 
            out_angle = 0.0f; 
            dbg_target_heading_ = heading_0_;
        }
    }

    // (Giữ nguyên các hàm extractCenterline, detectAsymmetry, detectOpenings, updateTurnMemory, buildPathWithTurn)
    struct CenterlineInfo { double near_error = 0, far_error = 0, curve_error = 0, curve_smooth = 0, coverage = 0; bool valid = false; };
    bool largestRun(const uchar* row, int w, int& bs, int& be) { int bestLen = -1, s = -1; bs = be = -1; for (int x = 0; x <= w; x++) { bool on = (x < w) && row[x] > 0; if (on && s < 0) s = x; if (!on && s >= 0) { int len = (x - 1) - s; if (len > bestLen) { bestLen = len; bs = s; be = x - 1; } s = -1; } } return bs >= 0; }
    CenterlineInfo extractCenterline(const cv::Mat& m, double prevCurveSmooth, const Config& c, std::vector<std::pair<int, int>>& row_centers) {
        CenterlineInfo r; const int h = m.rows, w = m.cols; long nz = 0;
        for (int y = h - 1; y >= 0; y--) { const uchar* row = m.ptr<uchar>(y); for (int x = 0; x < w; x++) nz += row[x] > 0; int s, e; if (largestRun(row, w, s, e)) row_centers.push_back({(s + e) / 2, y}); }
        r.coverage = (double)nz / ((double)h * (double)w); if ((int)row_centers.size() < c.min_rows || r.coverage < c.coverage_threshold) { r.valid = false; return r; }
        std::vector<int> ys; for (auto& p : row_centers) ys.push_back(p.second); std::sort(ys.begin(), ys.end());
        int yBot = detail::percentileInt(ys, 0.98); int yTop = detail::percentileInt(ys, 0.02); int span = std::max(yBot - yTop, 1);
        int ySplitNear = yBot - (int)(span * 0.45); int ySplitFar  = yTop + (int)(span * 0.35);
        std::vector<int> nearC, farC; for (auto& p : row_centers) { if (p.second >= ySplitNear) nearC.push_back(p.first); if (p.second <= ySplitFar)  farC.push_back(p.first); }
        int nearCx = nearC.empty() ? -1 : (int)detail::medianInts(nearC); int farCx  = farC.empty()  ? -1 : (int)detail::medianInts(farC);
        r.near_error = (nearCx != -1) ? (nearCx - w / 2) : 0; r.far_error  = (farCx  != -1) ? (farCx  - w / 2) : 0;
        int topCx = farC.empty()  ? 0 : (int)detail::medianInts(farC); int botCx = nearC.empty() ? 0 : (int)detail::medianInts(nearC);
        r.curve_error  = topCx - botCx; r.curve_smooth = 0.90 * prevCurveSmooth + 0.10 * r.curve_error; r.valid = true; return r;
    }
    double detectAsymmetry(const cv::Mat& m, double maxBias = 45.0, double y0f = 0.18, double y1f = 0.60) {
        const int h = m.rows, w = m.cols; auto firstLast = [&](int y, int& a, int& b) { const uchar* row = m.ptr<uchar>(y); int cnt = 0; a = b = -1; for (int x = 0; x < w; x++) if (row[x] > 0) { if (a < 0) a = x; b = x; cnt++; } return cnt >= 2; };
        std::vector<int> bottomCxs; for (int y = (int)(h * 0.80); y < h; y += 4) { int a, b; if (firstLast(y, a, b)) bottomCxs.push_back((a + b) / 2); } if (bottomCxs.empty()) return 0.0;
        int refCx = (int)detail::medianInts(bottomCxs); std::vector<int> lws, rws; for (int y = (int)(h * y0f); y < (int)(h * y1f); y += 4) { int a, b; if (!firstLast(y, a, b)) continue; int lw = refCx - a, rw = b - refCx; if (lw > 0 && rw > 0) { lws.push_back(lw); rws.push_back(rw); } }
        if (lws.empty()) return 0.0; double lw = detail::medianInts(lws), rw = detail::medianInts(rws); double total = lw + rw; if (total < 40) return 0.0; return (rw - lw) / total * maxBias;
    }
    std::vector<Opening> detectOpenings(const cv::Mat& mask, const SignState& sign) {
        std::vector<Opening> openings; if (!sign.active || sign.side == 0) return openings; int h = mask.rows, w = mask.cols;
        std::vector<float> xl(h, -1), xr(h, -1); for (int y = 0; y < h; ++y) { const uchar* row = mask.ptr<uchar>(y); int first = -1, last = -1; for (int x = 0; x < w; ++x) if (row[x]) { if (first < 0) first = x; last = x; } if (first >= 0) { xl[y] = first; xr[y] = last; } }
        auto xl_s = detail::movingAvg(xl, c_.smooth_win); auto xr_s = detail::movingAvg(xr, c_.smooth_win); auto bl = detail::rollingMedian(xl_s, c_.base_win); auto br = detail::rollingMedian(xr_s, c_.base_win);
        auto findPeaks = [&](const std::vector<float>& dev, const std::vector<float>& edges, const std::vector<float>& base, int side) {
            bool in_open = false; int y_start = 0;
            for (int y = 0; y < h; ++y) { if (dev[y] < 0) { if (in_open && y - y_start >= c_.min_height) { int y_mid = (y_start + y - 1) / 2; openings.push_back({side, y_start, y - 1, y_mid, (int)edges[y_mid], (int)base[y_mid]}); } in_open = false; continue; }
                if (dev[y] > c_.open_thresh && !in_open) { in_open = true; y_start = y; } else if (dev[y] <= c_.open_thresh && in_open) { if (y - y_start >= c_.min_height) { int y_mid = (y_start + y - 1) / 2; openings.push_back({side, y_start, y - 1, y_mid, (int)edges[y_mid], (int)base[y_mid]}); } in_open = false; } }
        };
        std::vector<float> dev_l(h, -1), dev_r(h, -1); for (int y = 0; y < h; ++y) { if (xl_s[y] >= 0 && bl[y] >= 0) dev_l[y] = bl[y] - xl_s[y]; if (xr_s[y] >= 0 && br[y] >= 0) dev_r[y] = xr_s[y] - br[y]; }
        if (sign.side == -1 || sign.side == 2) findPeaks(dev_l, xl_s, bl, -1); if (sign.side ==  1 || sign.side == 2) findPeaks(dev_r, xr_s, br, +1); return openings;
    }
    void updateTurnMemory(const SignState& sign, const std::vector<Opening>& openings) {
        if (!sign.active) { turn_memory_.active = false; return; } if (!openings.empty()) { Opening best = openings[0]; for (const auto& op : openings) if (op.y_mid > best.y_mid) best = op; turn_memory_.active = true; turn_memory_.side = best.side; turn_memory_.turn_width = std::abs(best.x_extreme - best.x_baseline); turn_memory_.last_y_mid = std::min(turn_memory_.last_y_mid, best.y_mid); }
    }
    std::vector<Point> buildPathWithTurn(const std::vector<Point>& centerline, const cv::Mat& mask, const Opening& opening) {
        if (centerline.empty()) return centerline; int y_turn = opening.y_start; int idx_G = 0, min_d = 9999;
        for (size_t i = 0; i < centerline.size(); ++i) { int d = std::abs(centerline[i].y - y_turn); if (d < min_d) { min_d = d; idx_G = (int)i; } }
        Point P2 = centerline[idx_G]; int sign = (opening.x_extreme < P2.x) ? -1 : 1; int dist_x = std::abs(opening.x_extreme - P2.x);
        if (dist_x < 15) { std::vector<Point> path(centerline.begin(), centerline.begin() + idx_G + 1); path.push_back({opening.x_extreme, P2.y}); return path; }
        int dist_y = std::abs(P2.y - centerline.front().y); int R = std::min({c_.corner_radius, dist_x / 2, dist_y / 2}); 
        Point A = { P2.x, P2.y + R }; Point B = { P2.x + sign * R, P2.y }; std::vector<Point> path;
        for (int i = 0; i <= idx_G; ++i) { if (centerline[i].y >= A.y) { path.push_back(centerline[i]); } } if (path.empty() || path.back().y != A.y) { path.push_back(A); }
        auto bez = detail::quadraticBezier(A, P2, B, 10); for (size_t i = 1; i < bez.size(); ++i) { path.push_back(bez[i]); }
        int step = 5; int x_curr = B.x; int y_curr = B.y;
        while (true) { x_curr += sign * step; if ((sign == 1 && x_curr >= opening.x_extreme) || (sign == -1 && x_curr <= opening.x_extreme)) { path.push_back({opening.x_extreme, y_curr}); break; }
            if (x_curr >= 0 && x_curr < mask.cols && y_curr >= 0 && y_curr < mask.rows) { if (mask.at<uchar>(y_curr, x_curr) == 0) { path.push_back({x_curr, y_curr}); break; } } path.push_back({x_curr, y_curr}); }
        return path;
    }
};

}  // namespace v21
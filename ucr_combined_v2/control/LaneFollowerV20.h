// ============================================================
// LaneFollowerV20.h
// ============================================================
#pragma once

#ifdef LANEV20_USE_SHIM
#include "cvshim.hpp"
#else
#include <opencv2/core.hpp>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

namespace v20 {

struct Config {
    int  mask_width   = 320;
    bool keep_largest = true;

    double max_speed = 100.0;
    double min_speed = 75.0;

    double asym_strength = 0.20, asym_threshold = 6.0, asym_ema = 0.85;
    double asym_cap = 10.0, asym_decay = 0.92;

    double hard_step_px = 30.0, err_step_px = 18.0, cov_drop = 0.22, jagged_px = 15.0;
    int    occl_hangover = 4;
    double far_outlier_px = 40.0, far_outlier_near_px = 20.0, far_spread_px = 40.0;
    int    far_window = 5;
    double cov_ema_alpha = 0.9;

    int emergency_streak_req = 2;
    int emg_hold_after_occl  = 5;

    double kp = 0.45, ki = 0.0, kd = 0.70; 
    double integral_limit = 2.0, derivative_alpha = 0.30;
    double output_limit = 45.0, rate_limit = 24.0, deadband = 1.0;
    double emergency_output_limit = 25.0, emergency_rate_limit = 25.0, emergency_error_px = 26.0;

    double auth_tau = 0.35, auth_sat_frac = 0.9;
    int    auth_sat_frames = 3;
    double auth_lose_enter_px = 28.0, auth_lose_exit_px = 18.0;

    int    min_rows = 5;
    double coverage_threshold = 0.05;
    double curve_scale = 140.0;
};

namespace detail {
    inline double medianInts(std::vector<int> v) {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        size_t n = v.size();
        return (n % 2) ? (double)v[n / 2] : ((double)v[n / 2 - 1] + (double)v[n / 2]) / 2.0;
    }

    inline int percentileInt(const std::vector<int>& sorted, double q) {
        if (sorted.empty()) return 0;
        double pos = q * (double)(sorted.size() - 1);
        size_t lo = (size_t)pos;
        double fr = pos - (double)lo;
        double v = (double)sorted[lo] + (lo + 1 < sorted.size() ? fr * (double)(sorted[lo + 1] - sorted[lo]) : 0.0);
        return (int)v;
    }

    inline double sgn(double x) { return (x > 0) - (x < 0); }

    inline cv::Mat preprocessMask(const cv::Mat& in, const Config& c) {
        if (in.empty() || in.type() != CV_8UC1) return cv::Mat();
        const int h = in.rows, w = in.cols;
        int outW = w, outH = h;
        if (c.mask_width > 0 && w != c.mask_width) {
            outW = c.mask_width;
            outH = std::max(1, (int)std::lround((double)h * c.mask_width / (double)w));
        }
        cv::Mat out(outH, outW, CV_8UC1);
        for (int y = 0; y < outH; y++) {
            int sy = std::min(h - 1, (int)((y + 0.5) * h / outH));
            const uchar* src = in.ptr<uchar>(sy);
            uchar* dst = out.ptr<uchar>(y);
            for (int x = 0; x < outW; x++) {
                int sx = std::min(w - 1, (int)((x + 0.5) * w / outW));
                dst[x] = src[sx] ? 255 : 0;
            }
        }
        if (!c.keep_largest) return out;

        std::vector<int> labels((size_t)outH * outW, 0);
        std::vector<int> stack, areas(1, 0);
        int nLabels = 0;
        for (int y = 0; y < outH; y++) {
            const uchar* row = out.ptr<uchar>(y);
            for (int x = 0; x < outW; x++) {
                if (!row[x] || labels[(size_t)y * outW + x]) continue;
                ++nLabels;
                int area = 0;
                stack.clear();
                stack.push_back(y * outW + x);
                labels[(size_t)y * outW + x] = nLabels;
                while (!stack.empty()) {
                    int p = stack.back(); stack.pop_back();
                    area++;
                    int py = p / outW, px = p % outW;
                    for (int dy = -1; dy <= 1; dy++) {
                        int ny = py + dy;
                        if (ny < 0 || ny >= outH) continue;
                        const uchar* nrow = out.ptr<uchar>(ny);
                        for (int dx = -1; dx <= 1; dx++) {
                            if (!dx && !dy) continue;
                            int nx = px + dx;
                            if (nx < 0 || nx >= outW) continue;
                            size_t q = (size_t)ny * outW + nx;
                            if (nrow[nx] && !labels[q]) { labels[q] = nLabels; stack.push_back((int)q); }
                        }
                    }
                }
                areas.push_back(area);
            }
        }
        if (nLabels <= 1) return out;
        int best = 1;
        for (int i = 2; i <= nLabels; i++) if (areas[i] > areas[best]) best = i;
        for (int y = 0; y < outH; y++) {
            uchar* row = out.ptr<uchar>(y);
            for (int x = 0; x < outW; x++) if (labels[(size_t)y * outW + x] != best) row[x] = 0;
        }
        return out;
    }

    struct Centerline {
        double near_error = 0, mid_error = 0, far_error = 0;
        double curve_error = 0, curve_smooth = 0, coverage = 0;
        bool valid = false;
        std::vector<std::pair<int, int>> row_centers;
        std::vector<std::pair<int, int>> left_pts;
        std::vector<std::pair<int, int>> right_pts;
    };

    inline Centerline extractCenterline(const cv::Mat& m, double prevCurveSmooth, const Config& c, int preferred_side = 0) {
        Centerline r;
        const int h = m.rows, w = m.cols;
        const double halfW = (w - 1) * 0.5;
        long nz = 0;

        for (int y = h - 1; y >= 0; y--) {
            const uchar* row = m.ptr<uchar>(y);
            for (int x = 0; x < w; x++) nz += row[x] > 0;

            int bestLen = -1, bs = -1, be = -1;
            bool hasPref = false;
            int s = -1;
            for (int x = 0; x <= w; x++) {
                bool on = (x < w) && row[x] > 0;
                if (on && s < 0) s = x;
                if (!on && s >= 0) {
                    int e = x - 1;
                    int len = e - s;
                    if (len > 20) { 
                        const double segCenter = 0.5 * (s + e);
                        bool isPref = (preferred_side < 0 && segCenter <= halfW) ||
                                      (preferred_side > 0 && segCenter >= halfW);
                        bool take_this = false;
                        if (preferred_side != 0) {
                            if (isPref && !hasPref) take_this = true;
                            else if (isPref && hasPref) {
                                if (preferred_side < 0 && s < bs) take_this = true;
                                if (preferred_side > 0 && e > be) take_this = true;
                            } else if (!hasPref && len > bestLen) {
                                take_this = true;
                            }
                        } else if (len > bestLen) {
                            take_this = true;
                        }

                        if (take_this) {
                            bestLen = len; bs = s; be = e;
                            if (isPref) hasPref = true;
                        }
                    }
                    s = -1;
                }
            }

            if (bs >= 0) {
                r.row_centers.push_back({(bs + be) / 2, y});
                r.left_pts.push_back({bs, y});
                r.right_pts.push_back({be, y});
            }
        }

        r.coverage = (double)nz / ((double)h * (double)w);
        if ((int)r.row_centers.size() < c.min_rows || r.coverage < c.coverage_threshold) {
            r.valid = false; return r;
        }

        std::vector<int> ys; ys.reserve(r.row_centers.size());
        for (auto& p : r.row_centers) ys.push_back(p.second);
        std::sort(ys.begin(), ys.end());
        int yBot = percentileInt(ys, 0.98), yTop = percentileInt(ys, 0.02);
        int span = std::max(yBot - yTop, 1);

        int ySplitNear = yBot - (int)(span * 0.35);
        int ySplitFar  = yTop + (int)(span * 0.30);

        std::vector<int> nearC, midC, farC;
        for (size_t i = 0; i < r.row_centers.size(); ++i) {
            int y  = r.row_centers[i].second;
            int cx = r.row_centers[i].first;
            if (y >= ySplitNear) nearC.push_back(cx);
            else if (y <= ySplitFar) farC.push_back(cx);
            else midC.push_back(cx);
        }

        double nearCx = nearC.empty() ? halfW : medianInts(nearC);
        double midCx  = midC.empty()  ? nearCx : medianInts(midC);
        double farCx  = farC.empty()  ? midCx  : medianInts(farC);

        r.near_error = nearCx - halfW;
        r.mid_error  = midCx  - halfW;
        r.far_error  = farCx  - halfW;
        r.curve_error  = farCx - nearCx;
        r.curve_smooth = 0.65 * prevCurveSmooth + 0.35 * r.curve_error;
        r.valid = true;
        return r;
    }

    inline double detectAsymmetry(const cv::Mat& m, double maxBias = 45.0, double y0f = 0.18, double y1f = 0.60) {
        const int h = m.rows, w = m.cols;
        auto firstLast = [&](int y, int& a, int& b) {
            const uchar* row = m.ptr<uchar>(y);
            int cnt = 0; a = b = -1;
            for (int x = 0; x < w; x++) if (row[x] > 0) { if (a < 0) a = x; b = x; cnt++; }
            return cnt >= 2;
        };
        std::vector<int> bottomCxs;
        for (int y = (int)(h * 0.80); y < h; y += 4) {
            int a, b;
            if (firstLast(y, a, b)) bottomCxs.push_back((a + b) / 2);
        }
        if (bottomCxs.empty()) return 0.0;
        int refCx = (int)medianInts(bottomCxs);
        std::vector<int> lws, rws;
        for (int y = (int)(h * y0f); y < (int)(h * y1f); y += 4) {
            int a, b;
            if (!firstLast(y, a, b)) continue;
            int lw = refCx - a, rw = b - refCx;
            if (lw > 0 && rw > 0) { lws.push_back(lw); rws.push_back(rw); }
        }
        if (lws.empty()) return 0.0;
        double lw = medianInts(lws), rw = medianInts(rws);
        double total = lw + rw;
        if (total < 40) return 0.0;
        return (rw - lw) / total * maxBias;
    }

    inline double planSpeed(double curveSmooth, double maxSp, double minSp, double scale) {
        double ratio = std::min(std::abs(curveSmooth) / scale, 1.0);
        return maxSp - (maxSp - minSp) * ratio;
    }

    inline double errorSpeedFactor(double error, double startPx = 8.0, double fullPx = 35.0, double minFactor = 0.30) {
        double a = std::abs(error);
        if (a <= startPx) return 1.0;
        if (a >= fullPx) return minFactor;
        double t = (a - startPx) / (fullPx - startPx);
        return 1.0 - t * (1.0 - minFactor);
    }
} // namespace detail

class OcclusionGate {
public:
    explicit OcclusionGate(const Config& c = Config()) : c_(c) { reset(); }
    void reset() {
        covEma_ = 0; prevErr_ = 0; init_ = false; hangover_ = 0;
        farHist_.clear(); farSpread_ = 0; lastHard_ = false; lastFarOutlier_ = false;
    }
    bool step(double error, double coverage, const std::vector<std::pair<int, int>>& rows,
              bool valid, double farE, double nearE) {
        farHist_.push_back(farE);
        if ((int)farHist_.size() > c_.far_window) farHist_.pop_front();
        if (farHist_.size() >= 2) {
            auto mm = std::minmax_element(farHist_.begin(), farHist_.end());
            farSpread_ = *mm.second - *mm.first;
        } else farSpread_ = 0;

        if (!init_) { covEma_ = coverage; prevErr_ = error; init_ = true; hangover_ = 0;
                      lastHard_ = false; lastFarOutlier_ = false; return false; }

        covEma_ = c_.cov_ema_alpha * covEma_ + (1.0 - c_.cov_ema_alpha) * coverage;
        if (!valid) { prevErr_ = error; hangover_ = c_.occl_hangover; lastHard_ = false; return true; }

        double errStep = std::abs(error - prevErr_);
        double covDrop = (covEma_ > 0.05) ? std::max(0.0, (covEma_ - coverage) / covEma_) : 0.0;
        double jagged = 0;
        if (rows.size() >= 6) {
            double sum = 0; int n = 0;
            for (size_t i = 1; i + 1 < rows.size(); i++) {
                sum += std::abs(rows[i + 1].first - 2 * rows[i].first + rows[i - 1].first);
                n++;
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
            if (satStreak_ >= c_.auth_sat_frames && std::abs(pred) > std::abs(error) &&
                std::abs(error) > c_.auth_lose_enter_px) losing_ = true;
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
    void reset() {
        integral_ = prevErr_ = filtD_ = prevOut_ = 0; first_ = true;
        auth_.reset(); emgAuth_ = false; emgStreak_ = 0; forced_ = false;
    }
    void softReset() { integral_ = prevErr_ = filtD_ = 0; first_ = true; }
    void setForcedEmergency(bool f) { forced_ = f; }
    void overrideOutput(double a) { prevOut_ = a; }
    bool losing() const { return auth_.losing(); }
    bool emergencyAuthority() const { return emgAuth_; }

    double step(double error, double customKp = -1.0, double customRateLim = -1.0) {
        double baseKp = (customKp > 0.0) ? customKp : c_.kp;
        double baseRateLim = (customRateLim > 0.0) ? customRateLim : c_.rate_limit;
        double kpEff = baseKp * (auth_.losing() ? 1.5 : 1.0);
        double kdEff = c_.kd * (auth_.losing() ? 0.7 : 1.0);
        double outLim, rateLim;
        if (forced_) {
            emgStreak_ = c_.emergency_streak_req;
            outLim = c_.emergency_output_limit; rateLim = c_.emergency_rate_limit; emgAuth_ = true;
        } else {
            if (std::abs(error) > c_.emergency_error_px) emgStreak_++; else emgStreak_ = 0;
            if (emgStreak_ >= c_.emergency_streak_req) {
                outLim = c_.emergency_output_limit; rateLim = c_.emergency_rate_limit; emgAuth_ = true;
            } else { outLim = c_.output_limit; rateLim = baseRateLim; emgAuth_ = false; }
        }
        double sg = detail::sgn(error);
        double e = (std::abs(error) < c_.deadband) ? sg * (std::abs(error) - c_.deadband)
                                                    : error - sg * c_.deadband;
        double p = kpEff * e;
        integral_ = std::clamp(integral_ + e, -c_.integral_limit, c_.integral_limit);
        double i = c_.ki * integral_;
        double rawD = first_ ? 0.0 : (error - prevErr_);
        filtD_ = c_.derivative_alpha * rawD + (1.0 - c_.derivative_alpha) * filtD_;
        double d = kdEff * filtD_;
        double raw = p + i + d;
        double pidOut = std::clamp(raw, -outLim, outLim);
        if (auth_.losing()) integral_ = 0;
        auth_.step(error, prevErr_, pidOut);
        double delta = std::clamp(pidOut - prevOut_, -rateLim, rateLim);
        double out = std::clamp(prevOut_ + delta, -outLim, outLim);
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

// ==========================================================
// LANE TURN CONTROLLER (State Machine 7 Pha cho Roundabout + Đếm T-Junction không biển)
// ==========================================================
class LaneTurnController {
public:
    enum class State { IDLE, WAIT_CURVE, WAIT_STRAIGHT, EXECUTE, SETTLE, DONE };

    enum class RoundaboutPhase {
        NONE = 0,
        S1_WAIT_STRAIGHT,
        S2_APPROACH_ISLAND,
        S3_ENTRY_TURN_RIGHT,
        S4_PASS_NOSE_RIGHT,
        S5_SKIP_BRANCH1_LEFT,
        S6_HUG_ISLAND_LEFT,
        S7_EXIT_TURN_RIGHT
    };

    double TURN_STEER_DEG      = 33.0;   
    double TURN_INIT_DEG       = 18.0;   
    double TURN_RAMP_DEG_PER_S = 115.0;  
    double MIN_TURN_TIME       = 0.8;
    double TURN_BIAS_PX        = 38.0;
    double turn_floor_current  = 0.0;

    // --- PHA 1 ---
    int    RB_S1_STABLE_FRAMES  = 3;     
    // --- PHA 2 ---
    double RB_S2_BIAS_PX        = 15.0;  
    double RB_S2_FLOOR_DEG      = 7.0;   
    double RB_S2_CAP_DEG        = 10.0;  
    double RB_S2_MAX_TIME       = 3.50;  
    // --- PHA 3 ---
    double RB_S3_BIAS_PX        = 26.0;  
    double RB_S3_FLOOR_DEG      = 28.0;  
    double RB_S3_CAP_DEG        = 38.0;  
    double RB_S3_MIN_TIME       = 1.30;  
    double RB_S3_MAX_TIME       = 2.00;  
    // --- PHA 4 ---
    double RB_S4_BIAS_PX        = 13.0;  
    double RB_S4_FLOOR_DEG      = 20.0;  
    double RB_S4_CAP_DEG        = 28.0;  
    double RB_S4_MIN_TIME       = 0.85;  
    double RB_S4_MAX_TIME       = 1.40;  
    // --- PHA 5 ---
    double RB_S5_BIAS_PX        = 18.0;  
    double RB_S5_FLOOR_DEG      = 15.0;  
    double RB_S5_CAP_DEG        = 25.0;  
    double RB_S5_MIN_TIME       = 0.75;  
    double RB_S5_MAX_TIME       = 1.25;  
    // --- PHA 6 ---
    double RB_S6_BIAS_PX        = 20.0;  
    double RB_S6_FLOOR_DEG      = 18.0;  
    double RB_S6_CAP_DEG        = 28.0;  
    double RB_S6_MIN_TIME       = 0.40;  
    double RB_S6_MAX_TIME       = 0.95;  
    // --- PHA 7 ---
    double RB_S7_BIAS_PX        = 26.0;  
    double RB_S7_FLOOR_DEG      = 25.0;  
    double RB_S7_CAP_DEG        = 38.0;  
    double RB_S7_MIN_TIME       = 0.80;  
    double RB_S7_MAX_TIME       = 2.15;  

    int    rb_entry_stable_frames     = 0;
    bool   rb_approach_straight       = false;
    int    ALIGN_EDGE_MARGIN        = 5;
    int    ALIGN_MIN_WIDTH_PX       = 60;
    double ALIGN_FAR_CENTER_PX      = 60.0;
    double ALIGN_FAR_OPPOSITE_PX    = 30.0;
    int    REQUIRED_ALIGNED_FRAMES  = 2;
    int    aligned_frames = 0;
    bool   far_swept = false;

    int    JUNCTION_FAR_Y           = 160;
    int    TJUNCTION_OPEN_WIDTH_PX  = 400;     
    double TJUNCTION_COOLDOWN_S     = 1.5;     

    double settle_start_bias  = 0.0;
    double settle_start_floor = 0.0;
    double settle_elapsed     = 0.0;
    double SETTLE_TIME        = 0.5;
    double turn_bias_current  = 0.0;

    double MAX_TURN_TIME            = 4.0;
    double CURVE_THRESHOLD          = 25.0;
    double STRAIGHT_THRESHOLD       = 12.0;
    int    REQUIRED_STRAIGHT_FRAMES = 3;
    int    straight_frames          = 0;

    bool            is_roundabout_mode      = false;
    RoundaboutPhase rb_phase                = RoundaboutPhase::NONE;
    int             roundabout_exits_passed = 0;
    bool            branch1_cleared         = false;
    int             rb_phase_Confirm_frames = 0;

    bool            waiting_for_tjunction   = false; 
    bool            is_unmarked_tjunction   = false; 
    double          tjunction_cooldown      = 0.0;
    int             t_junction_count        = 0;     
    int             tjunction_detect_frames = 0;
    int             rb_exit_straight_frames = 0;

    bool   tjunction_geo_detected       = false;   
    int    tjunction_geo_confirm_frames = 0;        
    bool   tjunction_count_locked       = false;    

    State       state          = State::IDLE;
    std::string direction      = "NONE";
    double      turn_elapsed   = 0.0;
    bool        curve_detected = false;
    bool        completed      = false;

    double get_turn_elapsed() const { return turn_elapsed; }

    std::vector<cv::Point> select_three_points(std::vector<cv::Point> points) {
        std::sort(points.begin(), points.end(), [](const cv::Point& a, const cv::Point& b){ return a.y < b.y; });
        int n = static_cast<int>(points.size());
        if (n < 3) return {};
        if (n == 3) return points;
        if (n == 4) return {points[0], points[1], points[2]};
        int start = (n - 3) / 2;
        return {points[start], points[start + 1], points[start + 2]};
    }

    double calculate_deviation(std::vector<cv::Point> points) {
        auto p = select_three_points(points);
        if (p.empty()) return -1.0;
        double dx = p[2].x - p[0].x;
        double dy = p[2].y - p[0].y;
        double length = std::hypot(dx, dy);
        if (length < 1e-6) return -1.0;
        return std::abs(dx * (p[1].y - p[0].y) - dy * (p[1].x - p[0].x)) / length;
    }

    int get_preferred_side() const {
        if (!is_roundabout_mode) return 0;
        switch (rb_phase) {
            case RoundaboutPhase::S1_WAIT_STRAIGHT:     return 0;
            case RoundaboutPhase::S2_APPROACH_ISLAND:
            case RoundaboutPhase::S3_ENTRY_TURN_RIGHT:
            case RoundaboutPhase::S4_PASS_NOSE_RIGHT:
            case RoundaboutPhase::S7_EXIT_TURN_RIGHT:   return 1;
            case RoundaboutPhase::S5_SKIP_BRANCH1_LEFT:
            case RoundaboutPhase::S6_HUG_ISLAND_LEFT:   return -1;
            default:                                    return 0;
        }
    }

    const char* get_roundabout_phase_str() const {
        switch (rb_phase) {
            case RoundaboutPhase::S1_WAIT_STRAIGHT:     return "RB_S1_WAIT_STRAIGHT";
            case RoundaboutPhase::S2_APPROACH_ISLAND:   return "RB_S2_APPROACH";
            case RoundaboutPhase::S3_ENTRY_TURN_RIGHT:  return "RB_S3_ENTRY_RIGHT";
            case RoundaboutPhase::S4_PASS_NOSE_RIGHT:   return "RB_S4_PASS_NOSE";
            case RoundaboutPhase::S5_SKIP_BRANCH1_LEFT: return "RB_S5_SKIP_BR1_L";
            case RoundaboutPhase::S6_HUG_ISLAND_LEFT:   return "RB_S6_HUG_ISL_L";
            case RoundaboutPhase::S7_EXIT_TURN_RIGHT:   return "RB_S7_EXIT_RIGHT";
            default:                                    return "NONE";
        }
    }

    void force_execute_now() {
        if (state != State::WAIT_CURVE && state != State::WAIT_STRAIGHT) return;
        const double dir_sign = (direction == "LEFT") ? -1.0 : 1.0;
        state              = State::EXECUTE;
        turn_elapsed       = 0.0;
        aligned_frames     = 0;
        far_swept          = false;
        straight_frames    = 0;
        turn_bias_current  = dir_sign * TURN_BIAS_PX;
        turn_floor_current = 0.0;
    }

    void start_turn(const std::string& dir, bool is_roundabout = false, bool unmarked_tjunction = false) {
        if (state != State::IDLE && state != State::WAIT_CURVE && state != State::WAIT_STRAIGHT) return;
        if ((state == State::WAIT_CURVE || state == State::WAIT_STRAIGHT) &&
            direction == dir && is_roundabout_mode == is_roundabout && is_unmarked_tjunction == unmarked_tjunction) return;

        direction          = dir;
        state              = State::WAIT_CURVE;
        turn_elapsed       = 0.0;
        aligned_frames     = 0;
        straight_frames    = 0;
        turn_bias_current  = 0.0;
        turn_floor_current = 0.0;
        settle_start_floor = 0.0;
        far_swept          = false;
        curve_detected     = false;
        completed          = false;

        is_roundabout_mode      = is_roundabout;
        is_unmarked_tjunction   = unmarked_tjunction;
        roundabout_exits_passed = 0;
        branch1_cleared         = false;
        rb_phase_Confirm_frames = 0;
        rb_entry_stable_frames  = 0;
        rb_approach_straight    = false;

        if (is_roundabout_mode) {
            rb_phase = RoundaboutPhase::S1_WAIT_STRAIGHT;
            std::cout << "[ROUNDABOUT] Mode Active -> Phase S1: WAIT_STRAIGHT on approach road...\n";
        } else if (is_unmarked_tjunction) {
            rb_phase = RoundaboutPhase::NONE;
            std::cout << "[T-JUNCTION] Armed Unmarked T-Junction #" << (t_junction_count + 1)
                      << " -> Waiting for junction (" << dir << ")...\n";
        } else {
            rb_phase = RoundaboutPhase::NONE;
            std::cout << "[TURN] Requested: " << dir << " -> Waiting for junction...\n";
        }
    }

    struct TurnResult {
        State state;
        double steer_bias_px;
        std::string direction;
        double dev              = -1.0;
        bool   junction_opened  = false;
        int    straight_frames  = 0;
        int    roundabout_exit  = 0;
        int    t_junction_count = 0;
        int    preferred_side   = 0;
        double steer_floor_deg  = 0.0;  
        double steer_cap_deg    = 0.0;  
        bool   lock_opposite_err = false;
        bool   tjunction_detected = false;
        cv::Point p0 = cv::Point(-1, -1);
        cv::Point p1 = cv::Point(-1, -1);
        cv::Point p2 = cv::Point(-1, -1);
    };

    TurnResult update(const std::vector<cv::Point>& left_pts, const std::vector<cv::Point>& right_pts,
                      double dt, bool lane_valid,
                      int far_lx = -1, int far_rx = -1,
                      int mid_lx = -1, int mid_rx = -1,
                      int img_w = 480,
                      bool has_active_sign = false,
                      double near_e = 0.0, double mid_e = 0.0, double far_e = 0.0,
                      int lx_bot = -1, int rx_bot = -1) {
        TurnResult res{state, 0.0, direction};
        res.t_junction_count  = t_junction_count;

        const int far_width = (far_lx >= 0 && far_rx >= 0) ? (far_rx - far_lx) : 0;
        const int mid_width = (mid_lx >= 0 && mid_rx >= 0) ? (mid_rx - mid_lx) : 0;
        const int bot_width = (lx_bot >= 0 && rx_bot >= 0) ? (rx_bot - lx_bot) : 0;

        tjunction_geo_detected = false;
        if (!tjunction_count_locked && !is_roundabout_mode && !has_active_sign) {
            const bool bilateral_open = (mid_lx >= 0 && mid_lx <= 5) && (mid_rx >= img_w - 10);
            const double flare_ratio = (bot_width > 30) ? ((double)mid_width / (double)bot_width) : 0.0;
            const bool flare_ok = flare_ratio >= 1.85;
            const bool perpendicular = (std::abs(near_e) <= 5.0) && (std::abs(mid_e) <= 8.0);
            const bool ceiling_cut = (far_width <= 360) || (far_lx < 0);

            if (bilateral_open && flare_ok && perpendicular && ceiling_cut) {
                tjunction_geo_detected = true;
                tjunction_geo_confirm_frames++;
            } else {
                tjunction_geo_confirm_frames = 0;
            }
        } else {
            tjunction_geo_confirm_frames = 0;
        }
        res.tjunction_detected = tjunction_geo_detected;

        if (state == State::IDLE) {
            if (tjunction_cooldown > 0.0) {
                tjunction_cooldown = std::max(0.0, tjunction_cooldown - dt);
            }
            if (waiting_for_tjunction && tjunction_cooldown <= 0.0 && !has_active_sign && t_junction_count < 2) {
                if (lane_valid && std::abs(near_e) <= 6.5 && std::abs(mid_e) <= 14.0) {
                    rb_exit_straight_frames++;
                } else {
                    rb_exit_straight_frames = 0;
                }

                if (rb_exit_straight_frames >= 2) {
                    const std::string target_dir = (t_junction_count == 0) ? "LEFT" : "RIGHT";
                    waiting_for_tjunction   = false;
                    rb_exit_straight_frames = 0;
                    start_turn(target_dir, false, true);
                    res.state     = state;
                    res.direction = direction;
                }
            }
            if (state == State::IDLE) return res;
        }

        if (is_roundabout_mode) {
            turn_elapsed += dt;

            if (rb_phase == RoundaboutPhase::S1_WAIT_STRAIGHT) {
                const bool straight_now = lane_valid && (std::abs(near_e) <= 8.0) && (std::abs(mid_e) <= 18.0);
                if (straight_now) {
                    rb_entry_stable_frames++;
                } else {
                    rb_entry_stable_frames = 0;
                }

                if (rb_entry_stable_frames >= RB_S1_STABLE_FRAMES) {
                    rb_approach_straight    = true;
                    rb_phase                = RoundaboutPhase::S2_APPROACH_ISLAND;
                    turn_elapsed            = 0.0;
                    rb_phase_Confirm_frames = 0;
                    std::cout << "[ROUNDABOUT] >> S1 -> S2: APPROACH_ISLAND (Approach road straight, locking anti-left!)\n";
                } else {
                    res.state             = State::WAIT_CURVE;
                    res.direction         = "RIGHT";
                    res.preferred_side    = 0;
                    res.steer_bias_px     = 0.0;
                    res.steer_floor_deg   = 0.0;
                    res.steer_cap_deg     = 0.0;
                    res.lock_opposite_err = false;
                    return res;
                }
            }

            if (rb_phase == RoundaboutPhase::S2_APPROACH_ISLAND) {
                res.state             = State::WAIT_STRAIGHT;
                res.direction         = "RIGHT";
                res.preferred_side    = 1;
                res.steer_bias_px     = RB_S2_BIAS_PX;
                res.steer_floor_deg   = RB_S2_FLOOR_DEG;
                res.steer_cap_deg     = RB_S2_CAP_DEG;
                res.lock_opposite_err = true;

                const bool nose_at_mid_right = (far_rx >= img_w - 15) && (mid_rx >= img_w - 15) &&
                                               (far_lx >= 200) &&
                                               (mid_lx >= 55 || mid_e >= 22.0);
                const bool nose_split_at_mid = (far_rx >= 0 && far_rx <= 290) &&
                                               ((mid_rx >= 0 && mid_rx <= img_w - 35) || (mid_e <= -20.0) || (mid_lx >= 50));

                if (nose_at_mid_right || nose_split_at_mid || turn_elapsed >= RB_S2_MAX_TIME) {
                    rb_phase                = RoundaboutPhase::S3_ENTRY_TURN_RIGHT;
                    state                   = State::EXECUTE;
                    turn_elapsed            = 0.0;
                    branch1_cleared         = false;
                    rb_phase_Confirm_frames = 0;
                    std::cout << "[ROUNDABOUT] >> S2 -> S3: ENTRY_TURN_RIGHT (Island nose at mid: far_lx="
                              << far_lx << ", mid_lx=" << mid_lx << ", mid_e=" << mid_e << ")\n";
                } else {
                    return res;
                }
            }

            if (rb_phase == RoundaboutPhase::S3_ENTRY_TURN_RIGHT) {
                res.direction         = "RIGHT";
                res.preferred_side    = 1;
                res.steer_bias_px     = RB_S3_BIAS_PX;
                res.steer_floor_deg   = RB_S3_FLOOR_DEG;
                res.steer_cap_deg     = RB_S3_CAP_DEG;
                res.lock_opposite_err = true;

                if (mid_lx >= 90 || mid_e >= 30.0) {
                    branch1_cleared = true;
                }

                const bool nose_peak_reached = (turn_elapsed >= RB_S3_MIN_TIME) && branch1_cleared;
                if (nose_peak_reached || turn_elapsed >= RB_S3_MAX_TIME) {
                    rb_phase                = RoundaboutPhase::S4_PASS_NOSE_RIGHT;
                    turn_elapsed            = 0.0;
                    rb_phase_Confirm_frames = 0;
                    std::cout << "[ROUNDABOUT] >> S3 -> S4: PASS_NOSE_RIGHT (Passing island nose peak)\n";
                }
            }
            else if (rb_phase == RoundaboutPhase::S4_PASS_NOSE_RIGHT) {
                res.direction         = "RIGHT";
                res.preferred_side    = 1;
                res.steer_bias_px     = RB_S4_BIAS_PX;
                res.steer_floor_deg   = RB_S4_FLOOR_DEG;
                res.steer_cap_deg     = RB_S4_CAP_DEG;
                res.lock_opposite_err = true;

                bool rb_right_open_near = false, rb_right_open_far = false;
                for (const auto& p : right_pts) {
                    if (p.x >= img_w - 10) {
                        if (p.y >= 160 && p.y <= 210) rb_right_open_near = true;
                        if (p.y <= 160)               rb_right_open_far  = true;
                    }
                }
                const bool rb_junction_opened = rb_right_open_near && rb_right_open_far;

                const bool nose_passed = (turn_elapsed >= RB_S4_MIN_TIME) &&
                                         ((mid_lx >= 0 && mid_lx <= 90) || (mid_e <= 22.0));

                if (rb_junction_opened && turn_elapsed >= RB_S4_MIN_TIME) {
                    rb_phase                = RoundaboutPhase::S5_SKIP_BRANCH1_LEFT;
                    turn_elapsed            = 0.0;
                    roundabout_exits_passed = 1;
                    rb_phase_Confirm_frames = 0;
                    std::cout << "[ROUNDABOUT] >> S4 -> S5: SKIP_BRANCH1_LEFT (Junction OPENED on right -> steering LEFT immediately!)\n";
                } else if (nose_passed || turn_elapsed >= RB_S4_MAX_TIME) {
                    rb_phase                = RoundaboutPhase::S5_SKIP_BRANCH1_LEFT;
                    turn_elapsed            = 0.0;
                    roundabout_exits_passed = 1;
                    rb_phase_Confirm_frames = 0;
                    std::cout << "[ROUNDABOUT] >> S4 -> S5: SKIP_BRANCH1_LEFT (Steering LEFT to skip Branch 1)\n";
                }
            }
            else if (rb_phase == RoundaboutPhase::S5_SKIP_BRANCH1_LEFT) {
                res.direction         = "LEFT";
                res.preferred_side    = -1;
                res.steer_bias_px     = -RB_S5_BIAS_PX;
                res.steer_floor_deg   = -RB_S5_FLOOR_DEG;
                res.steer_cap_deg     = -RB_S5_CAP_DEG;
                res.lock_opposite_err = true;

                const bool branch1_skipped = (turn_elapsed >= RB_S5_MIN_TIME) &&
                                             (near_e <= -6.0 || mid_e <= -12.0);

                if (branch1_skipped || turn_elapsed >= RB_S5_MAX_TIME) {
                    rb_phase                = RoundaboutPhase::S6_HUG_ISLAND_LEFT;
                    turn_elapsed            = 0.0;
                    rb_phase_Confirm_frames = 0;
                    std::cout << "[ROUNDABOUT] >> S5 -> S6: HUG_ISLAND_LEFT (Hugging central island, watching for Exit 2)\n";
                }
            }
            else if (rb_phase == RoundaboutPhase::S6_HUG_ISLAND_LEFT) {
                res.direction         = "LEFT";
                res.preferred_side    = -1;
                res.steer_bias_px     = -RB_S6_BIAS_PX;
                res.steer_floor_deg   = -RB_S6_FLOOR_DEG;
                res.steer_cap_deg     = -RB_S6_CAP_DEG;
                res.lock_opposite_err = true;

                const bool exit2_opened = (turn_elapsed >= RB_S6_MIN_TIME) &&
                                          ((far_lx >= 145 && far_rx >= img_w - 15) ||
                                           (mid_lx >= 55 && mid_rx >= img_w - 15) ||
                                           (mid_e >= 18.0));

                rb_phase_Confirm_frames = exit2_opened ? (rb_phase_Confirm_frames + 1) : 0;
                if (rb_phase_Confirm_frames >= 2 || turn_elapsed >= RB_S6_MAX_TIME) {
                    roundabout_exits_passed = 2;
                    rb_phase                = RoundaboutPhase::S7_EXIT_TURN_RIGHT;
                    turn_elapsed            = 0.0;
                    aligned_frames          = 0;
                    far_swept               = false;
                    turn_bias_current       = RB_S7_BIAS_PX;
                    turn_floor_current      = RB_S7_FLOOR_DEG;
                    std::cout << "[ROUNDABOUT] >>> S6 -> S7: EXIT_TURN_RIGHT (Taking 2nd exit out of roundabout)\n";
                }
            }
            else if (rb_phase == RoundaboutPhase::S7_EXIT_TURN_RIGHT) {
                res.direction         = "RIGHT";
                res.preferred_side    = 1;
                res.steer_bias_px     = RB_S7_BIAS_PX;
                res.steer_floor_deg   = RB_S7_FLOOR_DEG;
                res.steer_cap_deg     = RB_S7_CAP_DEG;
                res.lock_opposite_err = true;

                const bool far_bounded = (far_lx > ALIGN_EDGE_MARGIN) && (far_rx >= 0) &&
                                         (far_rx < img_w - 1 - ALIGN_EDGE_MARGIN) &&
                                         (far_rx - far_lx >= ALIGN_MIN_WIDTH_PX);
                if (!far_bounded) far_swept = true;

                const double far_center_off = far_bounded ? (0.5 * (far_lx + far_rx) - 0.5 * img_w) : 9999.0;
                const bool far_centered = far_bounded &&
                                          (far_center_off <  ALIGN_FAR_CENTER_PX) &&
                                          (far_center_off > -ALIGN_FAR_OPPOSITE_PX);

                const bool exit_road_straight = (turn_elapsed >= RB_S7_MIN_TIME) && lane_valid &&
                                                (far_lx >= 0 && far_lx <= 115) &&
                                                (std::abs(mid_e) <= 24.0) && (std::abs(far_e) <= 25.0);

                if ((turn_elapsed >= RB_S7_MIN_TIME && lane_valid && far_swept && far_centered) || exit_road_straight) {
                    aligned_frames++;
                } else {
                    aligned_frames = 0;
                }

                if (aligned_frames >= REQUIRED_ALIGNED_FRAMES || turn_elapsed >= RB_S7_MAX_TIME) {
                    is_roundabout_mode      = false;
                    rb_phase                = RoundaboutPhase::NONE;
                    roundabout_exits_passed = 0;
                    branch1_cleared         = false;
                    rb_phase_Confirm_frames = 0;
                    rb_approach_straight    = false;

                    waiting_for_tjunction   = true;
                    tjunction_cooldown      = TJUNCTION_COOLDOWN_S;
                    rb_exit_straight_frames = 0;

                    state              = State::SETTLE;
                    direction          = "RIGHT";
                    settle_start_bias  = RB_S7_BIAS_PX;
                    settle_start_floor = RB_S7_FLOOR_DEG;
                    settle_elapsed     = 0.0;
                    std::cout << "[ROUNDABOUT] >>> EXIT COMPLETED -> RESET ROUNDABOUT FLAGS & ARM T-JUNCTION COUNTER!\n";
                }
            }

            res.roundabout_exit = roundabout_exits_passed;
            res.state           = state;
            return res;
        }

        auto& pts = (direction == "LEFT") ? left_pts : right_pts;
        auto selected_pts = select_three_points(pts);
        double dev = calculate_deviation(pts);

        if (selected_pts.size() == 3) {
            res.p0 = selected_pts[0];
            res.p1 = selected_pts[1];
            res.p2 = selected_pts[2];
        }

        bool side_open_near = false, side_open_far = false;
        for (const auto& p : pts) {
            bool touch = (direction == "LEFT") ? (p.x <= 5) : (p.x >= img_w - 10);
            if (!touch) continue;
            if (p.y >= 160 && p.y <= 210) side_open_near = true;
            if (p.y <= JUNCTION_FAR_Y)    side_open_far  = true;
        }

        bool tjunction_wall_ahead = false;
        if (is_unmarked_tjunction) {
            const bool both_mid_open = (mid_lx >= 0 && mid_lx <= 15) && (mid_rx >= img_w - 15);
            const bool far_t_signature = (far_lx >= 0 && far_lx <= 15) ||
                                         (far_rx >= img_w - 15) ||
                                         (far_rx >= 0 && far_rx <= 360);
            if (both_mid_open && far_t_signature) {
                side_open_near       = true;
                side_open_far        = true;
                tjunction_wall_ahead = true;
            } else {
                side_open_near = false;
                side_open_far  = false;
            }
        }
        bool junction_opened = side_open_near && side_open_far;

        bool pts_stuck_on_border = false;
        if (selected_pts.size() == 3) {
            if (direction == "LEFT" &&
                selected_pts[0].x <= 3 && selected_pts[1].x <= 3 && selected_pts[2].x <= 3) {
                pts_stuck_on_border = true;
            } else if (direction == "RIGHT" &&
                       selected_pts[0].x >= img_w - 4 && selected_pts[1].x >= img_w - 4 && selected_pts[2].x >= img_w - 4) {
                pts_stuck_on_border = true;
            }
        }

        res.dev             = dev;
        res.junction_opened = junction_opened;
        res.straight_frames = straight_frames;
        res.roundabout_exit = roundabout_exits_passed;

        if (state == State::WAIT_CURVE) {
            straight_frames = 0;
            const bool trigger_wait_straight = is_unmarked_tjunction
                ? junction_opened
                : (dev >= CURVE_THRESHOLD || junction_opened);
            if (trigger_wait_straight) {
                curve_detected  = true;
                state           = State::WAIT_STRAIGHT;
                straight_frames = 1;
                res.straight_frames = straight_frames;
                std::cout << "[TURN] >> JUNCTION DETECTED (dev: " << dev << ", opened: " << junction_opened << ")\n";
            }
        }
        else if (state == State::WAIT_STRAIGHT) {
            const bool valid_straight_dev = (dev >= 0.0 && dev <= STRAIGHT_THRESHOLD) ||
                                            (is_unmarked_tjunction && tjunction_wall_ahead);
            const bool junction_apex_reached = junction_opened && (dev >= 25.0 || mid_width >= 330);
            if ((valid_straight_dev && junction_opened && (!pts_stuck_on_border || mid_width >= 350 || tjunction_wall_ahead)) ||
                (straight_frames >= 1 && junction_apex_reached)) {
                straight_frames++;
            } else {
                straight_frames = 0;
            }

            const int req_straight = 4;

            if (straight_frames >= req_straight) {
                if (is_unmarked_tjunction) {
                    t_junction_count++;
                    tjunction_count_locked = true;
                    res.t_junction_count = t_junction_count;
                    std::cout << "[T-JUNCTION] >>> EXECUTING UNMARKED T-JUNCTION #" << t_junction_count
                              << " (" << direction << ")\n";
                }
                state          = State::EXECUTE;
                turn_elapsed   = 0.0;
                aligned_frames = 0;
                far_swept      = false;
                const double dir_sign = (direction == "LEFT") ? -1.0 : 1.0;
                turn_bias_current     = dir_sign * TURN_BIAS_PX;
                turn_floor_current    = dir_sign * std::min(TURN_STEER_DEG, std::max(TURN_INIT_DEG, TURN_RAMP_DEG_PER_S * dt));
                res.steer_bias_px     = turn_bias_current;
                res.steer_floor_deg   = turn_floor_current;
                res.lock_opposite_err = true;
                std::cout << "[TURN] >>> EXECUTING TURN (Verified straight " << req_straight << " frames)\n";
            }
        }
        else if (state == State::EXECUTE) {
            turn_elapsed += dt;
            const double dir_sign = (direction == "LEFT") ? -1.0 : 1.0;

            const double mag = std::min(TURN_STEER_DEG, std::abs(turn_floor_current) + TURN_RAMP_DEG_PER_S * dt);
            turn_floor_current = dir_sign * mag;

            const bool far_bounded = (far_lx > ALIGN_EDGE_MARGIN) && (far_rx >= 0) &&
                                     (far_rx < img_w - 1 - ALIGN_EDGE_MARGIN) &&
                                     (far_rx - far_lx >= ALIGN_MIN_WIDTH_PX);
            if (!far_bounded) far_swept = true;

            const double far_center_off = far_bounded ? (0.5 * (far_lx + far_rx) - 0.5 * img_w) : 9999.0;
            const double off_dir = dir_sign * far_center_off;
            const bool far_centered = far_bounded &&
                                      (off_dir <  ALIGN_FAR_CENTER_PX) &&
                                      (off_dir > -ALIGN_FAR_OPPOSITE_PX);

            const bool lane_aligned =
                (turn_elapsed >= MIN_TURN_TIME) &&
                lane_valid &&
                far_swept &&
                far_centered;

            if (lane_aligned) aligned_frames++;
            else aligned_frames = 0;

            res.steer_bias_px     = turn_bias_current;
            res.steer_floor_deg   = turn_floor_current;
            res.lock_opposite_err = true;

            if (aligned_frames >= REQUIRED_ALIGNED_FRAMES) {
                state              = State::SETTLE;
                settle_start_bias  = turn_bias_current;
                settle_start_floor = turn_floor_current;
                settle_elapsed     = 0.0;
                std::cout << "[TURN] NEW LANE ALIGNED (far off: " << far_center_off << ") - SETTLE\n";
            }

            const bool wide_open_intersection =
                (direction == "RIGHT" && mid_lx >= 0 && mid_lx <= 25 && mid_rx >= img_w - 15 && mid_e < 12.0) ||
                (direction == "LEFT"  && mid_rx >= img_w - 26 && mid_lx >= 0 && mid_lx <= 15 && mid_e > -12.0);
            const double effective_max_turn_time = wide_open_intersection ? 6.8 : MAX_TURN_TIME;

            if (turn_elapsed >= effective_max_turn_time && state == State::EXECUTE) {
                state              = State::SETTLE;
                settle_start_bias  = turn_bias_current;
                settle_start_floor = turn_floor_current;
                settle_elapsed     = 0.0;
                std::cout << "[TURN] TIMEOUT - RELEASE TURN BIAS\n";
            }
        }
        else if (state == State::SETTLE) {
            settle_elapsed += dt;
            double ratio = std::clamp(1.0 - settle_elapsed / std::max(SETTLE_TIME, 0.01), 0.0, 1.0);
            turn_bias_current     = settle_start_bias  * ratio;
            turn_floor_current    = settle_start_floor * ratio;
            res.steer_bias_px     = turn_bias_current;
            res.steer_floor_deg   = turn_floor_current;
            res.lock_opposite_err = true;

            if (settle_elapsed >= SETTLE_TIME) {
                turn_bias_current  = 0.0;
                turn_floor_current = 0.0;
                res.steer_bias_px  = 0.0;
                res.steer_floor_deg = 0.0;
                state     = State::DONE;
                completed = true;
                std::cout << "[TURN] TURN BIAS RELEASED\n";
            }
        }
        else if (state == State::DONE) {
            state                 = State::IDLE;
            direction             = "NONE";
            is_unmarked_tjunction = false;
            turn_elapsed          = 0.0;
            aligned_frames        = 0;
            straight_frames       = 0;
            turn_bias_current     = 0.0;
            turn_floor_current    = 0.0;
            curve_detected        = false;
            far_swept             = false;
            settle_start_bias     = 0.0;
            settle_start_floor    = 0.0;
            settle_elapsed        = 0.0;
            tjunction_count_locked = false;
            tjunction_geo_confirm_frames = 0;
            std::cout << "[TURN] RESET TO IDLE\n";
        }

        res.state = state;
        return res;
    }
};

// ==========================================================
// CORNER ASSIST
// ==========================================================
class CornerAssist {
public:
    enum class Phase { IDLE, ACTIVE, RELEASE };

    double CORNER_STEER_DEG      = 45.0;  
    double CORNER_INIT_DEG       = 26.0;  
    double CORNER_RAMP_DEG_PER_S = 140.0; 
    double CORNER_RELEASE_TIME   = 0.35;  
    double CORNER_MIN_TIME       = 0.55;  
    double CORNER_MAX_TIME       = 2.6;   
    double CORNER_COOLDOWN_S     = 1.0;   
    int    CORNER_BEND_PX        = 110;   
    int    CORNER_EARLY_FAR_PX   = 310;   
    int    CORNER_CONFIRM_FRAMES = 1;     
    int    EDGE_TOUCH_PX         = 3;
    int    ALIGN_CENTER_PX       = 70;
    int    ALIGN_MIN_WIDTH_PX    = 60;
    int    OVERTURN_PX           = 40;    

    void reset() {
        phase_ = Phase::IDLE; dir_ = 0; pending_dir_ = 0; confirm_ = 0;
        aligned_ = 0; elapsed_ = 0.0; floor_ = 0.0; release_start_ = 0.0; release_t_ = 0.0;
    }
    bool   active()    const { return phase_ != Phase::IDLE; }
    int    direction() const { return dir_; }
    double floor()     const { return floor_; }

    double update(int lx1, int rx1, int lx2, int rx2, int lx_bot, int rx_bot, int w, double dt,
                  bool has_active_sign = false,
                  double near_e = 0.0, double mid_e = 0.0, double far_e = 0.0) {
        const int  edgeR    = w - 1 - EDGE_TOUCH_PX;
        const bool far_none = (lx1 < 0 || rx1 < 0);

        const bool on_bridge_crest = (std::abs(near_e) <= 6.0) &&
            ((std::abs(far_e) >= 75.0 && std::abs(near_e) <= 4.0) ||
             (std::abs(mid_e) >= 15.0 && std::abs(far_e) >= 65.0));

        const bool right_far_ok = !far_none
            ? ((rx1 >= edgeR) && ((lx1 - lx2) >= CORNER_BEND_PX) &&
               (lx2 > EDGE_TOUCH_PX + 12 || lx1 >= CORNER_EARLY_FAR_PX) && (far_e >= 32.0))
            : (!has_active_sign && (lx2 >= 130) && (mid_e >= 18.0) && (far_e >= 50.0));

        const bool left_far_ok = !far_none
            ? ((lx1 >= 0 && lx1 <= EDGE_TOUCH_PX) && ((rx2 - rx1) >= CORNER_BEND_PX) &&
               (rx2 < edgeR - 12 || rx1 <= w - 1 - CORNER_EARLY_FAR_PX) && (far_e <= -32.0))
            : (!has_active_sign && (rx2 >= 0 && rx2 <= w - 1 - 130) && (mid_e <= -18.0) && (far_e <= -50.0));

        const bool right_sig = !on_bridge_crest &&
                               (rx2 >= edgeR) && (rx_bot >= edgeR) &&
                               (lx2 >= 0) && right_far_ok;

        const bool left_sig  = !on_bridge_crest &&
                               (lx2 >= 0 && lx2 <= EDGE_TOUCH_PX) && (lx_bot >= 0 && lx_bot <= EDGE_TOUCH_PX) &&
                               (rx2 >= 0) && left_far_ok;

        if (cooldown_ > 0.0) cooldown_ = std::max(0.0, cooldown_ - dt);

        if (phase_ == Phase::IDLE) {
            floor_ = 0.0;
            if (cooldown_ > 0.0) { confirm_ = 0; pending_dir_ = 0; return 0.0; }
            const int sig = right_sig ? 1 : (left_sig ? -1 : 0);
            if (sig != 0 && sig == pending_dir_) confirm_++;
            else { pending_dir_ = sig; confirm_ = (sig != 0) ? 1 : 0; }

            if (confirm_ >= CORNER_CONFIRM_FRAMES) {
                phase_ = Phase::ACTIVE; dir_ = pending_dir_;
                elapsed_ = 0.0; aligned_ = 0;
                floor_ = dir_ * CORNER_INIT_DEG;
                std::cout << "[CORNER] >> SHARP " << (dir_ > 0 ? "RIGHT" : "LEFT") << " CORNER -> ASSIST ON\n";
            }
        }
        else if (phase_ == Phase::ACTIVE) {
            elapsed_ += dt;
            const double mag = std::min(CORNER_STEER_DEG, std::abs(floor_) + CORNER_RAMP_DEG_PER_S * dt);
            floor_ = dir_ * mag;

            const bool far_bounded = (lx1 > EDGE_TOUCH_PX) && (rx1 >= 0) && (rx1 < edgeR) &&
                                     (rx1 - lx1 >= ALIGN_MIN_WIDTH_PX);
            const double far_off = far_bounded ? (0.5 * (lx1 + rx1) - 0.5 * w) : 9999.0;
            const bool aligned = (elapsed_ >= CORNER_MIN_TIME) && far_bounded && (std::abs(far_off) < ALIGN_CENTER_PX);
            aligned_ = aligned ? aligned_ + 1 : 0;

            const bool overturn = (dir_ > 0) ? (rx_bot >= 0 && rx_bot < w / 2 + OVERTURN_PX)
                                             : (lx_bot >= 0 && lx_bot > w / 2 - OVERTURN_PX);

            const bool edge_detached = (dir_ > 0) ? (rx2 >= 0 && rx2 < edgeR - 20)
                                                  : (lx2 >= 0 && lx2 > EDGE_TOUCH_PX + 20);

            if (aligned_ >= 2 || overturn || edge_detached || elapsed_ >= CORNER_MAX_TIME) {
                phase_ = Phase::RELEASE; release_start_ = floor_; release_t_ = 0.0;
                std::cout << "[CORNER] RELEASE ("
                          << (aligned_ >= 2 ? "aligned" : (overturn ? "overturn" : (edge_detached ? "edge_detached" : "timeout"))) << ")\n";
            }
        }
        else {
            release_t_ += dt;
            const double r = std::clamp(1.0 - release_t_ / std::max(CORNER_RELEASE_TIME, 0.01), 0.0, 1.0);
            floor_ = release_start_ * r;
            if (release_t_ >= CORNER_RELEASE_TIME) {
                reset();
                cooldown_ = CORNER_COOLDOWN_S;
                std::cout << "[CORNER] ASSIST OFF\n";
            }
        }
        return floor_;
    }

private:
    Phase  phase_ = Phase::IDLE;
    int    dir_ = 0, pending_dir_ = 0, confirm_ = 0, aligned_ = 0;
    double elapsed_ = 0.0, floor_ = 0.0, release_start_ = 0.0, release_t_ = 0.0, cooldown_ = 0.0;
};

// ==========================================================
// OBSTACLE AVOIDANCE CONTROLLER
// ==========================================================
class ObstacleAvoidanceController {
public:
    enum class State {
        IDLE = 0, DODGE, PASSING, CLEARING
    };

    struct ObstacleResult {
        State       state             = State::IDLE;
        int         obstacle_side     = 0;
        int         dodge_dir         = 0;
        int         y_cut             = -1;
        int         dist_px           = -1;
        double      steer_bias_px     = 0.0;
        double      steer_floor_deg   = 0.0;
        double      steer_cap_deg     = 0.0;
        bool        lock_obstacle_dir = false;
        int         preferred_side    = 0;
        int         pass_count        = 0;
        bool        near_cut          = false;
        std::string state_str         = "IDLE";
    };

    double DODGE_BIAS_PX         = 24.0;   
    double DODGE_FLOOR_DEG       = 13.0;   
    double PASSING_BIAS_PX       = 12.0;   
    double DODGE_MIN_TIME        = 0.35;   
    double DODGE_MAX_TIME        = 1.20;   
    int    CUT_STEP_THRESHOLD_PX = 50;     
    int    CLEAR_CONFIRM_FRAMES  = 3;      

    void reset() {
        state_ = State::IDLE;
        obstacle_side_ = 0;
        dodge_dir_ = 0;
        y_cut_ = -1;
        dist_px_ = -1;
        elapsed_ = 0.0;
        clear_frames_ = 0;
        near_cut_current_ = false;
    }

    bool active() const { return state_ != State::IDLE; }
    State get_state() const { return state_; }
    int get_pass_count() const { return pass_count_; }

    ObstacleResult update(const cv::Mat& mask,
                          const std::vector<cv::Point>& left_pts,
                          const std::vector<cv::Point>& right_pts,
                          int lx1, int rx1, int lx2, int rx2, int lx_bot, int rx_bot,
                          int w, double dt,
                          double near_e = 0.0, double mid_e = 0.0, double far_e = 0.0) {
        ObstacleResult res;
        res.pass_count = pass_count_;
        if (mask.empty()) return res;

        const bool on_bridge = (std::abs(near_e) <= 6.0) &&
            ((std::abs(far_e) >= 75.0 && std::abs(near_e) <= 4.0) ||
             (std::abs(mid_e) >= 15.0 && std::abs(far_e) >= 65.0));
        if (on_bridge) {
            confirm_count_ = 0;
            return res;
        }

        const int h = mask.rows;
        elapsed_ += dt;

        auto get_row_bounds = [&](int y_row, int& lx, int& rx) -> bool {
            lx = -1; rx = -1;
            if (y_row < 0 || y_row >= h) return false;
            const uchar* ptr = mask.ptr<uchar>(y_row);
            int start = -1, max_len = 0;
            for (int x = 0; x < w; ++x) {
                if (ptr[x] > 0) {
                    if (start < 0) start = x;
                } else {
                    if (start >= 0) {
                        int len = x - start;
                        if (len > max_len && len > 20) {
                            max_len = len; lx = start; rx = x - 1;
                        }
                        start = -1;
                    }
                }
            }
            if (start >= 0) {
                int len = w - start;
                if (len > max_len && len > 20) {
                    max_len = len; lx = start; rx = w - 1;
                }
            }
            return (lx >= 0 && rx >= lx);
        };

        const int y_sample_start = static_cast<int>(h * 0.88);
        const int y_sample_end   = static_cast<int>(h * 0.56);
        const int step_y = 6;

        struct RowSample {
            int y, lx, rx, width;
            bool valid;
        };
        std::vector<RowSample> samples;
        for (int y = y_sample_start; y >= y_sample_end; y -= step_y) {
            int lx, rx;
            bool ok = get_row_bounds(y, lx, rx);
            samples.push_back({y, lx, rx, ok ? (rx - lx) : 0, ok});
        }

        int near_w_sum = 0, near_w_cnt = 0;
        int near_lx_sum = 0, near_rx_sum = 0;
        for (const auto& s : samples) {
            if (s.valid && s.y >= static_cast<int>(h * 0.78)) {
                near_w_sum += s.width;
                near_lx_sum += s.lx;
                near_rx_sum += s.rx;
                near_w_cnt++;
            }
        }
        const int avg_near_w  = near_w_cnt ? (near_w_sum / near_w_cnt) : (w / 2);
        const int avg_near_lx = near_w_cnt ? (near_lx_sum / near_w_cnt) : 0;
        const int avg_near_rx = near_w_cnt ? (near_rx_sum / near_w_cnt) : (w - 1);

        int detected_side = 0;
        int detected_y_cut = -1;
        int max_step_found = 0;

        for (size_t i = 1; i + 1 < samples.size(); ++i) {
            if (!samples[i].valid) continue;
            int prev_idx = static_cast<int>(i) - 2;
            if (prev_idx < 0) prev_idx = static_cast<int>(i) - 1;
            if (!samples[prev_idx].valid) continue;

            const int step_rx = samples[prev_idx].rx - samples[i].rx;
            const int step_lx = samples[i].lx - samples[prev_idx].lx;

            if (step_rx >= CUT_STEP_THRESHOLD_PX && step_rx > step_lx + 30 &&
                samples[i].width <= avg_near_w - 40) {
                if (step_rx > max_step_found) {
                    max_step_found = step_rx;
                    detected_side = 1;
                    detected_y_cut = samples[i].y;
                }
            }
            else if (step_lx >= CUT_STEP_THRESHOLD_PX && step_lx > step_rx + 30 &&
                     samples[i].width <= avg_near_w - 40) {
                if (step_lx > max_step_found) {
                    max_step_found = step_lx;
                    detected_side = -1;
                    detected_y_cut = samples[i].y;
                }
            }
        }

        bool near_is_cut_right = false;
        bool near_is_cut_left  = false;
        if (state_ == State::DODGE || state_ == State::PASSING) {
            if (obstacle_side_ == 1) {
                near_is_cut_right = (rx2 >= 0 && rx2 <= w - 1 - 90) || (rx_bot >= 0 && rx_bot <= w - 1 - 70) ||
                                    (avg_near_rx <= w - 1 - 80);
            }
            else if (obstacle_side_ == -1) {
                near_is_cut_left = (lx2 >= 0 && lx2 >= 90) || (lx_bot >= 0 && lx_bot >= 70) ||
                                   (avg_near_lx >= 80);
            }
        }
        near_cut_current_ = (obstacle_side_ == 1 && near_is_cut_right) || (obstacle_side_ == -1 && near_is_cut_left);

        if (state_ == State::IDLE) {
            if (detected_side != 0 && detected_y_cut > 0) {
                confirm_count_++;
                if (confirm_count_ >= 2) {
                    state_         = State::DODGE;
                    obstacle_side_ = detected_side;
                    dodge_dir_     = -detected_side;
                    y_cut_         = detected_y_cut;
                    dist_px_       = (h - 1) - y_cut_;
                    elapsed_       = 0.0;
                    clear_frames_  = 0;
                    std::cout << "[OBSTACLE] >> DETECTED OBSTACLE on " << (obstacle_side_ > 0 ? "RIGHT" : "LEFT")
                              << " at Y=" << y_cut_ << " (Dist: " << dist_px_ << "px) -> DODGE "
                              << (dodge_dir_ < 0 ? "LEFT" : "RIGHT") << "!\n";
                }
            } else {
                confirm_count_ = 0;
            }
        }
        else if (state_ == State::DODGE) {
            if (detected_side == obstacle_side_ && detected_y_cut > 0) {
                y_cut_   = detected_y_cut;
                dist_px_ = (h - 1) - y_cut_;
            }

            const bool shifted_to_clear_lane = (dodge_dir_ < 0 && near_e >= 6.0) ||
                                               (dodge_dir_ > 0 && near_e <= -6.0);
            if ((near_cut_current_ && elapsed_ >= 0.35) || (shifted_to_clear_lane && elapsed_ >= DODGE_MIN_TIME) ||
                elapsed_ >= DODGE_MAX_TIME) {
                state_        = State::PASSING;
                elapsed_      = 0.0;
                clear_frames_ = 0;
                std::cout << "[OBSTACLE] >> PULLING ALONGSIDE OBSTACLE -> STATE: PASSING (Near cut on "
                          << (obstacle_side_ > 0 ? "RIGHT" : "LEFT") << ", LOCKING ANTI-SWERVE!)\n";
            }
        }
        else if (state_ == State::PASSING) {
            bool clear_now = false;
            if (obstacle_side_ == 1) {
                clear_now = (rx2 >= w - 1 - 35) && (rx_bot >= w - 1 - 25) && (avg_near_w >= 280);
            } else {
                clear_now = (lx2 >= 0 && lx2 <= 35) && (lx_bot >= 0 && lx_bot <= 25) && (avg_near_w >= 280);
            }

            if (clear_now && !near_cut_current_) {
                clear_frames_++;
            } else {
                clear_frames_ = 0;
            }

            if (clear_frames_ >= CLEAR_CONFIRM_FRAMES || elapsed_ >= 3.20) {
                pass_count_++;
                state_        = State::CLEARING;
                elapsed_      = 0.0;
                clear_frames_ = 0;
                std::cout << "[OBSTACLE] >>> OBSTACLE #" << pass_count_
                          << " CLEARED! Near/Mid fully restored & stable -> Scanning for next obstacle...\n";
            }
        }
        else if (state_ == State::CLEARING) {
            if (detected_side != 0 && detected_y_cut > 0 && elapsed_ >= 0.25) {
                state_         = State::DODGE;
                obstacle_side_ = detected_side;
                dodge_dir_     = -detected_side;
                y_cut_         = detected_y_cut;
                dist_px_       = (h - 1) - y_cut_;
                elapsed_       = 0.0;
                clear_frames_  = 0;
                std::cout << "[OBSTACLE] >>> IMMEDIATELY ENCOUNTERED NEXT OBSTACLE on "
                          << (obstacle_side_ > 0 ? "RIGHT" : "LEFT") << " at Y=" << y_cut_
                          << " -> DODGE " << (dodge_dir_ < 0 ? "LEFT" : "RIGHT") << "!\n";
            } else if (elapsed_ >= 0.70) {
                state_ = State::IDLE;
                obstacle_side_ = 0;
                dodge_dir_ = 0;
                y_cut_ = -1;
                dist_px_ = -1;
                elapsed_ = 0.0;
                std::cout << "[OBSTACLE] >>> PATH FULLY CLEAR -> RETURN TO IDLE\n";
            }
        }

        res.state         = state_;
        res.obstacle_side = obstacle_side_;
        res.dodge_dir     = dodge_dir_;
        res.y_cut         = y_cut_;
        res.dist_px       = dist_px_;
        res.near_cut      = near_cut_current_;

        switch (state_) {
            case State::IDLE:
                res.state_str         = "IDLE";
                res.steer_bias_px     = 0.0;
                res.steer_floor_deg   = 0.0;
                res.steer_cap_deg     = 0.0;
                res.lock_obstacle_dir = false;
                res.preferred_side    = 0;
                break;

            case State::DODGE:
                res.state_str         = (dodge_dir_ < 0) ? "DODGE_LEFT" : "DODGE_RIGHT";
                res.steer_bias_px     = dodge_dir_ * DODGE_BIAS_PX;
                res.steer_floor_deg   = dodge_dir_ * DODGE_FLOOR_DEG;
                res.lock_obstacle_dir = true;
                res.preferred_side    = dodge_dir_;
                break;

            case State::PASSING:
                res.state_str         = (obstacle_side_ > 0) ? "PASS_LOCK_RIGHT" : "PASS_LOCK_LEFT";
                res.steer_bias_px     = dodge_dir_ * PASSING_BIAS_PX;
                res.steer_floor_deg   = (obstacle_side_ > 0) ? -2.0 : 2.0;
                res.lock_obstacle_dir = true;
                res.preferred_side    = dodge_dir_;
                break;

            case State::CLEARING:
                res.state_str         = "CLEARING_STABLE";
                res.steer_bias_px     = 0.0;
                res.steer_floor_deg   = 0.0;
                res.lock_obstacle_dir = false;
                res.preferred_side    = 0;
                break;
        }

        return res;
    }

private:
    State  state_             = State::IDLE;
    int    obstacle_side_     = 0;
    int    dodge_dir_         = 0;
    int    y_cut_             = -1;
    int    dist_px_           = -1;
    double elapsed_           = 0.0;
    int    confirm_count_     = 0;
    int    clear_frames_      = 0;
    int    pass_count_        = 0;
    bool   near_cut_current_  = false;
};

// ==========================================================
// LANE FOLLOWER
// ==========================================================
class LaneFollower {
public:
    struct Output {
        double angle = 0, speed = 0;
        bool valid = false, occl = false, forced_emer = false, emer_auth = false, losing = false;
        double near_e = 0, far_e = 0, eff_error = 0, coverage = 0; 
        double mid_e = 0;
        std::vector<std::pair<int, int>> centerline_pts;
        std::vector<std::pair<int, int>> left_line_pts;
        std::vector<std::pair<int, int>> right_line_pts;
        int proc_width = 0, proc_height = 0;
    };

    explicit LaneFollower(const Config& c = Config()) : c_(c), pid_(c), gate_(c) {
        speed_ = c_.min_speed;
        resetTracking();
    }

    void resetTracking() {
        pid_.reset(); gate_.reset();
        lastAngle_ = 0; curveSmooth_ = 0; prevValid_ = false;
        asymSmooth_ = 0; emerHold_ = 0;
        invalid_frames_ = 0;
    }
    double speed() const { return speed_; }

Output step(const cv::Mat& roadMask, double totalBiasPx = 0.0, bool isExecutingTurn = false, double turnFloorDeg = 0.0, double turnCapDeg = 0.0, int preferredSide = 0, bool lockObstacleDir = false, int obstacleSide = 0) {
    cv::Mat m = detail::preprocessMask(roadMask, c_);
    Output o;
    detail::Centerline cl;
    if (m.empty()) { 
        cl.valid = false; 
    } else {
        cl = detail::extractCenterline(m, curveSmooth_, c_, preferredSide);
    }

    const double nearE = cl.near_error;
    const double midE  = cl.mid_error;
    const double farE  = cl.far_error;
    curveSmooth_ = cl.curve_smooth;
    const bool valid = cl.valid;
    
    double lane_err = 0.0;
    
    const bool bridge_far_artefact = (std::abs(nearE) <= 6.0) &&
        ((std::abs(farE) >= 75.0 && std::abs(nearE) <= 4.0) ||
         (std::abs(midE) >= 15.0 && std::abs(farE) >= 65.0));

    const bool is_genuinely_opposite = (nearE * farE < 0.0) && (std::abs(nearE) > 3.0);

    const bool obstacle_one_side = !isExecutingTurn && !bridge_far_artefact &&
                                   (std::abs(nearE) <= 3.5) &&
                                   (std::abs(midE) >= 14.0) && (std::abs(farE) >= 28.0) &&
                                   (midE * farE > 0.0);

    if (bridge_far_artefact && !isExecutingTurn) {
        lane_err = (0.85 * nearE) + (0.15 * std::clamp(midE, -8.0, 8.0));
    } else if (obstacle_one_side) {
        double max_far_obs = std::max(std::abs(midE) * 2.0, 45.0);
        double safe_far = std::clamp(farE, -max_far_obs, max_far_obs);
        lane_err = (0.10 * nearE) + (0.50 * midE) + (0.40 * safe_far);
    } else if (is_genuinely_opposite) {
        if (midE * farE > 0.0 && std::abs(midE) > 6.0) {
            double capped_far = std::clamp(farE, -25.0, 25.0);
            lane_err = (0.45 * nearE) + (0.35 * midE) + (0.20 * capped_far);
        } else {
            lane_err = (0.65 * nearE) + (0.35 * midE);
        }
    } else {
        double base_far_cap = (!isExecutingTurn && !bridge_far_artefact && std::abs(midE) < 15.0 && std::abs(farE) >= 40.0) ? 65.0 : 30.0;
        double max_far_allowed = std::max(std::abs(midE) * 1.5, base_far_cap);
        double safe_far = std::clamp(farE, -max_far_allowed, max_far_allowed);
        lane_err = (0.40 * nearE) + (0.35 * midE) + (0.25 * safe_far);
    }

    const double abs_err = std::abs(lane_err);
    if (abs_err > 13.0) {
        double error_scale = std::min(1.0 + (abs_err - 13.0) * 0.027, 1.50);
        lane_err *= error_scale;
    }

    if (lockObstacleDir) {
        if (obstacleSide > 0 && nearE < 28.0) {
            lane_err = std::min(lane_err, 0.0);
        } else if (obstacleSide < 0 && nearE > -28.0) {
            lane_err = std::max(lane_err, 0.0);
        }
    } else if (isExecutingTurn) {
        if (totalBiasPx < -5.0 || turnFloorDeg < -0.5 || preferredSide < 0) {
            lane_err = std::min(lane_err, 0.0);
        } else if (totalBiasPx > 5.0 || turnFloorDeg > 0.5 || preferredSide > 0) {
            lane_err = std::max(lane_err, 0.0);
        }
    }

    double eff = lane_err + totalBiasPx;

    bool occl = gate_.step(eff, cl.coverage, cl.row_centers, valid, farE, nearE);
    if (occl) emerHold_ = c_.emg_hold_after_occl;
    else if (emerHold_ > 0) emerHold_--;

    if (!occl && valid && !m.empty()) {
        double asymRaw = detail::detectAsymmetry(m, 45.0, 0.22, 0.72);
        if (!bridge_far_artefact && !isExecutingTurn && std::abs(asymRaw) >= c_.asym_threshold) {
            asymSmooth_ = 0.65 * asymSmooth_ + 0.35 * asymRaw;
        } else {
            asymSmooth_ *= c_.asym_decay;
        }
        asymSmooth_ = std::clamp(asymSmooth_, -c_.asym_cap, c_.asym_cap);
    }
    
    double asymApplied = (isExecutingTurn || std::abs(asymSmooth_) < c_.asym_threshold) ? 0.0 : c_.asym_strength * asymSmooth_;
    eff += asymApplied;

    bool forcedEmer = (emerHold_ > 0);
    pid_.setForcedEmergency(forcedEmer);

    double angle;
    if (valid) {
        invalid_frames_ = 0;
        if (!prevValid_) { pid_.softReset(); gate_.reset(); }
        
        angle = pid_.step(eff, c_.kp, c_.rate_limit);
        
        double target = detail::planSpeed(curveSmooth_, c_.max_speed, c_.min_speed, c_.curve_scale);
        
        if (std::abs(nearE) > 15.0) {
            target = std::min(target, c_.min_speed + 1.0);
        }
        
        double errMult = detail::errorSpeedFactor(eff);
        double authSpeed = pid_.losing() ? 0.6 : 1.0;
        target = std::max(target * errMult * authSpeed, c_.min_speed);
        speed_ = 0.65 * speed_ + 0.35 * target;
    } else {
        invalid_frames_++;
        angle = lastAngle_;
        if (invalid_frames_ > 5) {
            speed_ = 0.0;
            forcedEmer = true; 
        } else {
            speed_ = std::max(c_.min_speed, 0.85 * speed_);
        }
    }
    
    if (turnFloorDeg < -0.5 && angle > turnFloorDeg) {
        angle = turnFloorDeg;
        pid_.overrideOutput(angle);
    } else if (turnFloorDeg > 0.5 && angle < turnFloorDeg) {
        angle = turnFloorDeg;
        pid_.overrideOutput(angle);
    }
    
    if (turnCapDeg > 0.5 && angle > turnCapDeg) {
        angle = turnCapDeg;
        pid_.overrideOutput(angle);
    } else if (turnCapDeg < -0.5 && angle < turnCapDeg) {
        angle = turnCapDeg;
        pid_.overrideOutput(angle);
    }

    if (lockObstacleDir) {
        if (obstacleSide > 0 && angle > 0.0) {
            angle = 0.0;
            pid_.overrideOutput(angle);
        } else if (obstacleSide < 0 && angle < 0.0) {
            angle = 0.0;
            pid_.overrideOutput(angle);
        }
    }
    
    if (valid || std::abs(turnFloorDeg) > 0.5) lastAngle_ = angle;
    prevValid_ = valid;

    o.angle = angle;
    o.speed = speed_;
    o.valid = valid;
    o.occl = occl;
    o.mid_e = midE;
    o.forced_emer = forcedEmer;
    o.emer_auth = pid_.emergencyAuthority();
    o.losing = pid_.losing();
    o.near_e = nearE;
    o.far_e = farE;
    o.eff_error = eff;
    o.coverage = cl.coverage;
    o.centerline_pts = cl.row_centers;
    o.left_line_pts  = cl.left_pts;
    o.right_line_pts = cl.right_pts;
    o.proc_width = m.cols;
    o.proc_height = m.rows;

    return o;
}

private:
    Config c_;
    SteeringPID pid_;
    OcclusionGate gate_;
    double speed_, lastAngle_, curveSmooth_, asymSmooth_;
    bool prevValid_;
    int emerHold_;
    int invalid_frames_ = 0;
};

}  // namespace v20
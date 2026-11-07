// ============================================================
// LaneFollowerV20.h — C++17 Lane Follower (Corner-Locked & Wheel-Clearance Aware)
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
#include <utility>
#include <vector>

namespace v20 {

struct Config {
    int  mask_width   = 320;
    bool keep_largest = true;

    double max_speed = 18.0;
    double min_speed = 11.0;

    double asym_strength = 0.15, asym_threshold = 8.0, asym_ema = 0.7;
    double asym_cap = 10.0, asym_decay = 0.80;

    double far_delta_clip = 45.0;
    double d_far_clip = 8.0, d_far_ema_alpha = 0.4, lead_rate_gain = 0.80;

    double early_curve_gain     = 0.55;  
    double early_curve_thresh   = 8.0;
    double narrow_road_width    = 190.0; 
    double standard_road_width  = 260.0;
    
    double wheel_clearance_px   = 120.0; 
    double wheel_guard_gain     = 1.55;  

    // Ngưỡng an toàn lề đường khi rẽ (tương đương khoảng 4cm ngoài đời)
    double margin_safe_threshold_px = 16.0; 
    // Lực phản kháng kéo lại nếu vi phạm ngưỡng an toàn (phải mềm)
    double margin_pushback_gain = 0.45;

    double hard_step_px = 30.0, err_step_px = 18.0, cov_drop = 0.22, jagged_px = 15.0;
    int    occl_hangover = 4;
    double far_outlier_px = 40.0, far_outlier_near_px = 20.0, far_spread_px = 40.0;
    int    far_window = 5;
    double cov_ema_alpha = 0.9;

    int emergency_streak_req = 2;
    int emg_hold_after_occl  = 5;

    double kp = 0.45, ki = 0.0, kd = 0.70; 
    double integral_limit = 2.0, derivative_alpha = 0.30;
    double output_limit = 25.0, rate_limit = 24.0, deadband = 1.0;
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
    double v = (double)sorted[lo] +
               (lo + 1 < sorted.size() ? fr * (double)(sorted[lo + 1] - sorted[lo]) : 0.0);
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
        for (int x = 0; x < outW; x++)
            if (labels[(size_t)y * outW + x] != best) row[x] = 0;
    }
    return out;
}

struct Centerline {
    double corner_error = 0, near_error = 0, mid_error = 0, far_error = 0;
    double curve_error = 0, curve_smooth = 0, coverage = 0;
    bool valid = false;
    std::vector<std::pair<int, int>> row_centers;
    std::vector<std::pair<int, int>> left_pts;
    std::vector<std::pair<int, int>> right_pts;
    
    double road_width_near = 0.0;
    double road_width_far  = 0.0;
    
    double dist_to_left    = 0.0;
    double dist_to_right   = 0.0;
    int left_near_x        = -1;
    int left_near_y        = -1;
    int right_near_x       = -1;
    int right_near_y       = -1;
    
    double left_corner_x   = 0.0;
    double right_corner_x  = 0.0;
};

inline bool largestRun(const uchar* row, int w, int& bs, int& be) {
    int bestLen = -1, s = -1;
    bs = be = -1;
    for (int x = 0; x <= w; x++) {
        bool on = (x < w) && row[x] > 0;
        if (on && s < 0) s = x;
        if (!on && s >= 0) {
            int len = (x - 1) - s;
            if (len > bestLen) { bestLen = len; bs = s; be = x - 1; }
            s = -1;
        }
    }
    return bs >= 0;
}

inline Centerline extractCenterline(const cv::Mat& m, double prevCurveSmooth, const Config& c) {
    Centerline r;
    const int h = m.rows, w = m.cols;
    const double halfW = (w - 1) * 0.5;
    long nz = 0;
    
    for (int y = h - 1; y >= 0; y--) {
        const uchar* row = m.ptr<uchar>(y);
        for (int x = 0; x < w; x++) nz += row[x] > 0;
        int s, e;
        if (largestRun(row, w, s, e)) {
            r.row_centers.push_back({(s + e) / 2, y});
            r.left_pts.push_back({s, y});
            r.right_pts.push_back({e, y});
        }
    }
    r.coverage = (double)nz / ((double)h * (double)w);
    if ((int)r.row_centers.size() < c.min_rows || r.coverage < c.coverage_threshold) {
        r.valid = false;
        return r;
    }

    std::vector<int> ys;
    ys.reserve(r.row_centers.size());
    for (auto& p : r.row_centers) ys.push_back(p.second);
    std::sort(ys.begin(), ys.end());
    int yBot = percentileInt(ys, 0.98);
    int yTop = percentileInt(ys, 0.02);
    int span = std::max(yBot - yTop, 1);

    int ySplitCorner = yBot - std::max(4, (int)(span * 0.18));
    int ySplitNear   = yBot - (int)(span * 0.35);
    int ySplitFar    = yTop + (int)(span * 0.30);

    std::vector<int> cornerL, cornerR;
    std::vector<int> nearC_all, nearC_unclipped, midC, farC;
    std::vector<int> nearWidths, farWidths;
    std::vector<int> nearLefts, nearRights;

    for (size_t i = 0; i < r.row_centers.size(); ++i) {
        int y  = r.row_centers[i].second;
        int cx = r.row_centers[i].first;
        int lx = r.left_pts[i].first;
        int rx = r.right_pts[i].first;
        int width = rx - lx;

        if (y >= ySplitCorner) {
            cornerL.push_back(lx);
            cornerR.push_back(rx);
            
            if (r.left_near_x == -1) {
                r.left_near_x = lx;
                r.left_near_y = y;
                r.right_near_x = rx;
                r.right_near_y = y;
            }
        }

        if (y >= ySplitNear) {
            nearC_all.push_back(cx);
            if (lx > 2 || rx < w - 3) {
                nearC_unclipped.push_back(cx);
            }
            nearWidths.push_back(width);
            nearLefts.push_back(lx); 
            nearRights.push_back(rx);
        } else if (y <= ySplitFar) {
            farC.push_back(cx);
            farWidths.push_back(width);
        } else {
            midC.push_back(cx);
            if (lx > 2 || rx < w - 3) {
                nearC_unclipped.push_back(cx);
            }
        }
    }
    
    if (r.left_near_x == -1 && !r.left_pts.empty()) {
        r.left_near_x = r.left_pts.front().first;
        r.left_near_y = r.left_pts.front().second;
        r.right_near_x = r.right_pts.front().first;
        r.right_near_y = r.right_pts.front().second;
    }

    r.left_corner_x  = cornerL.empty() ? 0.0 : medianInts(cornerL);
    r.right_corner_x = cornerR.empty() ? (double)(w - 1) : medianInts(cornerR);
    r.corner_error   = 0.5 * (r.left_corner_x + r.right_corner_x) - halfW;

    double nearCx = halfW;
    if (!nearC_unclipped.empty()) {
        double rawNearCx = nearC_all.empty() ? halfW : medianInts(nearC_all);
        double unclipCx  = medianInts(nearC_unclipped);
        if (r.left_corner_x <= 2.0 && r.right_corner_x >= w - 3.0) {
            nearCx = 0.40 * rawNearCx + 0.60 * unclipCx;
        } else {
            nearCx = rawNearCx;
        }
    } else if (!nearC_all.empty()) {
        nearCx = medianInts(nearC_all);
    }

    double midCx = midC.empty() ? nearCx : medianInts(midC);
    double farCx = farC.empty() ? midCx  : medianInts(farC);

    r.near_error = nearCx - halfW;
    r.mid_error  = midCx  - halfW;
    r.far_error  = farCx  - halfW;

    r.curve_error  = farCx - nearCx;
    r.curve_smooth = 0.85 * prevCurveSmooth + 0.15 * r.curve_error;

    r.road_width_near = nearWidths.empty() ? 0.0 : medianInts(nearWidths);
    r.road_width_far  = farWidths.empty()  ? 0.0 : medianInts(farWidths);
    
    double avg_lx = nearLefts.empty() ? 0.0 : medianInts(nearLefts);
    double avg_rx = nearRights.empty() ? (w-1) : medianInts(nearRights);
    
    r.dist_to_left    = avg_lx; 
    r.dist_to_right   = (w - 1) - avg_rx; 

    r.valid = true;
    return r;
}

inline double detectAsymmetry(const cv::Mat& m, double maxBias = 45.0,
                              double y0f = 0.18, double y1f = 0.60) {
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

}  // namespace detail

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

        std::vector<std::pair<int, int>> centerline_pts;
        std::vector<std::pair<int, int>> left_line_pts;
        std::vector<std::pair<int, int>> right_line_pts;
        
        double road_width_near = 0.0;
        double road_width_far  = 0.0;
        double dist_to_left    = 0.0;
        double dist_to_right   = 0.0;
        
        int left_near_x        = 0;
        int left_near_y        = 0;
        int right_near_x       = 0;
        int right_near_y       = 0;
        
        bool is_narrow_road    = false;
        bool is_curve_ahead    = false;
        int proc_width = 0;
        int proc_height = 0;
    };

    explicit LaneFollower(const Config& c = Config()) : c_(c), pid_(c), gate_(c) {
        speed_ = c_.min_speed;
        resetTracking();
    }
    void resetTracking() {
        pid_.reset(); gate_.reset();
        lastAngle_ = 0; curveSmooth_ = 0; prevValid_ = false;
        prevFarE_ = 0; dFarFilt_ = 0; asymSmooth_ = 0; emerHold_ = 0;
    }
    double speed() const { return speed_; }
    void setSpeed(double s) { speed_ = s; }

    Output step(const cv::Mat& roadMask, double externalBiasPx = 0.0) {
        cv::Mat m = detail::preprocessMask(roadMask, c_);
        Output o;
        detail::Centerline cl;
        if (m.empty()) { cl.valid = false; }
        else cl = detail::extractCenterline(m, curveSmooth_, c_);

        const double nearE = cl.near_error, midE = cl.mid_error, farE = cl.far_error;
        curveSmooth_ = cl.curve_smooth;
        const bool valid = cl.valid;

        // 1. NHẬN DIỆN ĐƯỜNG HẸP & MẤT LÀN
        bool isNarrow = (cl.road_width_near > 0.0 && cl.road_width_near < c_.narrow_road_width) ||
                        (cl.road_width_far  > 0.0 && cl.road_width_far  < c_.narrow_road_width * 0.75);

        bool missing_left  = (cl.dist_to_left <= 2.0);
        bool missing_right = (cl.dist_to_right <= 2.0);
        bool is_single_lane = (missing_left ^ missing_right) && isNarrow;

        double wheelGuardBias = 0.0;

        // 2. DYNAMIC WHEEL GUARD (CHỐNG OUTLINE KHI RẼ GẮT)
        // Nếu externalBiasPx đang điều hướng rẽ trái/phải, ta giám sát khoảng cách lề tương ứng.
        if (externalBiasPx < -5.0) { 
            // Đang rẽ trái (Steer Bias Âm) -> Giám sát lề TRÁI
            if (cl.dist_to_left > c_.margin_safe_threshold_px) {
                // Vi phạm ngưỡng, đẩy mềm về phía mốc an toàn
                wheelGuardBias = +c_.margin_pushback_gain * (cl.dist_to_left - c_.margin_safe_threshold_px);
            }
        } else if (externalBiasPx > 5.0) {
            // Đang rẽ phải (Steer Bias Dương) -> Giám sát lề PHẢI
            if (cl.dist_to_right > c_.margin_safe_threshold_px) {
                // Vi phạm ngưỡng, đẩy mềm về phía mốc an toàn
                wheelGuardBias = -c_.margin_pushback_gain * (cl.dist_to_right - c_.margin_safe_threshold_px);
            }
        } else {
            // Trường hợp chạy thẳng thông thường
            if (is_single_lane) {
                double targetMargin = 45.0; 
                if (missing_left && !missing_right) {
                    wheelGuardBias = -c_.wheel_guard_gain * (cl.dist_to_right - targetMargin);
                } else if (missing_right && !missing_left) {
                    wheelGuardBias = +c_.wheel_guard_gain * (cl.dist_to_left - targetMargin);
                }
                wheelGuardBias = std::clamp(wheelGuardBias, -25.0, 25.0);
            } else {
                double marginDiff = cl.dist_to_left - cl.dist_to_right;
                wheelGuardBias = 0.15 * marginDiff;
            }
        }

        // 3. TÍNH SAI SỐ TỔNG
        double posError = 0.55 * nearE + 0.30 * cl.corner_error + 0.15 * midE 
                        + wheelGuardBias + externalBiasPx;

        // 4. ĐÓN CUA SỚM
        double rawCurveDiff = 0.50 * (midE - nearE) + 0.50 * (farE - midE);
        double clippedCurveDiff = std::clamp(rawCurveDiff, -c_.far_delta_clip, c_.far_delta_clip);
        bool isCurveAhead = std::abs(rawCurveDiff) > c_.early_curve_thresh;

        double dFarRaw = (valid && prevValid_) ? (farE - prevFarE_) : 0.0;
        double dFar = std::clamp(dFarRaw, -c_.d_far_clip, c_.d_far_clip);
        dFarFilt_ = c_.d_far_ema_alpha * dFar + (1.0 - c_.d_far_ema_alpha) * dFarFilt_;
        prevFarE_ = farE;

        double curveLead = 0.0;
        if (isCurveAhead && std::abs(externalBiasPx) < 1.0) {
            curveLead = c_.early_curve_gain * clippedCurveDiff + c_.lead_rate_gain * dFarFilt_;
        }

        if (posError * curveLead < 0.0) {
            double offTrackRatio = std::clamp((std::abs(nearE) - 18.0) / 25.0, 0.0, 0.90);
            curveLead *= (1.0 - offTrackRatio);
        }

        double blended = posError + curveLead;

        bool occl = gate_.step(blended, cl.coverage, cl.row_centers, valid, farE, nearE);
        if (occl) emerHold_ = c_.emg_hold_after_occl;
        else if (emerHold_ > 0) emerHold_--;

        if (!occl && valid && !m.empty()) {
            double asymRaw = detail::detectAsymmetry(m);
            asymSmooth_ = c_.asym_ema * asymSmooth_ + (1.0 - c_.asym_ema) * asymRaw;
            if (std::abs(nearE) < 5.0 && std::abs(asymSmooth_) < c_.asym_cap) asymSmooth_ *= c_.asym_decay;
            asymSmooth_ = std::clamp(asymSmooth_, -c_.asym_cap, c_.asym_cap);
        }
        
        double asymApplied = (std::abs(asymSmooth_) < c_.asym_threshold) ? 0.0 : c_.asym_strength * asymSmooth_;
        if (posError * asymApplied < 0.0 && std::abs(nearE) > 25.0) {
            asymApplied = 0.0;
        }
        double eff = blended + asymApplied;

        // 5. PID 
        double currentKp = c_.kp;
        double currentRateLim = c_.rate_limit;
        
        if (isNarrow || std::abs(wheelGuardBias) > 5.0) {
            currentKp = c_.kp * 1.30; 
            currentRateLim = c_.rate_limit * 1.1; 
        }

        bool forcedEmer = (emerHold_ > 0);
        pid_.setForcedEmergency(forcedEmer);

        double angle;
        if (valid) {
            if (!prevValid_) { pid_.softReset(); gate_.reset(); }
            angle = pid_.step(eff, currentKp, currentRateLim);
            double target = detail::planSpeed(curveSmooth_, c_.max_speed, c_.min_speed, c_.curve_scale);
            double errMult = detail::errorSpeedFactor(eff);
            double narrowMult = isNarrow ? 0.72 : 1.0;
            double authSpeed = pid_.losing() ? 0.6 : 1.0;
            target = std::max(target * errMult * narrowMult * authSpeed, c_.min_speed);

            speed_ = 0.65 * speed_ + 0.35 * target;
            lastAngle_ = angle;
        } else {
            angle = lastAngle_;
            speed_ = std::max(c_.min_speed, 0.85 * speed_);
        }
        prevValid_ = valid;

        o.angle = angle; o.speed = speed_;
        o.valid = valid; o.occl = occl; o.forced_emer = forcedEmer;
        o.emer_auth = pid_.emergencyAuthority(); o.losing = pid_.losing();
        o.near_e = nearE; o.far_e = farE; o.eff_error = eff; o.coverage = cl.coverage;
        o.centerline_pts = cl.row_centers;
        o.left_line_pts  = cl.left_pts;
        o.right_line_pts = cl.right_pts;
        o.road_width_near = cl.road_width_near;
        o.road_width_far  = cl.road_width_far;
        
        o.dist_to_left    = cl.dist_to_left;
        o.dist_to_right   = cl.dist_to_right;
        o.left_near_x     = cl.left_near_x;
        o.left_near_y     = cl.left_near_y;
        o.right_near_x    = cl.right_near_x;
        o.right_near_y    = cl.right_near_y;
        
        o.is_narrow_road  = isNarrow;
        o.is_curve_ahead  = isCurveAhead;
        o.proc_width = m.cols;
        o.proc_height = m.rows;

        return o;
    }

private:
    Config c_;
    SteeringPID pid_;
    OcclusionGate gate_;
    double speed_, lastAngle_, curveSmooth_, prevFarE_, dFarFilt_, asymSmooth_;
    bool prevValid_;
    int emerHold_;
};

}  // namespace v20

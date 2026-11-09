// ============================================================
// LaneFollowerV20.h  —  header-only C++17 port of ucr_main_v20.py
//
// Logic giữ nguyên v20:
//   keep_largest_component -> extract_centerline (near/far/curve)
//   -> lead (far_delta + d_far) -> OcclusionGate (5 tín hiệu + hangover)
//   -> asymmetry (cap + decay) -> SteeringPID
//        (AuthorityManager "losing", emergency authority, forced emergency
//         khi occl, soft_reset sau khi mất mask)
//   -> plan_speed * error_speed_factor * authority speed, EMA 0.7/0.3
//   mask invalid -> giữ last_angle, speed *= 0.85   (KHÔNG có blind mode)
//
// Dùng:
//   #include "control/LaneFollowerV20.h"
//   v20::LaneFollower follower;
//   auto out = follower.step(roadMask);
//   servo.SetAngle(std::lround(out.angle));
//   motor.SetSpeedCms(out.speed);
//
// Chỉ dùng cv::Mat cơ bản (rows/cols/type/ptr) -> không phụ thuộc hàm
// imgproc, build được với OpenCV 4.1.1 trên Jetson.
//
// >>> CHANGED: step() được thiết kế để gọi ở TỐC ĐỘ CỐ ĐỊNH (~10 Hz).
//     Tất cả alpha/rate-limit bên trong là per-step, không phải per-dt.
//     Caller phải ghim tick bằng steady_clock, không gọi theo inference rate.
// ============================================================
#pragma once

#ifdef LANEV20_USE_SHIM
#include "cvshim.hpp"      // chỉ dùng khi test trên PC không có OpenCV
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

// ------------------------------------------------------------
// Config: mặc định = hằng số của ucr_main_v20.py
// ------------------------------------------------------------
struct Config {
    // --- mask preprocessing ---
    int  mask_width   = 320;    // resize (nearest) mask về rộng 320 để ngưỡng px của v20 còn đúng; 0 = tắt
    bool keep_largest = true;   // v20 get_segmentation: chỉ giữ thành phần liên thông lớn nhất

    // >>> CHANGED: expose kept_frac để caller phát hiện intersection/branching.
    //     Nếu kept_frac < branch_frac_thresh (thành phần lớn nhất chiếm <60%
    //     tổng diện tích mask), coi như mask đang phân nhánh (ngã 3/4).
    //     Mặc định tắt (false) để giữ hành vi cũ. Bật khi test intersection.
    double branch_frac_thresh       = 0.60;
    bool   keep_branch_when_detected = false;

    // >>> CHANGED: STEER_GAIN cũ (test_run.cpp) chuyển vào config.
    //     1.0 = parity với sim v20. 2.2 = tuning hiện tại của xe.
    //     Calibrate empirically: đo yaw-rate thực của servo ở từng góc,
    //     rồi tune để PID output 1 đơn vị ≈ 1° lái thực.
    //     out.angle trong Output đã được nhân post_pid_gain.
    double post_pid_gain = 2.2;

    // --- speed ---
    double max_speed = 26.0;    // ĐỔI cho xe thật nếu cần (đơn vị = đơn vị Motor::SetSpeedCms)
    double min_speed = 5.0;     // <-- Đã tăng lên 5.0 để không bị dừng khi vào cua

    // --- lead / asymmetry ---
    double asym_strength = 0.20, asym_threshold = 8.0, asym_ema = 0.7;
    double asym_cap = 12.0, asym_decay = 0.80;
    double far_delta_clip = 25.0, d_far_clip = 6.0, d_far_ema_alpha = 0.4, lead_rate_gain = 0.8;

    // --- occlusion gate ---
    double hard_step_px = 30.0, err_step_px = 18.0, cov_drop = 0.22, jagged_px = 15.0;
    int    occl_hangover = 4;
    double far_outlier_px = 40.0, far_outlier_near_px = 20.0, far_spread_px = 40.0;
    int    far_window = 5;
    double cov_ema_alpha = 0.9;

    // --- emergency ---
    int emergency_streak_req = 2;
    int emg_hold_after_occl  = 5;

    // --- PID ---
    double kp = 0.32, ki = 0.0, kd = 0.6;
    double integral_limit = 2.0, derivative_alpha = 0.55;
    double output_limit = 22.0, rate_limit = 10.0, deadband = 1.5;
    double emergency_output_limit = 25.0, emergency_rate_limit = 20.0, emergency_error_px = 30.0;

    // --- authority ---
    double auth_tau = 0.35, auth_sat_frac = 0.9;
    int    auth_sat_frames = 3;
    double auth_lose_enter_px = 30.0, auth_lose_exit_px = 20.0;

    // --- centerline ---
    int    min_rows = 20;
    double coverage_threshold = 0.18;
    double curve_scale = 80.0;
};

namespace detail {

inline double medianInts(std::vector<int> v) {          // = np.median
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2) ? (double)v[n / 2] : ((double)v[n / 2 - 1] + (double)v[n / 2]) / 2.0;
}

inline int percentileInt(const std::vector<int>& sorted, double q) {   // = int(np.percentile(a, q*100))
    double pos = q * (double)(sorted.size() - 1);
    size_t lo = (size_t)pos;
    double fr = pos - (double)lo;
    double v = (double)sorted[lo] +
               (lo + 1 < sorted.size() ? fr * (double)(sorted[lo + 1] - sorted[lo]) : 0.0);
    return (int)v;
}

inline double sgn(double x) { return (x > 0) - (x < 0); }

// >>> CHANGED: thêm out-param kept_frac và branch-keep logic.
// Nearest-resize về rộng cfg.mask_width, nhị phân hoá (>0 -> 255), giữ thành phần lớn nhất
// (trừ khi phát hiện nhánh và keep_branch_when_detected = true).
inline cv::Mat preprocessMask(const cv::Mat& in, const Config& c, double& kept_frac_out) {
    kept_frac_out = 1.0;
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

    // keep_largest_component: flood-fill 8-liên-thông, nhãn theo thứ tự quét raster (như cv2)
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
    int total_area = 0;
    for (int i = 1; i <= nLabels; i++) total_area += areas[i];
    for (int i = 2; i <= nLabels; i++) if (areas[i] > areas[best]) best = i;   // argmax: lấy cái đầu khi hoà
    kept_frac_out = (total_area > 0) ? (double)areas[best] / (double)total_area : 1.0;

    // >>> CHANGED: nếu phát hiện nhánh và config cho phép, giữ nguyên tất cả component.
    if (c.keep_branch_when_detected && kept_frac_out < c.branch_frac_thresh) {
        return out;   // out vẫn là mask nhị phân ban đầu, không xoá gì
    }

    for (int y = 0; y < outH; y++) {
        uchar* row = out.ptr<uchar>(y);
        for (int x = 0; x < outW; x++)
            if (labels[(size_t)y * outW + x] != best) row[x] = 0;
    }
    return out;
}

struct Centerline {
    double near_error = 0, far_error = 0, curve_error = 0, curve_smooth = 0, coverage = 0;
    bool valid = false;
    std::vector<std::pair<int, int>> row_centers;       // (cx, y), từ đáy lên
};

// largest_run_bounds: đoạn liên tiếp dài nhất của 1 hàng (hoà -> đoạn đầu)
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
    long nz = 0;
    for (int y = h - 1; y >= 0; y--) {
        const uchar* row = m.ptr<uchar>(y);
        for (int x = 0; x < w; x++) nz += row[x] > 0;
        int s, e;
        if (largestRun(row, w, s, e)) r.row_centers.push_back({(s + e) / 2, y});
    }
    r.coverage = (double)nz / ((double)h * (double)w);
    if ((int)r.row_centers.size() < c.min_rows || r.coverage < c.coverage_threshold) {
        r.valid = false;                                 // curve_smooth = 0 như Python
        return r;
    }
    std::vector<int> ys;
    ys.reserve(r.row_centers.size());
    for (auto& p : r.row_centers) ys.push_back(p.second);
    std::sort(ys.begin(), ys.end());
    int yBot = percentileInt(ys, 0.98);
    int yTop = percentileInt(ys, 0.02);
    int span = std::max(yBot - yTop, 1);
    int ySplitNear = yBot - (int)(span * 0.45);
    int ySplitFar  = yTop + (int)(span * 0.35);

    std::vector<int> nearC, farC;
    for (auto& p : r.row_centers) {
        if (p.second >= ySplitNear) nearC.push_back(p.first);
        if (p.second <= ySplitFar)  farC.push_back(p.first);
    }
    int nearCx = nearC.empty() ? -1 : (int)medianInts(nearC);
    int farCx  = farC.empty()  ? -1 : (int)medianInts(farC);
    r.near_error = (nearCx != -1) ? (nearCx - w / 2) : 0;
    r.far_error  = (farCx  != -1) ? (farCx  - w / 2) : 0;
    int topCx = farC.empty()  ? 0 : (int)medianInts(farC);
    int botCx = nearC.empty() ? 0 : (int)medianInts(nearC);
    r.curve_error  = topCx - botCx;
    r.curve_smooth = 0.90 * prevCurveSmooth + 0.10 * r.curve_error;
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

// ------------------------------------------------------------
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
        if (rows.size() >= 6) {                          // mean(|diff(cx, n=2)|)
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
    // giữ prev_out / authority / emg_streak: dùng sau khi mất mask (không giật góc lái)
    void softReset() { integral_ = prevErr_ = filtD_ = 0; first_ = true; }
    void setForcedEmergency(bool f) { forced_ = f; }
    bool losing() const { return auth_.losing(); }
    bool emergencyAuthority() const { return emgAuth_; }

    double step(double error) {
        double kpEff = c_.kp * (auth_.losing() ? 1.5 : 1.0);
        double kdEff = c_.kd * (auth_.losing() ? 0.7 : 1.0);
        double outLim, rateLim;
        if (forced_) {
            emgStreak_ = c_.emergency_streak_req;
            outLim = c_.emergency_output_limit; rateLim = c_.emergency_rate_limit; emgAuth_ = true;
        } else {
            if (std::abs(error) > c_.emergency_error_px) emgStreak_++; else emgStreak_ = 0;
            if (emgStreak_ >= c_.emergency_streak_req) {
                outLim = c_.emergency_output_limit; rateLim = c_.emergency_rate_limit; emgAuth_ = true;
            } else { outLim = c_.output_limit; rateLim = c_.rate_limit; emgAuth_ = false; }
        }
        double sg      = detail::sgn(error);
        double abs_err = std::abs(error);
        double e       = (abs_err <= c_.deadband) ? 0.0 : (error - sg * c_.deadband);
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

// ------------------------------------------------------------
class LaneFollower {
public:
    struct Output {
        double angle = 0, speed = 0;               // angle: độ (đã nhân post_pid_gain), dương = phải
        bool valid = false, occl = false, forced_emer = false, emer_auth = false, losing = false;
        double near_e = 0, far_e = 0, eff_error = 0, coverage = 0;

        // >>> CHANGED: expose post-gain pre-gain và kept_frac
        double angle_pre_gain = 0;                 // PID-space angle trước post_pid_gain
        double kept_frac = 1.0;                    // diện tích largest component / tổng (1.0 nếu tắt keep_largest)

        std::vector<std::pair<int, int>> centerline_pts;
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

    // >>> CHANGED: gọi step() ở tần số CỐ ĐỊNH (~10 Hz). Xem docstring ở đầu file.
    Output step(const cv::Mat& roadMask) {
        double kept_frac = 1.0;
        cv::Mat m = detail::preprocessMask(roadMask, c_, kept_frac);
        Output o;
        detail::Centerline cl;
        if (m.empty()) { cl.valid = false; }
        else cl = detail::extractCenterline(m, curveSmooth_, c_);

        const double nearE = cl.near_error, farE = cl.far_error;
        curveSmooth_ = cl.curve_smooth;
        const bool valid = cl.valid;

        double farDelta = std::clamp(farE - nearE, -c_.far_delta_clip, c_.far_delta_clip);
        double dFarRaw = (valid && prevValid_) ? (farE - prevFarE_) : 0.0;
        double dFar = std::clamp(dFarRaw, -c_.d_far_clip, c_.d_far_clip);
        dFarFilt_ = c_.d_far_ema_alpha * dFar + (1.0 - c_.d_far_ema_alpha) * dFarFilt_;
        double lead = 0.5 * farDelta + c_.lead_rate_gain * dFarFilt_;
        double blended = nearE + lead;
        prevFarE_ = farE;

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
        double eff = blended + asymApplied;

        bool forcedEmer = emerHold_ > 0;
        pid_.setForcedEmergency(forcedEmer);

        double angle_pre_gain = 0.0;
        double angle = 0.0;
        if (valid) {
            if (!prevValid_) { pid_.softReset(); gate_.reset(); }
            angle_pre_gain = pid_.step(eff);
            // >>> CHANGED: post_pid_gain áp dụng ngay tại đây. out.angle là "góc lái thực"
            //     (vẫn theo quy ước dấu PID: dương = phải). Caller chỉ cần round/clamp/invert.
            angle = angle_pre_gain * c_.post_pid_gain;
            double target = detail::planSpeed(curveSmooth_, c_.max_speed, c_.min_speed, c_.curve_scale);
            double errMult = detail::errorSpeedFactor(eff);
            double authSpeed = pid_.losing() ? 0.6 : 1.0;
            double anomSpeed = (std::abs(asymSmooth_) > 15) ? 0.5 : 1.0;
            target = std::max(target * errMult * std::min(authSpeed, anomSpeed), 0.0);
            speed_ = 0.7 * speed_ + 0.3 * target;
            lastAngle_ = angle;
        } else {
            angle = lastAngle_;
            angle_pre_gain = (c_.post_pid_gain > 0.0) ? lastAngle_ / c_.post_pid_gain : lastAngle_;
            speed_ = 0.85 * speed_;
        }
        prevValid_ = valid;

        o.angle = angle;
        o.angle_pre_gain = angle_pre_gain;
        o.kept_frac = kept_frac;
        o.speed = speed_;
        o.valid = valid; o.occl = occl; o.forced_emer = forcedEmer;
        o.emer_auth = pid_.emergencyAuthority(); o.losing = pid_.losing();
        o.near_e = nearE; o.far_e = farE; o.eff_error = eff; o.coverage = cl.coverage;

        o.centerline_pts = cl.row_centers;
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

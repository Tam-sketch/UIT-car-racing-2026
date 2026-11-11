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

    double asym_strength = 0.55, asym_threshold = 6.0, asym_ema = 0.85;
    double asym_cap = 26.0, asym_decay = 0.92;

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

// ... [Giữ nguyên namespace detail với medianInts, percentileInt, sgn, preprocessMask, Centerline, extractCenterline, detectAsymmetry, planSpeed, errorSpeedFactor] ...
// (Lưu ý: Bạn copy nguyên khối namespace detail từ file cũ vào đây, không thay đổi)
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

            // Tìm TẤT CẢ các khoảng trống trên hàng ngang thay vì chỉ 1 khoảng lớn nhất
            int bestLen = -1, bs = -1, be = -1;
            int s = -1;
            for (int x = 0; x <= w; x++) {
                bool on = (x < w) && row[x] > 0;
                if (on && s < 0) s = x;
                if (!on && s >= 0) {
                    int len = (x - 1) - s;
                    // Nếu đang trong trạng thái né dạt về một bên (preferred_side != 0):
                    // Ưu tiên dải trống nằm hẳn về bên đó để tránh bị nhảy tâm
                    bool take_this = false;
                    if (preferred_side < 0 && (x - 1) < halfW + 30) take_this = true; // Ưu tiên nửa trái
                    else if (preferred_side > 0 && s > halfW - 30)  take_this = true; // Ưu tiên nửa phải
                    else if (len > bestLen) take_this = true;

                    if (take_this && len > 20) { // Bỏ qua đốm nhiễu < 20px
                        bestLen = len; bs = s; be = x - 1;
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

// ... [Giữ nguyên OcclusionGate, AuthorityManager, SteeringPID] ...
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
// LANE TURN CONTROLLER (Hỗ trợ Roundabout 4 pha + Đếm T-Junction không biển)
// ==========================================================
class LaneTurnController {
public:
    enum class State { IDLE, WAIT_CURVE, WAIT_STRAIGHT, EXECUTE, SETTLE, DONE };
    enum class RoundaboutPhase { NONE = 0, WAIT_ENTRY, ENTER_RIGHT, CIRCLE_LEFT, EXIT_RIGHT };

    // ---------------- GÓC QUÀNH T-JUNCTION / NGÃ RẼ THƯỜNG ----------------
    double TURN_STEER_DEG      = 35.0;   // Góc sàn khi quành T-junction / ngã rẽ (đủ gắt cho cả đường lớn)
    double TURN_RAMP_DEG_PER_S = 95.0;
    double MIN_TURN_TIME       = 0.8;
    double TURN_BIAS_PX        = 38.0;
    double turn_floor_current  = 0.0;

    // ---------------- THÔNG SỐ RIÊNG CHO ROUNDABOUT (Đồng bộ theo hình học đảo thực tế) ----------------
    double ROUNDABOUT_ENTER_DEG       = 28.0;  // Góc sàn ôm phải vòng qua mũi đảo khi bắt đầu nhập xuyến
    double ROUNDABOUT_MAX_ENTER_DEG   = 0.0;   // Không khóa trần lúc ôm mũi đảo (mũi đảo cần 35-45 độ)
    double ROUNDABOUT_CIRCLE_DEG      = 6.0;   // Góc giữ nhẹ bên trái khi đi dọc thân đảo để lướt qua nhánh phụ
    double ROUNDABOUT_EXIT_DEG        = 20.0;  // Góc sàn ôm phải theo đường cong thoát vòng xuyến
    double ROUNDABOUT_ENTER_BIAS_PX   = 24.0;
    double ROUNDABOUT_CIRCLE_BIAS_PX  = 8.0;
    double ROUNDABOUT_EXIT_BIAS_PX    = 20.0;
    double RB_ENTER_MIN_TIME          = 1.20;  // Thời gian tối thiểu ôm qua mũi đảo (s)
    double RB_ENTER_MAX_TIME          = 3.60;  // Thời gian tối đa ôm mũi đảo (s)
    double RB_CIRCLE_MIN_TIME         = 0.85;  // Thời gian tối thiểu đi dọc thân đảo qua nhánh phụ (s)
    double RB_CIRCLE_MAX_TIME         = 2.60;  // Thời gian tối đa đi dọc thân đảo trước khi ôm ra lối thoát (s)
    double RB_EXIT_MAX_TIME           = 3.00;  // Thời gian tối đa ôm theo lối thoát vòng xuyến (s)
    int    RB_BRANCH_OPEN_WIDTH_PX    = 380;
    int    RB_BRANCH_CLOSE_WIDTH_PX   = 340;

    int    RB_ENTRY_STABLE_FRAMES     = 2;     // Số frame xe phải đi thẳng ổn định trên đường dẫn trước khi bắt mũi đảo
    int    RB_ONROAD_MARGIN_PX        = 110;
    int    RB_OFFROAD_MARGIN_PX       = 20;
    int    rb_entry_stable_frames     = 0;
    bool   rb_approach_straight       = false;
    // ---------------- ĐIỀU KIỆN ĐÃ VÀO LÀN MỚI (đo ở tầng xa y1) ----------------
    int    ALIGN_EDGE_MARGIN        = 5;
    int    ALIGN_MIN_WIDTH_PX       = 60;
    double ALIGN_FAR_CENTER_PX      = 60.0;
    double ALIGN_FAR_OPPOSITE_PX    = 30.0;
    int    REQUIRED_ALIGNED_FRAMES  = 2;
    int    aligned_frames = 0;
    bool   far_swept = false;

    // ---------------- MIỆNG NGÃ RẼ & T-JUNCTION KHÔNG BIỂN ----------------
    int    JUNCTION_FAR_Y           = 150;
    int    TJUNCTION_OPEN_WIDTH_PX  = 410;     // Độ rộng tầng giữa khi gặp T-junction lớn
    double TJUNCTION_COOLDOWN_S     = 0.6;     // Nghỉ ngắn sau khi thoát hẳn vòng xuyến để bắt kịp T-junction kế tiếp

    double settle_start_bias  = 0.0;
    double settle_start_floor = 0.0;
    double settle_elapsed     = 0.0;
    double SETTLE_TIME        = 0.5;
    double turn_bias_current  = 0.0;

    double MAX_TURN_TIME            = 4.0;
    double CURVE_THRESHOLD          = 25.0;
    double STRAIGHT_THRESHOLD       = 12.0;
    int    REQUIRED_STRAIGHT_FRAMES = 2;       // Giảm từ 3 xuống 2 frame để quành T-junction sớm hơn, không sát mép trên
    int    straight_frames          = 0;

    // --- CÁC BIẾN TRẠNG THÁI ROUNDABOUT & T-JUNCTION ---
    bool            is_roundabout_mode      = false;
    RoundaboutPhase rb_phase                = RoundaboutPhase::NONE;
    int             roundabout_exits_passed = 0;
    bool            branch1_cleared         = false;
    int             rb_phase_Confirm_frames = 0;

    bool            waiting_for_tjunction   = false; // Bật sau khi hoàn thành mỗi roundabout
    double          tjunction_cooldown      = 0.0;
    int             t_junction_count        = 0;     // 1 -> rẽ TRÁI, 2 -> rẽ PHẢI
    int             tjunction_detect_frames = 0;

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

    void start_turn(const std::string& dir, bool is_roundabout = false) {
        if (state != State::IDLE && state != State::WAIT_CURVE && state != State::WAIT_STRAIGHT) return;
        // Nếu đang chờ ngã rẽ mà biển xác nhận lại đúng hướng cũ thì giữ nguyên tiến trình
        if ((state == State::WAIT_CURVE || state == State::WAIT_STRAIGHT) &&
            direction == dir && is_roundabout_mode == is_roundabout) return;

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
        roundabout_exits_passed = 0;
        branch1_cleared         = false;
        rb_phase_Confirm_frames = 0;
        rb_entry_stable_frames  = 0;
        rb_approach_straight    = false;

        if (is_roundabout_mode) {
            rb_phase = RoundaboutPhase::WAIT_ENTRY;
            std::cout << "[ROUNDABOUT] Mode Active -> Waiting for roundabout entry...\n";
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
        double steer_floor_deg  = 0.0;  // Góc sàn tối thiểu theo hướng quành (0 = không ép)
        double steer_cap_deg    = 0.0;  // Trần góc tối đa khi nhập xuyến (>0 = khóa trần góc phải)
        bool   lock_opposite_err = false;
        cv::Point p0 = cv::Point(-1, -1);
        cv::Point p1 = cv::Point(-1, -1);
        cv::Point p2 = cv::Point(-1, -1);
    };

    // Nhận thêm mid_lx, mid_rx (ở tầng y2 = 65% ảnh) và sai số near_e, mid_e, far_e
    TurnResult update(const std::vector<cv::Point>& left_pts, const std::vector<cv::Point>& right_pts,
                      double dt, bool lane_valid,
                      int far_lx = -1, int far_rx = -1,
                      int mid_lx = -1, int mid_rx = -1,
                      int img_w = 480,
                      bool has_active_sign = false,
                      double near_e = 0.0, double mid_e = 0.0, double far_e = 0.0) {
        TurnResult res{state, 0.0, direction};
        res.t_junction_count = t_junction_count;

        const int far_width = (far_lx >= 0 && far_rx >= 0) ? (far_rx - far_lx) : 0;
        const int mid_width = (mid_lx >= 0 && mid_rx >= 0) ? (mid_rx - mid_lx) : 0;

        // -----------------------------------------------------------------
        // TỰ ĐỘNG PHÁT HIỆN & ĐẾM T-JUNCTION KHÔNG BIỂN BÁO SAU ROUNDABOUT
        // -----------------------------------------------------------------
        if (state == State::IDLE) {
            if (tjunction_cooldown > 0.0) {
                tjunction_cooldown = std::max(0.0, tjunction_cooldown - dt);
            }
            if (waiting_for_tjunction && tjunction_cooldown <= 0.0 && !has_active_sign && t_junction_count < 2) {
                // T-junction 1 -> rẽ TRÁI ; T-junction 2 -> rẽ PHẢI
                const std::string target_dir = (t_junction_count == 0) ? "LEFT" : "RIGHT";
                const auto& t_pts = (target_dir == "LEFT") ? left_pts : right_pts;

                bool open_near = false, open_far = false;
                for (const auto& p : t_pts) {
                    bool touch = (target_dir == "LEFT") ? (p.x <= 5) : (p.x >= img_w - 10);
                    if (!touch) continue;
                    if (p.y >= 160 && p.y <= 210) open_near = true;
                    if (p.y <= JUNCTION_FAR_Y)    open_far  = true;
                }
                // Nhận biết sớm miệng T-junction khi tầng xa đã chạm mép và tầng giữa đã thu sát mép (<= 35px)
                if (target_dir == "LEFT" && far_lx >= 0 && far_lx <= 5 && mid_lx >= 0 && mid_lx <= 35) {
                    open_near = true;
                    open_far  = true;
                } else if (target_dir == "RIGHT" && far_rx >= img_w - 10 && mid_rx >= img_w - 36) {
                    open_near = true;
                    open_far  = true;
                }
                const bool wide_t_mouth = (mid_width >= TJUNCTION_OPEN_WIDTH_PX) &&
                                          ((target_dir == "LEFT") ? (far_lx >= 0 && far_lx <= 10)
                                                                  : (far_rx >= img_w - 15));

                if ((open_near && open_far) || wide_t_mouth) {
                    tjunction_detect_frames++;
                } else {
                    tjunction_detect_frames = 0;
                }

                if (tjunction_detect_frames >= 2) {
                    t_junction_count++;
                    waiting_for_tjunction   = false;
                    tjunction_detect_frames = 0;
                    std::cout << "[T-JUNCTION] >> UNMARKED T-JUNCTION #" << t_junction_count
                              << " DETECTED -> AUTO TURN " << target_dir << "\n";
                    start_turn(target_dir, false);
                    // Đã xác nhận 2 frame mở miệng T-junction -> nhảy thẳng vào WAIT_STRAIGHT với 1 frame tích lũy để quành sớm
                    state           = State::WAIT_STRAIGHT;
                    curve_detected  = true;
                    straight_frames = 1;
                    res.t_junction_count = t_junction_count;
                }
            }
            if (state == State::IDLE) return res;
        }

        // -----------------------------------------------------------------
        // XỬ LÝ CHUYÊN BIỆT CHO ROUNDABOUT (ĐỒNG BỘ THEO MŨI ĐẢO & LỐI THOÁT THỰC TẾ)
        // -----------------------------------------------------------------
        if (is_roundabout_mode) {
            turn_elapsed += dt;

            //PHA 1: Chờ trên đường dẫn cho tới khi xe thực sự gặp mũi đảo vòng xuyến
            // (Không bẻ sớm lúc vừa chốt biển từ xa trên đường dẫn!)
            if (rb_phase == RoundaboutPhase::WAIT_ENTRY) {
                const bool straight_now = lane_valid && (std::abs(near_e) <= 6.0) && (std::abs(mid_e) <= 15.0);
                if (straight_now) {
                    rb_entry_stable_frames++;
                    if (rb_entry_stable_frames >= RB_ENTRY_STABLE_FRAMES) {
                        rb_approach_straight = true;
                    }
                } else if (!rb_approach_straight) {
                    rb_entry_stable_frames = 0;
                }

                // Khi tiến sát mũi đảo vòng xuyến:
                // Đường uốn gắt sang phải quanh đảo làm far_lx vọt cao (>= 220px), biên phải mở sát rìa (far_rx, mid_rx >= img_w - 10)
                // và sai số tầng giữa mid_e tăng mạnh sang phải (>= 25px) hoặc mid_lx đã dạt sang phải (>= 60px).
                const bool island_nose_reached = rb_approach_straight &&
                                                 (far_rx >= img_w - 10) && (mid_rx >= img_w - 10) &&
                                                 (far_lx >= 220) &&
                                                 (mid_e >= 25.0 || mid_lx >= 60);

                if (island_nose_reached) {
                    rb_phase                = RoundaboutPhase::ENTER_RIGHT;
                    state                   = State::EXECUTE;
                    turn_elapsed            = 0.0;
                    branch1_cleared         = false;
                    rb_phase_Confirm_frames = 0;
                    std::cout << "[ROUNDABOUT] >> Phase 1 -> 2: ENTER_RIGHT (Island nose reached: far_lx="
                              << far_lx << ", mid_e=" << mid_e << ")\n";
                } else {
                    // Đang trên đường dẫn vào vòng xuyến: giữ nguyên bám làn tự nhiên, tuyệt đối không bẻ trái/phải sớm
                    res.state             = State::WAIT_CURVE;
                    res.steer_bias_px     = 0.0;
                    res.steer_floor_deg   = 0.0;
                    res.lock_opposite_err = false;
                    return res;
                }
            }

            // PHA 2: Ôm phải vòng qua mũi đảo để nhập vào hành lang vòng xuyến (log frame 613-632 & 1326-1343)
            if (rb_phase == RoundaboutPhase::ENTER_RIGHT) {
                res.direction         = "RIGHT";
                res.steer_bias_px     = ROUNDABOUT_ENTER_BIAS_PX;
                res.steer_floor_deg   = ROUNDABOUT_ENTER_DEG;
                res.steer_cap_deg     = 0.0; // Không khóa trần để PID thoải mái đánh 35-50 độ qua mũi đảo
                res.lock_opposite_err = true;

                // Ghi nhận mũi đảo đã đi qua tầng giữa (mid_lx dâng cao hoặc mid_e vọt lớn)
                if (mid_lx >= 70 || mid_e >= 35.0) {
                    branch1_cleared = true;
                }

                // Đã vòng qua mũi đảo khi tầng giữa bên trái mở rộng trở lại (mid_lx <= 50) và mid_e hạ xuống <= 15
                const bool nose_cleared = (turn_elapsed >= RB_ENTER_MIN_TIME) &&
                                          (branch1_cleared || turn_elapsed >= 1.8) &&
                                          (mid_lx >= 0 && mid_lx <= 50) &&
                                          (mid_e <= 15.0);

                if (nose_cleared || turn_elapsed >= RB_ENTER_MAX_TIME) {
                    rb_phase                = RoundaboutPhase::CIRCLE_LEFT;
                    turn_elapsed            = 0.0;
                    roundabout_exits_passed = 1;
                    rb_phase_Confirm_frames = 0;
                    std::cout << "[ROUNDABOUT] >> Phase 2 -> 3: CIRCLE_LEFT (Nose cleared, gliding past side opening)\n";
                }
            }
            // PHA 3: Đi dọc thân đảo (giữ nhẹ bên trái / thẳng để không rẽ nhầm vào nhánh phụ bên phải),
            // chờ tới khi khúc cua lối thoát vòng xuyến xuất hiện (log frame 633-651 & 1344-1360)
            else if (rb_phase == RoundaboutPhase::CIRCLE_LEFT) {
                res.direction         = "LEFT";
                res.steer_bias_px     = -ROUNDABOUT_CIRCLE_BIAS_PX;
                res.steer_floor_deg   = -ROUNDABOUT_CIRCLE_DEG;
                // Trong 1.0s đầu khi lướt ngang miệng nhánh phụ, khóa trần góc phải ở 10 độ để không bị hút vào nhánh phụ
                res.steer_cap_deg     = (turn_elapsed < 1.0) ? 10.0 : 0.0;
                res.lock_opposite_err = false;

                // Khúc cua lối thoát bắt đầu khi đã đi dọc đảo >= RB_CIRCLE_MIN_TIME
                // và mép trái tầng xa + tầng giữa lại uốn sang phải (far_lx >= 200, mid_lx >= 25 hoặc mid_e >= 20)
                const bool exit_curve_reached = (turn_elapsed >= RB_CIRCLE_MIN_TIME) &&
                                                (far_lx >= 200) &&
                                                (mid_lx >= 25 || mid_e >= 20.0);

                rb_phase_Confirm_frames = exit_curve_reached ? (rb_phase_Confirm_frames + 1) : 0;
                if (rb_phase_Confirm_frames >= 2 || turn_elapsed >= RB_CIRCLE_MAX_TIME) {
                    roundabout_exits_passed = 2;
                    rb_phase                = RoundaboutPhase::EXIT_RIGHT;
                    turn_elapsed            = 0.0;
                    aligned_frames          = 0;
                    far_swept               = false;
                    turn_bias_current       = ROUNDABOUT_EXIT_BIAS_PX;
                    turn_floor_current      = ROUNDABOUT_EXIT_DEG;
                    std::cout << "[ROUNDABOUT] >>> Phase 3 -> 4: EXIT_RIGHT (Following exit curve out of roundabout)\n";
                }
            }
            // PHA 4: Ôm phải theo đường cong lối thoát cho tới khi xe thẳng lại trên đường ra (log frame 652-680)
            else if (rb_phase == RoundaboutPhase::EXIT_RIGHT) {
                res.direction         = "RIGHT";
                res.steer_bias_px     = ROUNDABOUT_EXIT_BIAS_PX;
                res.steer_floor_deg   = ROUNDABOUT_EXIT_DEG;
                res.lock_opposite_err = true;

                const bool far_bounded = (far_lx > ALIGN_EDGE_MARGIN) && (far_rx >= 0) &&
                                         (far_rx < img_w - 1 - ALIGN_EDGE_MARGIN) &&
                                         (far_rx - far_lx >= ALIGN_MIN_WIDTH_PX);
                if (!far_bounded) far_swept = true;

                const double far_center_off = far_bounded ? (0.5 * (far_lx + far_rx) - 0.5 * img_w) : 9999.0;
                const bool far_centered = far_bounded &&
                                          (far_center_off <  ALIGN_FAR_CENTER_PX) &&
                                          (far_center_off > -ALIGN_FAR_OPPOSITE_PX);

                // Khi thoát ra đường thẳng dẫn tới T-junction: far_lx hạ xuống < 110 và mid_e, far_e đều giảm nhỏ
                const bool exit_road_straight = (turn_elapsed >= 1.0) && lane_valid &&
                                                (far_lx >= 0 && far_lx <= 110) &&
                                                (mid_e <= 28.0) && (far_e <= 20.0);

                if ((turn_elapsed >= 0.8 && lane_valid && far_swept && far_centered) || exit_road_straight) {
                    aligned_frames++;
                } else {
                    aligned_frames = 0;
                }

                if (aligned_frames >= REQUIRED_ALIGNED_FRAMES || turn_elapsed >= RB_EXIT_MAX_TIME) {
                    // Reset cờ và biến đếm Roundabout, chuyển sang SETTLE rồi bật dò T-junction
                    is_roundabout_mode      = false;
                    rb_phase                = RoundaboutPhase::NONE;
                    roundabout_exits_passed = 0;
                    branch1_cleared         = false;
                    rb_phase_Confirm_frames = 0;
                    rb_approach_straight    = false;

                    waiting_for_tjunction   = true;
                    tjunction_cooldown      = TJUNCTION_COOLDOWN_S;

                    state              = State::SETTLE;
                    direction          = "RIGHT";
                    settle_start_bias  = ROUNDABOUT_EXIT_BIAS_PX;
                    settle_start_floor = ROUNDABOUT_EXIT_DEG;
                    settle_elapsed     = 0.0;
                    std::cout << "[ROUNDABOUT] >>> EXIT COMPLETED -> RESET ROUNDABOUT FLAGS & ARM T-JUNCTION COUNTER!\n";
                }
            }

            res.roundabout_exit = roundabout_exits_passed;
            res.state           = state;
            return res;
        }

        // -----------------------------------------------------------------
        // XỬ LÝ RẼ THƯỜNG / T-JUNCTION (WAIT_CURVE -> WAIT_STRAIGHT -> EXECUTE -> SETTLE)
        // -----------------------------------------------------------------
        auto& pts = (direction == "LEFT") ? left_pts : right_pts;
        auto selected_pts = select_three_points(pts);
        double dev = calculate_deviation(pts);

        if (selected_pts.size() == 3) {
            res.p0 = selected_pts[0];
            res.p1 = selected_pts[1];
            res.p2 = selected_pts[2];
        }

        // Miệng ngã rẽ mở ở cả tầng gần (160-210) và tầng xa (<= JUNCTION_FAR_Y)
        bool side_open_near = false, side_open_far = false;
        for (const auto& p : pts) {
            bool touch = (direction == "LEFT") ? (p.x <= 5) : (p.x >= img_w - 10);
            if (!touch) continue;
            if (p.y >= 160 && p.y <= 210) side_open_near = true;
            if (p.y <= JUNCTION_FAR_Y)    side_open_far  = true;
        }
        // Bổ sung: nếu tầng xa đã chạm biên và tầng giữa (y2=187) đã thu sát biên (<= 35px) thì miệng ngã rẽ đã mở vào vùng quét
        if (direction == "LEFT" && far_lx >= 0 && far_lx <= 5 && mid_lx >= 0 && mid_lx <= 35) {
            side_open_near = true;
            side_open_far  = true;
        } else if (direction == "RIGHT" && far_rx >= img_w - 10 && mid_rx >= img_w - 36) {
            side_open_near = true;
            side_open_far  = true;
        }
        bool junction_opened = side_open_near && side_open_far;

        // Chặn lỗi dev = 0.0 ảo khi cả 3 điểm P0, P1, P2 đều nằm sát mép khung hình
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
            if (dev >= CURVE_THRESHOLD || junction_opened) {
                curve_detected = true;
                state = State::WAIT_STRAIGHT;
                std::cout << "[TURN] >> JUNCTION DETECTED (dev: " << dev << ", opened: " << junction_opened << ")\n";
            }
        }
        else if (state == State::WAIT_STRAIGHT) {
            if (dev >= 0.0 && dev <= STRAIGHT_THRESHOLD && junction_opened && (!pts_stuck_on_border || mid_width >= 360)) {
                straight_frames++;
            } else {
                straight_frames = 0;
            }

            if (straight_frames >= REQUIRED_STRAIGHT_FRAMES) {
                state          = State::EXECUTE;
                turn_elapsed   = 0.0;
                aligned_frames = 0;
                far_swept      = false;
                const double dir_sign = (direction == "LEFT") ? -1.0 : 1.0;
                turn_bias_current     = dir_sign * TURN_BIAS_PX;
                turn_floor_current    = dir_sign * std::min(TURN_STEER_DEG, TURN_RAMP_DEG_PER_S * dt);
                res.steer_bias_px     = turn_bias_current;
                res.steer_floor_deg   = turn_floor_current;
                res.lock_opposite_err = true;
                std::cout << "[TURN] >>> EXECUTING TURN (Verified straight 3 frames)\n";
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

            // Khi rẽ từ đường lớn sang đường lớn (như turn_right ở frame 1599-1648),
            // mask giữa ngã tư rộng toàn màn hình (mid_lx <= 25, mid_rx >= img_w - 15) suốt hơn 4s.
            // Không cho phép ngắt timeout 4.0s khi xe vẫn đang nằm giữa bụng ngã tư rộng mà chưa thấy mép làn mới!
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
            state              = State::IDLE;
            direction          = "NONE";
            turn_elapsed       = 0.0;
            aligned_frames     = 0;
            straight_frames    = 0;
            turn_bias_current  = 0.0;
            turn_floor_current = 0.0;
            curve_detected     = false;
            far_swept          = false;
            settle_start_bias  = 0.0;
            settle_start_floor = 0.0;
            settle_elapsed     = 0.0;
            std::cout << "[TURN] RESET TO IDLE\n";
        }

        res.state = state;
        return res;
    }
};

// ==========================================================
// CORNER ASSIST: ÔM CUA GẮT ~90° KHÔNG CÓ BIỂN BÁO
// Chỉ chạy khi LaneTurnController đang IDLE. Trả về góc sàn (âm = trái, dương = phải).
// ==========================================================
class CornerAssist {
public:
    enum class Phase { IDLE, ACTIVE, RELEASE };

    double CORNER_STEER_DEG      = 45.0;  // Góc sàn khi ôm cua 90° (PID vẫn được lên tới 50°)
    double CORNER_INIT_DEG       = 16.0;  // Góc sàn khởi điểm ngay frame chốt cua 90° để bẻ sớm, không bị trễ
    double CORNER_RAMP_DEG_PER_S = 85.0;  // ~14°/frame ở 6 FPS để lên góc 45° nhanh hơn
    double CORNER_RELEASE_TIME   = 0.35;  // Thời gian trả góc sàn về 0 (s)
    double CORNER_MIN_TIME       = 0.55;  // Tối thiểu trước khi xét đã vào đường mới (s)
    double CORNER_MAX_TIME       = 2.6;   // Lưới an toàn (s)
    double CORNER_COOLDOWN_S     = 1.0;   // Nghỉ sau khi nhả để không kích lại ngay
    int    CORNER_BEND_PX        = 145;   // Mép trong lệch giữa tầng giữa (y2) và tầng xa (y1) - hạ từ 180 xuống 145 để bắt cua sớm hơn
    int    CORNER_EARLY_FAR_PX   = 330;   // Khi góc cua 90° chưa chạm xuống y2 (lx2=0 hoặc rx2=479) nhưng ở y1 đã quét sâu >= 330px
    int    CORNER_CONFIRM_FRAMES = 2;
    int    EDGE_TOUCH_PX         = 3;
    int    ALIGN_CENTER_PX       = 70;
    int    ALIGN_MIN_WIDTH_PX    = 60;
    int    OVERTURN_PX           = 40;    // Mép phía trong cua ở đáy chạy vào gần tâm xe -> cắt cua lố

    void reset() {
        phase_ = Phase::IDLE; dir_ = 0; pending_dir_ = 0; confirm_ = 0;
        aligned_ = 0; elapsed_ = 0.0; floor_ = 0.0; release_start_ = 0.0; release_t_ = 0.0;
    }
    bool   active()    const { return phase_ != Phase::IDLE; }
    int    direction() const { return dir_; }
    double floor()     const { return floor_; }

    // lx1/rx1: tầng xa (45%), lx2/rx2: tầng giữa (65%), lx_bot/rx_bot: hàng đáy. Toạ độ ảnh mask gốc.
    double update(int lx1, int rx1, int lx2, int rx2, int lx_bot, int rx_bot, int w, double dt,
                  bool has_active_sign = false,
                  double near_e = 0.0, double mid_e = 0.0, double far_e = 0.0) {
        const int  edgeR    = w - 1 - EDGE_TOUCH_PX;
        const bool far_none = (lx1 < 0 || rx1 < 0);

        // Chặn nhiễu trên đỉnh cầu (auto_2.txt frame 325-353): xe đang ở giữa làn gần (|near_e| <= 3.5),
        // nhưng thành cầu cắt chéo làm |mid_e| >= 18 và |far_e| >= 95
        const bool on_bridge_crest = (std::abs(near_e) <= 3.5) && (std::abs(mid_e) >= 18.0) && (std::abs(far_e) >= 95.0);

        // 1. Trường hợp còn thấy tầng xa (!far_none):
        //    Cho phép bắt cua 90° SỚM ngay cả khi góc trong chưa chạm xuống y2 (lx2 == 0 hoặc rx2 == edgeR)
        //    miễn là biên trong ở tầng xa y1 đã quét rất sâu qua tâm (>= CORNER_EARLY_FAR_PX = 330px).
        // 2. Trường hợp mất tầng xa (far_none == true):
        //    Chặn hoàn toàn nhiễu lúc leo dốc cầu (frame 552-605: far_none=true nhưng lx2 chỉ 48..65, mid_e=-0.5..7.5, far_e=20..39.5)
        //    bằng cách đòi hỏi biên trong ở y2 phải khép sâu >= 140px VÀ |mid_e| >= 20, |far_e| >= 60.
        const bool right_far_ok = !far_none
            ? ((rx1 >= edgeR) && ((lx1 - lx2) >= CORNER_BEND_PX) &&
               (lx2 > EDGE_TOUCH_PX + 12 || lx1 >= CORNER_EARLY_FAR_PX) && (far_e >= 45.0))
            : (!has_active_sign && (lx2 >= 140) && (mid_e >= 20.0) && (far_e >= 60.0));

        const bool left_far_ok = !far_none
            ? ((lx1 >= 0 && lx1 <= EDGE_TOUCH_PX) && ((rx2 - rx1) >= CORNER_BEND_PX) &&
               (rx2 < edgeR - 12 || rx1 <= w - 1 - CORNER_EARLY_FAR_PX) && (far_e <= -45.0))
            : (!has_active_sign && (rx2 >= 0 && rx2 <= w - 1 - 140) && (mid_e <= -20.0) && (far_e <= -60.0));

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
                floor_ = dir_ * CORNER_INIT_DEG; // Bắt đầu ngay ở 16° thay vì 0° để xe quành sớm tức thì
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

            // Nhả ngay nếu mép phía cua ở tầng giữa (y2) đã tách khỏi rìa ảnh (hết cua hoặc kích nhầm)
            const bool edge_detached = (dir_ > 0) ? (rx2 >= 0 && rx2 < edgeR - 20)
                                                  : (lx2 >= 0 && lx2 > EDGE_TOUCH_PX + 20);

            if (aligned_ >= 2 || overturn || edge_detached || elapsed_ >= CORNER_MAX_TIME) {
                phase_ = Phase::RELEASE; release_start_ = floor_; release_t_ = 0.0;
                std::cout << "[CORNER] RELEASE ("
                          << (aligned_ >= 2 ? "aligned" : (overturn ? "overturn" : (edge_detached ? "edge_detached" : "timeout"))) << ")\n";
            }
        }
        else { // RELEASE
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
// LANE FOLLOWER
// ==========================================================
class LaneFollower {
public:
    struct Output {
        double angle = 0, speed = 0;
        bool valid = false, occl = false, forced_emer = false, emer_auth = false, losing = false;
        double near_e = 0, far_e = 0, eff_error = 0, coverage = 0; 
        double mid_e = 0;         // <--- Bổ sung biến này
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

// ==========================================================
// CẬP NHẬT TRONG LaneFollower::step (LaneFollowerV20.h)
// ==========================================================
// ============================================================================
// HÀM STEP HOÀN CHỈNH CHO CLASS LaneFollower (trong LaneFollowerV20.h)
// ============================================================================
Output step(const cv::Mat& roadMask, double totalBiasPx = 0.0, bool isExecutingTurn = false, double turnFloorDeg = 0.0, double turnCapDeg = 0.0) {    cv::Mat m = detail::preprocessMask(roadMask, c_);
    Output o;
    detail::Centerline cl;
    if (m.empty()) { 
        cl.valid = false; 
    } else {
        cl = detail::extractCenterline(m, curveSmooth_, c_);
    }

    const double nearE = cl.near_error;
    const double midE  = cl.mid_error;
    const double farE  = cl.far_error;
    curveSmooth_ = cl.curve_smooth;
    const bool valid = cl.valid;
    // ------------------------------------------------------------------------
    // 1. TÍNH TOÁN SAI SỐ LÀN: STRICT MID-GATED LOOKAHEAD
    // Triệt tiêu hiện tượng đón cua sớm, lọc nhiễu đỉnh cầu và hỗ trợ vào cua 90° sớm
    // ------------------------------------------------------------------------
    double lane_err = 0.0;
    
    // Nhiễu mặt đường dưới chân cầu khi xe đang ở đỉnh cầu (auto_2.txt frame 325-353):
    // Đầu xe hoàn toàn ở giữa làn (|nearE| <= 3.5), nhưng thành cầu cắt chéo làm |midE| >= 18 và |farE| >= 95
    const bool bridge_far_artefact = (std::abs(nearE) <= 3.5) && (std::abs(midE) >= 18.0) && (std::abs(farE) >= 95.0);

    // Chỉ coi là ngược chiều (cua chữ S) khi Near THỰC SỰ lệch rõ rệt (|nearE| > 3.0px)
    const bool is_genuinely_opposite = (nearE * farE < 0.0) && (std::abs(nearE) > 3.0);

    // Phát hiện vật cản đậu chắn một bên làn phía trước (auto_4.txt frame 18-62):
    // Mask xanh liếm xuống gầm xe vật cản làm |nearE| rất nhỏ (<= 3.5px), nhưng tầng giữa và tầng xa
    // đều bị thu hẹp cùng chiều rõ rệt (|midE| >= 14px, |farE| >= 28px, cùng dấu và không phải nhiễu đỉnh cầu).
    const bool obstacle_one_side = !isExecutingTurn && !bridge_far_artefact &&
                                   (std::abs(nearE) <= 3.5) &&
                                   (std::abs(midE) >= 14.0) && (std::abs(farE) >= 28.0) &&
                                   (midE * farE > 0.0);

    if (bridge_far_artefact && !isExecutingTurn) {
        // Trên đỉnh cầu: bỏ qua Far bị cắt bởi thành cầu, giữ xe đi thẳng theo tâm gần
        lane_err = (0.85 * nearE) + (0.15 * std::clamp(midE, -12.0, 12.0));
    } else if (obstacle_one_side) {
        // Né xe vật cản: giảm trọng số nearE (vì bị mask liếm gầm đánh lừa = -0.5),
        // dồn trọng số vào midE và farE để xe đánh lái né sớm và mạnh hơn mà không cần vẽ rào chắn ảo
        double max_far_obs = std::max(std::abs(midE) * 2.0, 45.0);
        double safe_far = std::clamp(farE, -max_far_obs, max_far_obs);
        lane_err = (0.10 * nearE) + (0.50 * midE) + (0.40 * safe_far);
    } else if (is_genuinely_opposite) {
        // Tình huống chữ S thực sự: Near lệch một bên, Far uốn bên ngược lại
        if (midE * farE > 0.0 && std::abs(midE) > 6.0) {
            double capped_far = std::clamp(farE, -25.0, 25.0);
            lane_err = (0.45 * nearE) + (0.35 * midE) + (0.20 * capped_far);
        } else {
            // Khúc cua chưa chạm tới Mid, giữ tâm làn gần
            lane_err = (0.65 * nearE) + (0.35 * midE);
        }
    } else {
        // Bình thường: Far chỉ được kéo tối đa 1.5 lần Mid (tối thiểu 30px).
        // Riêng khi sắp vào cua 90° (Mid còn nhỏ < 15px nhưng Far đã vọt >= 50px và không phải nhiễu đỉnh cầu),
        // mở trần Far lên 60px để PID bắt đầu nghiêng lái đón cua 90° sớm hơn 3-5 frame trước khi CornerAssist khóa góc sàn.
        double base_far_cap = (!isExecutingTurn && std::abs(midE) < 15.0 && std::abs(farE) >= 50.0) ? 60.0 : 30.0;
        double max_far_allowed = std::max(std::abs(midE) * 1.5, base_far_cap);
        double safe_far = std::clamp(farE, -max_far_allowed, max_far_allowed);
        lane_err = (0.40 * nearE) + (0.35 * midE) + (0.25 * safe_far);
    }

    // ------------------------------------------------------------------------
    // 2. KHUẾCH ĐẠI PHI TUYẾN (NONLINEAR GAIN) - giảm nhẹ 3-4 độ ở cua thoai thoải
    // ------------------------------------------------------------------------
    const double abs_err = std::abs(lane_err);
    if (abs_err > 14.0) {
        double error_scale = std::min(1.0 + (abs_err - 14.0) * 0.024, 1.42);
        lane_err *= error_scale;
    }

    // ------------------------------------------------------------------------
    // 3. KHÓA LỰC KÉO CỦA LÀN ĐI THẲNG CŨ TRONG TRẠNG THÁI RẼ (EXECUTE TURN)
    // ------------------------------------------------------------------------
    if (isExecutingTurn) {
        if (totalBiasPx < -10.0) {
            // Đang rẽ TRÁI: Không cho phép lực kéo dương (sang phải) của làn thẳng cũ can thiệp
            lane_err = std::min(lane_err, 0.0);
        } else if (totalBiasPx > 10.0) {
            // Đang rẽ PHẢI: Không cho phép lực kéo âm (sang trái) của làn thẳng cũ can thiệp
            lane_err = std::max(lane_err, 0.0);
        }
    }

    double eff = lane_err + totalBiasPx;

    // ------------------------------------------------------------------------
    // 4. KIỂM SOÁT VẬT CẢN (OCCLUSION GATE) & BẤT ĐỐI XỨNG LÀN (ASYMMETRY)
    // ------------------------------------------------------------------------
    bool occl = gate_.step(eff, cl.coverage, cl.row_centers, valid, farE, nearE);
    if (occl) emerHold_ = c_.emg_hold_after_occl;
    else if (emerHold_ > 0) emerHold_--;

    if (!occl && valid && !m.empty()) {
        double asymRaw = detail::detectAsymmetry(m, 45.0, 0.22, 0.72);
        if (!bridge_far_artefact && std::abs(asymRaw) >= c_.asym_threshold) {
            asymSmooth_ = 0.65 * asymSmooth_ + 0.35 * asymRaw;
        } else {
            // Khi đang đi ngang qua thân vật cản, duy trì lực né lâu hơn thay vì xả ngay khi nearE nhỏ
            asymSmooth_ *= c_.asym_decay;
        }
        asymSmooth_ = std::clamp(asymSmooth_, -c_.asym_cap, c_.asym_cap);
    }
    
    double asymApplied = (std::abs(asymSmooth_) < c_.asym_threshold) ? 0.0 : c_.asym_strength * asymSmooth_;
    eff += asymApplied;

    bool forcedEmer = (emerHold_ > 0);
    pid_.setForcedEmergency(forcedEmer);

    // ------------------------------------------------------------------------
    // 5. ĐIỀU KHIỂN PID VÀ TỰ ĐỘNG GIẢM TỐC THEO ĐỘ GẮT CỦA CUA
    // ------------------------------------------------------------------------
    double angle;
    if (valid) {
        invalid_frames_ = 0;
        if (!prevValid_) { pid_.softReset(); gate_.reset(); }
        
        angle = pid_.step(eff, c_.kp, c_.rate_limit);
        
        double target = detail::planSpeed(curveSmooth_, c_.max_speed, c_.min_speed, c_.curve_scale);
        
        // Hãm tốc độ chủ động về mức tối thiểu khi xe đang chịu sai số gần lớn
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
    // ------------------------------------------------------------------------
    // 6. ƯU TIÊN QUÀNH: GÓC SÀN & TRẦN GÓC NHẬP XUYẾN
    // ------------------------------------------------------------------------
    if (turnFloorDeg < -0.5 && angle > turnFloorDeg) {
        angle = turnFloorDeg;
        pid_.overrideOutput(angle);
    } else if (turnFloorDeg > 0.5 && angle < turnFloorDeg) {
        angle = turnFloorDeg;
        pid_.overrideOutput(angle);
    }
    // Khóa trần góc phải khi mới nhập vòng xuyến để không bị đảo tròn đẩy vọt vào Nhánh 1
    if (turnCapDeg > 0.5 && angle > turnCapDeg) {
        angle = turnCapDeg;
        pid_.overrideOutput(angle);
    }
    if (valid || std::abs(turnFloorDeg) > 0.5) lastAngle_ = angle;
    prevValid_ = valid;

    // Đóng gói Output telemetry
    o.angle = angle;
    o.speed = speed_;
    o.valid = valid;
    o.occl = occl;
    o.mid_e = midE;                  // <--- Gán giá trị
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

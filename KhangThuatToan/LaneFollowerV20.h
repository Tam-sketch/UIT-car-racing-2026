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
#include <utility>
#include <vector>
#include <iostream>

namespace v20 {

struct Config {
    int  mask_width   = 320;
    bool keep_largest = true;

    double max_speed = 18.0;
    double min_speed = 11.0;

    double asym_strength = 0.15, asym_threshold = 8.0, asym_ema = 0.7;
    double asym_cap = 10.0, asym_decay = 0.80;

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
        double corner_error = 0, near_error = 0, mid_error = 0, far_error = 0;
        double curve_error = 0, curve_smooth = 0, coverage = 0;
        bool valid = false;
        std::vector<std::pair<int, int>> row_centers;
        std::vector<std::pair<int, int>> left_pts;
        std::vector<std::pair<int, int>> right_pts;
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

        // ... [Phần tính toán phân vị yBot, yTop, nearC, midC, farC giữ nguyên] ...
        std::vector<int> ys; ys.reserve(r.row_centers.size());
        for (auto& p : r.row_centers) ys.push_back(p.second);
        std::sort(ys.begin(), ys.end());
        int yBot = percentileInt(ys, 0.98), yTop = percentileInt(ys, 0.02);
        int span = std::max(yBot - yTop, 1);

        int ySplitCorner = yBot - std::max(4, (int)(span * 0.18));
        int ySplitNear   = yBot - (int)(span * 0.35);
        int ySplitFar    = yTop + (int)(span * 0.30);

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
// LANE TURN CONTROLLER
// ==========================================================
class LaneTurnController {
public:
    enum class State { IDLE, WAIT_CURVE, WAIT_STRAIGHT, COUNTDOWN, EXECUTE, SETTLE, DONE };

    // Điều kiện xác nhận xe đã vào làn mới
    double ALIGN_NEAR_THRESHOLD = 12.0;
    double ALIGN_FAR_THRESHOLD  = 10.0;
    double ALIGN_HEADING_THRESHOLD = 6.0;

    double settle_start_bias = 0.0;
    double settle_elapsed = 0.0;
    
    int REQUIRED_ALIGNED_FRAMES = 6;
    int aligned_frames = 0;

    // Giảm bias từ từ trong SETTLE
    double SETTLE_TIME = 0.5;       // giây
    double turn_bias_current = 0.0;

    // Giới hạn thời gian để tránh xe ép cua vô hạn
    double MAX_TURN_TIME = 4.0;
    double CURVE_THRESHOLD = 20.0;
    double STRAIGHT_THRESHOLD = 12.0;
    int REQUIRED_STRAIGHT_FRAMES = 3;  // BẮT BUỘC thẳng liên tục trong 3 frame
    int straight_frames = 0;    
    double TURN_BIAS_PX = 35.0;

    State state = State::IDLE;
    std::string direction = "NONE";
    double countdown = 0.0;
    double turn_elapsed = 0.0;
    bool curve_detected = false;
    bool completed = false;

    // THÊM 2 DÒNG NÀY VÀO ĐÂY:
    double get_countdown() const { return countdown; }
    double get_turn_elapsed() const { return turn_elapsed; }

    std::vector<cv::Point> select_three_points(std::vector<cv::Point> points) {
        std::sort(points.begin(), points.end(), [](const cv::Point& a, const cv::Point& b){ return a.y < b.y; });
        int n = points.size();
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
        double dev = std::abs(dx * (p[1].y - p[0].y) - dy * (p[1].x - p[0].x)) / length;
        return dev;
    }

    void start_turn(const std::string& dir) {
        if (state != State::IDLE) return;

        direction = dir;
        state = State::WAIT_CURVE;

        countdown = 0.0;
        turn_elapsed = 0.0;

        aligned_frames = 0;
        straight_frames = 0; // Reset bộ đếm frame thẳng
        turn_bias_current = 0.0;

        curve_detected = false;
        completed = false;

        std::cout << "[TURN] Requested: " << dir << " -> Waiting for scanline curve...\n";
    }



    struct TurnResult {
        State state;
        double steer_bias_px;
        std::string direction;
        double dev = -1.0;           // <--- THÊM DÒNG NÀY
        bool mouth_reached = false;  // <--- THÊM DÒNG NÀY
        bool junction_opened = false;
        int  straight_frames = 0;
        // BỔ SUNG 3 ĐIỂM QUAN SÁT:
        cv::Point p0 = cv::Point(-1, -1);
        cv::Point p1 = cv::Point(-1, -1);
        cv::Point p2 = cv::Point(-1, -1);
    };

    TurnResult update(const std::vector<cv::Point>& left_pts, const std::vector<cv::Point>& right_pts, double dt, bool lane_valid, bool lane_occluded, double near_error, double far_error) {
        TurnResult res{state, 0.0, direction};
        if (state == State::IDLE) return res;

        auto& pts = (direction == "LEFT") ? left_pts : right_pts;
        auto selected_pts = select_three_points(pts);
        double dev = calculate_deviation(pts);

        if (selected_pts.size() == 3) {
            res.p0 = selected_pts[0];
            res.p1 = selected_pts[1];
            res.p2 = selected_pts[2];
        }

        // Vẫn ghi nhận mouth_reached cho telemetry nhưng KHÔNG dùng để bypass rẽ
        bool mouth_reached = false;
        if (!pts.empty()) {
            int touch_count = 0;
            for (const auto& p : pts) {
                if (p.y <= 220) { // Bỏ qua chân cản trước ở đáy ảnh
                    if (direction == "LEFT" && p.x <= 15) touch_count++;
                    if (direction == "RIGHT" && p.x >= 465) touch_count++;
                }
            }
            if (touch_count >= 1) mouth_reached = true;
        }


        bool junction_opened = false;
        if (direction == "LEFT") {
            // Nếu rẽ trái: Tại tầng quan sát (y = 187), vạch trái phải biến mất hoặc mở rộng hẳn về biên trái (lx <= 5)
            // Hoặc độ rộng lòng đường mở rộng bất thường
            for (const auto& p : left_pts) {
                if (p.y >= 160 && p.y <= 210 && p.x <= 5) {
                    junction_opened = true;
                    break;
                }
            }
        } else if (direction == "RIGHT") {
            // Nếu rẽ phải: Tại tầng y = 187, vạch phải biến mất hoặc mở rộng sát mép phải (rx >= 470)
            for (const auto& p : right_pts) {
                if (p.y >= 160 && p.y <= 210 && p.x >= 470) {
                    junction_opened = true;
                    break;
                }
            }
        }

        res.dev = dev;
        res.mouth_reached = mouth_reached;
        res.junction_opened = junction_opened;
        res.straight_frames = straight_frames;
        // BƯỚC 1: Chờ ngã rẽ thực sự xuất hiện trước mặt
        if (state == State::WAIT_CURVE) {
            straight_frames = 0;
            // Phải thỏa mãn: HOẶC scanline uốn cong thực sự, HOẶC miệng ngã rẽ bên hông đã mở ra
            if (dev >= CURVE_THRESHOLD || junction_opened) {
                curve_detected = true;
                state = State::WAIT_STRAIGHT;
                std::cout << "[TURN] >> JUNCTION DETECTED (dev: " << dev << ", opened: " << junction_opened << ")\n";
            }
        }
        // BƯỚC 2: Khi đã xác nhận đến ngã rẽ, chờ scanline ổn định trong 3 frame VÀ ngã rẽ vẫn đang mở
        else if (state == State::WAIT_STRAIGHT) {
            if (dev >= 0.0 && dev <= STRAIGHT_THRESHOLD && junction_opened) {
                straight_frames++;
            } else {
                straight_frames = 0;
            }

            if (straight_frames >= REQUIRED_STRAIGHT_FRAMES) {
                state = State::EXECUTE;
                turn_elapsed = 0.0;
                aligned_frames = 0;
                turn_bias_current = (direction == "LEFT") ? -TURN_BIAS_PX : TURN_BIAS_PX;
                res.steer_bias_px = turn_bias_current;
                std::cout << "[TURN] >>> EXECUTING TURN (Mouth open & verified straight 3 frames)\n";
            }
        }
        // BƯỚC 3: THỰC THI BẺ LÁI
        else if (state == State::EXECUTE) {
            turn_elapsed += dt;

            const double heading_error = std::abs(far_error - near_error);

            // Chỉ xét điều kiện thẳng làn mới sau khi đã bẻ lái tối thiểu 0.6 giây
            const bool lane_aligned =
                (turn_elapsed > 0.6) &&
                lane_valid &&
                !lane_occluded &&
                std::abs(near_error) < ALIGN_NEAR_THRESHOLD &&
                std::abs(far_error) < ALIGN_FAR_THRESHOLD &&
                heading_error < ALIGN_HEADING_THRESHOLD;

            if (lane_aligned) aligned_frames++;
            else aligned_frames = 0;

            res.steer_bias_px = turn_bias_current;

            if (aligned_frames >= REQUIRED_ALIGNED_FRAMES) {
                state = State::SETTLE;
                settle_start_bias = turn_bias_current;
                settle_elapsed = 0.0;
                std::cout << "[TURN] NEW LANE ALIGNED - SETTLE\n";
            }

            if (turn_elapsed >= MAX_TURN_TIME && state == State::EXECUTE) {
                state = State::SETTLE;
                settle_start_bias = turn_bias_current;
                settle_elapsed = 0.0;
                std::cout << "[TURN] TIMEOUT - RELEASE TURN BIAS\n";
            }
        }
        else if (state == State::SETTLE) {
            settle_elapsed += dt;
            double ratio = std::clamp(1.0 - settle_elapsed / std::max(SETTLE_TIME, 0.01), 0.0, 1.0);
            turn_bias_current = settle_start_bias * ratio;
            res.steer_bias_px = turn_bias_current;

            if (settle_elapsed >= SETTLE_TIME) {
                turn_bias_current = 0.0;
                res.steer_bias_px = 0.0;
                state = State::DONE;
                completed = true;
                std::cout << "[TURN] TURN BIAS RELEASED\n";
            }
        }
        else if (state == State::DONE) {
            state = State::IDLE;
            direction = "NONE";
            countdown = 0.0;
            turn_elapsed = 0.0;
            aligned_frames = 0;
            straight_frames = 0;
            turn_bias_current = 0.0;
            curve_detected = false;
            settle_start_bias = 0.0;
            settle_elapsed = 0.0;
            std::cout << "[TURN] RESET TO IDLE\n";
        }

        res.state = state;
        return res;
    }
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
        int dodge_side = 0;       // <--- Bổ sung biến này (lưu hướng né: -1, 0, 1)
        int dodge_frames = 0;     // <--- Bổ sung biến này (lưu số frame giữ khoảng hở)
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
        dodge_side_locked_ = 0;
        dodge_hold_frames_ = 0;
    }
    double speed() const { return speed_; }

// ==========================================================
// CẬP NHẬT TRONG LaneFollower::step (LaneFollowerV20.h)
// ==========================================================
// ============================================================================
// HÀM STEP HOÀN CHỈNH CHO CLASS LaneFollower (trong LaneFollowerV20.h)
// ============================================================================
Output step(const cv::Mat& roadMask, double totalBiasPx = 0.0, bool isExecutingTurn = false) {
    cv::Mat mask_clean = roadMask.clone();
    // KHAI BÁO CÁC BIẾN KÍCH THƯỚC TRƯỚC KHI TÍNH BARRIER
    const int h = mask_clean.rows;
    const int w = mask_clean.cols;
    const int start_y = static_cast<int>(h * 0.40);
    const int cut_h   = h - start_y;

    // BỨC TƯỜNG ẢO: XÓA SẠCH MASK DƯỚI GẦM XE VẬT CẢN KHI ĐANG NÉ
    if (dodge_side_locked_ == -1) {
        // Vật cản bên TRÁI: Xóa 30% mép trái (x: 0 -> 0.30*w)
        int cut_w = static_cast<int>(w * 0.30);
        cv::Rect left_barrier(0, start_y, cut_w, cut_h);
        mask_clean(left_barrier) = 0;
    } 
    else if (dodge_side_locked_ == 1) {
        // Vật cản bên PHẢI: Xóa 30% mép phải (x: 0.70*w -> w)
        int start_x = static_cast<int>(w * 0.70);
        int cut_w   = w - start_x;
        cv::Rect right_barrier(start_x, start_y, cut_w, cut_h);
        mask_clean(right_barrier) = 0;
    }
    
    cv::Mat m = detail::preprocessMask(mask_clean, c_);
    Output o;
    detail::Centerline cl;
    if (m.empty()) { 
        cl.valid = false; 
    } else {
        int search_side = (dodge_side_locked_ == 1) ? -1 : ((dodge_side_locked_ == -1) ? 1 : 0);
        cl = detail::extractCenterline(m, curveSmooth_, c_, search_side);    
    }

    const double nearE = cl.near_error;
    const double midE  = cl.mid_error;
    const double farE  = cl.far_error;
    curveSmooth_ = cl.curve_smooth;
    const bool valid = cl.valid;
    // -------------------------------------------------------------
    // PHÁT HIỆN VÀ KHÓA HƯỚNG NÉ VẬT CẢN (OBSTACLE CLEARANCE LOGIC)
    // -------------------------------------------------------------
    // Khi một bên làn bị co hẹp bất thường (Near hoặc Mid lệch > 18px)
    if (std::abs(nearE) > 18.0 || std::abs(midE) > 22.0) {
        dodge_side_locked_ = (nearE < 0 || midE < 0) ? -1 : 1; // Khóa hẳn hướng né Trái (-1) hoặc Phải (+1)
        dodge_hold_frames_ = 16; // Giữ trạng thái né thêm 8 frames (~1.3 giây) chống quẹt đuôi
    } else if (dodge_hold_frames_ > 0) {
        dodge_hold_frames_--;
        if (dodge_hold_frames_ == 0) {
            dodge_side_locked_ = 0; // Đã qua hẳn vật cản, mở khóa
        }
    }

    // ------------------------------------------------------------------------
    // 1. TÍNH TOÁN SAI SỐ LÀN: STRICT MID-GATED LOOKAHEAD
    // Triệt tiêu hiện tượng đón cua sớm và tự triệt tiêu sai số khi cua chữ S
    // ------------------------------------------------------------------------
    double lane_err = 0.0;
    
    // Chỉ coi là ngược chiều (cua chữ S) khi Near THỰC SỰ lệch rõ rệt (|nearE| > 3.0px)
    // Nếu |nearE| <= 3.0px, xe chỉ đang chạy thẳng ở giữa làn và nhìn thấy cua ở xa
    const bool is_genuinely_opposite = (nearE * farE < 0.0) && (std::abs(nearE) > 3.0);

    if (is_genuinely_opposite) {
        // Tình huống chữ S thực sự: Near lệch một bên, Far uốn bên ngược lại
        if (midE * farE > 0.0 && std::abs(midE) > 6.0) {
            double capped_far = std::clamp(farE, -25.0, 25.0);
            lane_err = (0.45 * nearE) + (0.35 * midE) + (0.20 * capped_far);
        } else {
            // Khúc cua chưa chạm tới Mid, giữ tâm làn gần
            lane_err = (0.65 * nearE) + (0.35 * midE);
        }
    } else {
        // Tình huống bình thường hoặc xe đang đi thẳng chuẩn bị vào cua:
        // Cấp quyền cho Far tham gia kéo lái sớm từ 20% đến 25%
        double max_far_allowed = std::max(std::abs(midE) * 2.0, 35.0); // Nâng trần cho Far từ 10 lên 25px
        double safe_far = std::clamp(farE, -max_far_allowed, max_far_allowed);
        lane_err = (0.35 * nearE) + (0.35 * midE) + (0.30 * safe_far);
    }

    // ------------------------------------------------------------------------
    // 2. KHUẾCH ĐẠI PHI TUYẾN (NONLINEAR GAIN)
    // Giữ xe êm khi ở giữa làn (|err| < 15px), bẻ gắt dứt khoát khi chớm lệch lớn
    // ------------------------------------------------------------------------
    const double abs_err = std::abs(lane_err);
    if (abs_err > 12.0) {
        // Tăng tuyến tính lực kéo từ 1.0 lên tối đa 1.8 lần khi sai số tăng từ 15px đến 35px
        double error_scale = std::min(1.0 + (abs_err - 10.0) * 0.04, 1.8);
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
        double asymRaw = detail::detectAsymmetry(m);
        asymSmooth_ = c_.asym_ema * asymSmooth_ + (1.0 - c_.asym_ema) * asymRaw;
        if (std::abs(nearE) < 5.0 && std::abs(asymSmooth_) < c_.asym_cap) asymSmooth_ *= c_.asym_decay;
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
        lastAngle_ = angle;
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
    prevValid_ = valid;

    // Đóng gói Output telemetry
    o.angle = angle;
    o.speed = speed_;
    o.valid = valid;
    o.occl = occl;
    o.mid_e = midE;                  // <--- Gán giá trị
    o.dodge_side = dodge_side_locked_;   // <--- Gán giá trị (hoặc biến tương ứng quản lý hướng né trong class)
    o.dodge_frames = dodge_hold_frames_; // <--- Gán giá trị (hoặc biến đếm frame giữ chỗ)
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
    // BỔ SUNG 2 BIẾN NÀY ĐỂ TRÁNH LỖI BIÊN DỊCH:
    int dodge_side_locked_ = 0;
    int dodge_hold_frames_ = 0;
};

}  // namespace v20

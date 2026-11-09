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
#include <iostream>

namespace v22 {

// ============================================================
// CONFIG (Kế thừa V20)
// ============================================================
struct Config {
    int  mask_width   = 320;    
    bool keep_largest = true;   
    double post_pid_gain = 1.0; // Set = 1.0 nếu đã nhân gain ở ngoài
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
    inline int percentileInt(const std::vector<int>& sorted, double q) {
        double pos = q * (double)(sorted.size() - 1);
        size_t lo = (size_t)pos;
        double fr = pos - (double)lo;
        double v = (double)sorted[lo] + (lo + 1 < sorted.size() ? fr * (double)(sorted[lo + 1] - sorted[lo]) : 0.0);
        return (int)v;
    }
    inline double sgn(double x) { return (x > 0) - (x < 0); }

    inline cv::Mat preprocessMask(const cv::Mat& in, const Config& c, double& kept_frac_out) {
        kept_frac_out = 1.0;
        if (in.empty() || in.type() != CV_8UC1) return cv::Mat();
        
        cv::Mat out;
        if (c.mask_width > 0 && in.cols != c.mask_width) {
            cv::resize(in, out, cv::Size(c.mask_width, in.rows * c.mask_width / in.cols), 0, 0, cv::INTER_NEAREST);
        } else out = in.clone();

        cv::Mat binary = (out > 0);
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 5));
        cv::morphologyEx(binary, out, cv::MORPH_CLOSE, kernel);

        if (!c.keep_largest) return out;

        std::vector<int> labels((size_t)out.rows * out.cols, 0);
        std::vector<int> stack, areas(1, 0);
        int nLabels = 0;
        for (int y = 0; y < out.rows; y++) {
            const uchar* row = out.ptr<uchar>(y);
            for (int x = 0; x < out.cols; x++) {
                if (!row[x] || labels[(size_t)y * out.cols + x]) continue;
                ++nLabels;
                int area = 0;
                stack.clear(); stack.push_back(y * out.cols + x);
                labels[(size_t)y * out.cols + x] = nLabels;
                while (!stack.empty()) {
                    int p = stack.back(); stack.pop_back();
                    area++;
                    int py = p / out.cols, px = p % out.cols;
                    for (int dy = -1; dy <= 1; dy++) {
                        int ny = py + dy;
                        if (ny < 0 || ny >= out.rows) continue;
                        const uchar* nrow = out.ptr<uchar>(ny);
                        for (int dx = -1; dx <= 1; dx++) {
                            if (!dx && !dy) continue;
                            int nx = px + dx;
                            if (nx < 0 || nx >= out.cols) continue;
                            size_t q = (size_t)ny * out.cols + nx;
                            if (nrow[nx] && !labels[q]) { labels[q] = nLabels; stack.push_back((int)q); }
                        }
                    }
                }
                areas.push_back(area);
            }
        }
        if (nLabels <= 1) return out;
        int best = 1; int total_area = 0;
        for (int i = 1; i <= nLabels; i++) total_area += areas[i];
        for (int i = 2; i <= nLabels; i++) if (areas[i] > areas[best]) best = i;
        kept_frac_out = (total_area > 0) ? (double)areas[best] / (double)total_area : 1.0;

        for (int y = 0; y < out.rows; y++) {
            uchar* row = out.ptr<uchar>(y);
            for (int x = 0; x < out.cols; x++) if (labels[(size_t)y * out.cols + x] != best) row[x] = 0;
        }
        return out;
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

// ============================================================
// V20 CORE: OcclusionGate, Authority, PID
// ============================================================
class OcclusionGate {
public:
    explicit OcclusionGate(const Config& c = Config()) : c_(c) { reset(); }
    void reset() { covEma_ = 0; prevErr_ = 0; init_ = false; hangover_ = 0; farHist_.clear(); farSpread_ = 0; lastHard_ = false; lastFarOutlier_ = false; }
    bool step(double error, double coverage, const std::vector<std::pair<int, int>>& rows, bool valid, double farE, double nearE) {
        farHist_.push_back(farE);
        if ((int)farHist_.size() > c_.far_window) farHist_.pop_front();
        if (farHist_.size() >= 2) { auto mm = std::minmax_element(farHist_.begin(), farHist_.end()); farSpread_ = *mm.second - *mm.first; } else farSpread_ = 0;
        if (!init_) { covEma_ = coverage; prevErr_ = error; init_ = true; hangover_ = 0; lastHard_ = false; lastFarOutlier_ = false; return false; }
        covEma_ = c_.cov_ema_alpha * covEma_ + (1.0 - c_.cov_ema_alpha) * coverage;
        if (!valid) { prevErr_ = error; hangover_ = c_.occl_hangover; lastHard_ = false; return true; }
        double errStep = std::abs(error - prevErr_);
        double covDrop = (covEma_ > 0.05) ? std::max(0.0, (covEma_ - coverage) / covEma_) : 0.0;
        double jagged = 0;
        if (rows.size() >= 6) {
            double sum = 0; int n = 0;
            for (size_t i = 1; i + 1 < rows.size(); i++) { sum += std::abs(rows[i + 1].first - 2 * rows[i].first + rows[i - 1].first); n++; }
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
    Config c_; double covEma_, prevErr_, farSpread_; bool init_, lastHard_, lastFarOutlier_; int hangover_; std::deque<double> farHist_;
};

class AuthorityManager {
public:
    explicit AuthorityManager(const Config& c = Config()) : c_(c) { reset(); }
    void reset() { satStreak_ = 0; losing_ = false; }
    void step(double error, double prevError, double pidOut) {
        double dErr = error - prevError; double pred = error + c_.auth_tau * dErr;
        if (std::abs(pidOut) > c_.auth_sat_frac * c_.output_limit) satStreak_++; else satStreak_ = 0;
        if (!losing_) {
            if (satStreak_ >= c_.auth_sat_frames && std::abs(pred) > std::abs(error) && std::abs(error) > c_.auth_lose_enter_px) losing_ = true;
        } else {
            if (std::abs(error) < c_.auth_lose_exit_px || satStreak_ < c_.auth_sat_frames) losing_ = false;
        }
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
    bool losing() const { return auth_.losing(); }
    bool emergencyAuthority() const { return emgAuth_; }
    double step(double error) {
        double kpEff = c_.kp * (auth_.losing() ? 1.5 : 1.0); double kdEff = c_.kd * (auth_.losing() ? 0.7 : 1.0);
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
    Config c_; AuthorityManager auth_; double integral_, prevErr_, filtD_, prevOut_; bool first_, emgAuth_, forced_; int emgStreak_;
};

// ============================================================
// TURN PLANNER (Ghi đè Centerline)
// ============================================================
struct TurnMemory {
    bool active = false;
    std::string side = "";
    int turn_width = 60;
    int last_y_mid = 9999; // Điểm bắt đầu bẻ lái
};

class TurnPlanner {
public:
    TurnMemory mem;
   
    void reset() { mem = TurnMemory(); }

    // Gọi từ bên ngoài khi YOLO thấy biển báo
    void setSign(const std::string& sign_name, int trigger_y = 150) {
        if (sign_name == "turn_left" || sign_name == "no_right" || sign_name == "construction_ahead") {
            mem.active = true;
            mem.side = "left";
            mem.turn_width = 60; 
            mem.last_y_mid = trigger_y; // <-- Gán điểm bắt đầu bẻ lái
        } else if (sign_name == "turn_right" || sign_name == "no_left") {
            mem.active = true;
            mem.side = "right";
            mem.turn_width = 60;
            mem.last_y_mid = trigger_y; // <-- Gán điểm bắt đầu bẻ lái
        }
    }

    void modifyCenterline(std::vector<std::pair<int, int>>& row_centers, int w) {
        if (row_centers.empty() || !mem.active) return;

        int idx_turn = -1;
        int x_c_turn = w / 2;
        
        // Tìm điểm bắt đầu bẻ lái (điểm có y nhỏ hơn last_y_mid)
        for (int i = 0; i < (int)row_centers.size(); i++) {
            if (row_centers[i].second <= mem.last_y_mid) {
                idx_turn = i;
                x_c_turn = row_centers[i].first;
                break;
            }
        }

        if (idx_turn >= 0) {
            int sign = (mem.side == "left") ? -1 : 1;
            int target_x = x_c_turn + sign * mem.turn_width;

            // Sửa lỗi Bezier: Điểm P1 nằm giữa P0 và P2 để tạo độ cong mượt
            int mid_x = (x_c_turn + target_x) / 2;

            for (int i = idx_turn; i < (int)row_centers.size(); i++) {
                float t = (float)(i - idx_turn) / (row_centers.size() - idx_turn);
                // Bezier bậc 2: P0=(x_c_turn), P1=(mid_x), P2=(target_x)
                float x_new = (1-t)*(1-t)*x_c_turn + 2*t*(1-t)*mid_x + t*t*target_x;
                row_centers[i].first = (int)std::clamp(x_new, 0.0f, (float)w - 1);
            }
        }
    }
};

// ============================================================
// LANE FOLLOWER V22 (V20 Core + TurnPlanner)
// ============================================================
class LaneFollower {
public:
    struct Output {
        double angle = 0, speed = 0;              
        bool valid = false, occl = false, forced_emer = false, emer_auth = false, losing = false;
        double near_e = 0, far_e = 0, eff_error = 0, coverage = 0;
        double angle_pre_gain = 0, kept_frac = 1.0; 
        std::vector<std::pair<int, int>> centerline_pts;
        int proc_width = 0, proc_height = 0;
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
        turn_planner_.reset();
    }
    void softReset() { pid_.softReset(); }

    // Cổng giao tiếp với AI
    void setSign(const std::string& sign_name, int trigger_y = 150) {
        turn_planner_.setSign(sign_name, trigger_y);
    }

    Output step(const cv::Mat& rawMask) {
        Output o;
        double kept_frac = 1.0;
        cv::Mat cleanMask = detail::preprocessMask(rawMask, c_, kept_frac);
        
        if (cleanMask.empty()) { o.state_str = "INVALID MASK"; return o; }
        o.proc_width = cleanMask.cols; o.proc_height = cleanMask.rows;

        std::vector<std::pair<int, int>> row_centers;
        auto cl = extractCenterline(cleanMask, curveSmooth_, c_, row_centers);
        
        // === GỌI TURN PLANNER ĐỂ BẺ CONG CENTERLINE ===
        turn_planner_.modifyCenterline(row_centers, cleanMask.cols);
        
        // Tính lại near/far sau khi bị ghi đè
        if (cl.valid && !row_centers.empty()) {
            std::vector<int> ys;
            for (auto& p : row_centers) ys.push_back(p.second);
            std::sort(ys.begin(), ys.end());
            int yBot = detail::percentileInt(ys, 0.98);
            int yTop = detail::percentileInt(ys, 0.02);
            int span = std::max(yBot - yTop, 1);
            int ySplitNear = yBot - (int)(span * 0.45);
            int ySplitFar  = yTop + (int)(span * 0.35);

            std::vector<int> nearC, farC;
            for (auto& p : row_centers) {
                if (p.second >= ySplitNear) nearC.push_back(p.first);
                if (p.second <= ySplitFar)  farC.push_back(p.first);
            }
            int nearCx = nearC.empty() ? -1 : (int)detail::medianInts(nearC);
            int farCx  = farC.empty()  ? -1 : (int)detail::medianInts(farC);
            
            cl.near_error = (nearCx != -1) ? (nearCx - cleanMask.cols / 2) : 0;
            cl.far_error  = (farCx  != -1) ? (farCx  - cleanMask.cols / 2) : 0;
            int topCx = farC.empty()  ? 0 : (int)detail::medianInts(farC);
            int botCx = nearC.empty() ? 0 : (int)detail::medianInts(nearC);
            cl.curve_error = topCx - botCx;
        }

        curveSmooth_ = 0.90 * curveSmooth_ + 0.10 * cl.curve_error;
        const double nearE = cl.near_error, farE = cl.far_error;
        const bool valid = cl.valid;

        double farDelta = std::clamp(farE - nearE, -c_.far_delta_clip, c_.far_delta_clip);
        double dFarRaw = (valid && prevValid_) ? (farE - prevFarE_) : 0.0;
        double dFar = std::clamp(dFarRaw, -c_.d_far_clip, c_.d_far_clip);
        dFarFilt_ = c_.d_far_ema_alpha * dFar + (1.0 - c_.d_far_ema_alpha) * dFarFilt_;
        double lead = 0.5 * farDelta + c_.lead_rate_gain * dFarFilt_;
        double blended = nearE + lead;
        prevFarE_ = farE;

        bool occl = gate_.step(blended, cl.coverage, row_centers, valid, farE, nearE);
        if (occl) emerHold_ = c_.emg_hold_after_occl;
        else if (emerHold_ > 0) emerHold_--;

        if (!occl && valid) {
            double asymRaw = detectAsymmetry(cleanMask);
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
            angle = angle_pre_gain * c_.post_pid_gain;
            
            double target = detail::planSpeed(curveSmooth_, c_.max_speed, c_.min_speed, c_.curve_scale);
            double errMult = detail::errorSpeedFactor(eff);
            double authSpeed = pid_.losing() ? 0.6 : 1.0;
            double anomSpeed = (std::abs(asymSmooth_) > 15) ? 0.5 : 1.0;
            target = std::max(target * errMult * std::min(authSpeed, anomSpeed), 0.0);
            speed_ = 0.7 * speed_ + 0.3 * target;
            lastAngle_ = angle;

            if (turn_planner_.mem.active) o.state_str = "TURNING " + turn_planner_.mem.side;
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
        o.centerline_pts = row_centers;

        return o;
    }

private:
    Config c_;
    SteeringPID pid_;
    OcclusionGate gate_;
    TurnPlanner turn_planner_;
    double speed_, lastAngle_, curveSmooth_, prevFarE_, dFarFilt_, asymSmooth_;
    bool prevValid_;
    int emerHold_;

    struct CenterlineInfo { double near_error = 0, far_error = 0, curve_error = 0, curve_smooth = 0, coverage = 0; bool valid = false; };
    
    bool largestRun(const uchar* row, int w, int& bs, int& be) {
        int bestLen = -1, s = -1; bs = be = -1;
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

    CenterlineInfo extractCenterline(const cv::Mat& m, double prevCurveSmooth, const Config& c, std::vector<std::pair<int, int>>& row_centers) {
        CenterlineInfo r;
        const int h = m.rows, w = m.cols;
        long nz = 0;
        for (int y = h - 1; y >= 0; y--) {
            const uchar* row = m.ptr<uchar>(y);
            for (int x = 0; x < w; x++) nz += row[x] > 0;
            int s, e;
            if (largestRun(row, w, s, e)) row_centers.push_back({(s + e) / 2, y});
        }
        r.coverage = (double)nz / ((double)h * (double)w);
        if ((int)row_centers.size() < c.min_rows || r.coverage < c.coverage_threshold) {
            r.valid = false; return r;
        }
        std::vector<int> ys;
        for (auto& p : row_centers) ys.push_back(p.second);
        std::sort(ys.begin(), ys.end());
        int yBot = detail::percentileInt(ys, 0.98);
        int yTop = detail::percentileInt(ys, 0.02);
        int span = std::max(yBot - yTop, 1);
        int ySplitNear = yBot - (int)(span * 0.45);
        int ySplitFar  = yTop + (int)(span * 0.35);
        std::vector<int> nearC, farC;
        for (auto& p : row_centers) {
            if (p.second >= ySplitNear) nearC.push_back(p.first);
            if (p.second <= ySplitFar)  farC.push_back(p.first);
        }
        int nearCx = nearC.empty() ? -1 : (int)detail::medianInts(nearC);
        int farCx  = farC.empty()  ? -1 : (int)detail::medianInts(farC);
        r.near_error = (nearCx != -1) ? (nearCx - w / 2) : 0;
        r.far_error  = (farCx  != -1) ? (farCx  - w / 2) : 0;
        int topCx = farC.empty()  ? 0 : (int)detail::medianInts(farC);
        int botCx = nearC.empty() ? 0 : (int)detail::medianInts(nearC);
        r.curve_error  = topCx - botCx;
        r.curve_smooth = 0.90 * prevCurveSmooth + 0.10 * r.curve_error;
        r.valid = true;
        return r;
    }

    double detectAsymmetry(const cv::Mat& m, double maxBias = 45.0, double y0f = 0.18, double y1f = 0.60) {
        const int h = m.rows, w = m.cols;
        auto firstLast = [&](int y, int& a, int& b) {
            const uchar* row = m.ptr<uchar>(y);
            int cnt = 0; a = b = -1;
            for (int x = 0; x < w; x++) if (row[x] > 0) { if (a < 0) a = x; b = x; cnt++; }
            return cnt >= 2;
        };
        std::vector<int> bottomCxs;
        for (int y = (int)(h * 0.80); y < h; y += 4) {
            int a, b; if (firstLast(y, a, b)) bottomCxs.push_back((a + b) / 2);
        }
        if (bottomCxs.empty()) return 0.0;
        int refCx = (int)detail::medianInts(bottomCxs);
        std::vector<int> lws, rws;
        for (int y = (int)(h * y0f); y < (int)(h * y1f); y += 4) {
            int a, b; if (!firstLast(y, a, b)) continue;
            int lw = refCx - a, rw = b - refCx;
            if (lw > 0 && rw > 0) { lws.push_back(lw); rws.push_back(rw); }
        }
        if (lws.empty()) return 0.0;
        double lw = detail::medianInts(lws), rw = detail::medianInts(rws);
        double total = lw + rw;
        if (total < 40) return 0.0;
        return (rw - lw) / total * maxBias;
    }
};

} // namespace v22

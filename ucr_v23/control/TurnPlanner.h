// ============================================================
// control/TurnPlanner.h
//
// Port 1:1 từ openinglogic.py.
// Nhận mask (320x184) + sign name, modify row_centers để PID tự bẻ lái.
// ============================================================
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace v20 {

struct TurnPlannerConfig {
    // Opening detection
    int open_thresh  = 15;
    int min_height   = 10;
    int smooth_win   = 5;
    int base_win     = 50;
    int pull_back    = 3;

    // Bézier / extension
    int corner_radius   = 20;
    int min_turn_dist   = 4;
    int straight_margin = 10;
    int extend_len      = 80;

    // Morph close
    int morph_close_w = 35;
    int morph_close_h = 5;

    // Trigger
    float turn_trigger_y_ratio = 0.75f;

    // Safety
    int max_turn_ticks = 80;
    int cooldown_ticks = 30;
};

class TurnPlanner {
public:
    struct Opening {
        std::string side;
        int y_start = 0, y_end = 0, y_mid = 0;
        int x_extreme = 0, x_baseline = 0;
    };

    struct Memory {
        bool        active      = false;
        std::string side;
        std::string sign_name;
        int         turn_width  = 0;
        int         last_y_mid  = 9999;
        int         cooldown    = 0;
        int         turn_start_tick = -1;
    };

    TurnPlanner() = default;
    explicit TurnPlanner(const TurnPlannerConfig& cfg) : cfg_(cfg) {}

    void reset() { mem_ = Memory(); }
    void resetCooldownOnly() { mem_.cooldown = 0; }
    const Memory& memory() const { return mem_; }
    bool isActive() const { return mem_.active; }

    bool update(const cv::Mat& mask,
                std::vector<std::pair<int,int>>& row_centers,
                const std::string& signName,
                uint32_t tick) {
        if (mem_.cooldown > 0) { --mem_.cooldown; return false; }

        bool active = false;
        std::string side;
        if (signName == "turn_left" || signName == "no_right" ||
            signName == "construct" || signName == "construction") {
            active = true; side = "left";
        } else if (signName == "turn_right" || signName == "no_left") {
            active = true; side = "right";
        }
        if (!active) return false;

        if (signName != mem_.sign_name) {
            mem_.active = false;
            mem_.sign_name = signName;
        }

        cv::Mat cleaned = morphClose(mask, cfg_.morph_close_w, cfg_.morph_close_h);
        std::vector<Opening> openings = detectOpenings(cleaned, side);

        const int H = mask.rows;
        const int y_trigger = static_cast<int>(H * cfg_.turn_trigger_y_ratio);
        bool modified = false;

        if (!openings.empty()) {
            const Opening* op = &openings[0];
            for (const auto& o : openings) if (o.y_mid > op->y_mid) op = &o;

            mem_.side = op->side;
            mem_.turn_width  = std::abs(op->x_extreme - op->x_baseline);
            mem_.last_y_mid  = op->y_mid;

            if (op->y_mid > y_trigger) {
                if ((signName == "no_left" || signName == "no_right") &&
                    !isStraightBlocked(row_centers, op->y_mid)) {
                    mem_.active = false;
                    mem_.turn_start_tick = -1;
                    return false;
                }
                if (mem_.turn_start_tick < 0) mem_.turn_start_tick = (int)tick;
                if ((int)tick - mem_.turn_start_tick > cfg_.max_turn_ticks) {
                    completeTurn();
                    return false;
                }
                mem_.active = true;
                modified = buildPathWithTurn(row_centers, cleaned, *op);
                return modified;
            }
            mem_.active = true;
            mem_.turn_start_tick = -1;
            return false;
        }

        // Ghost mode
        if (mem_.active && !row_centers.empty() && mem_.last_y_mid < 9999) {
            const int top_x = row_centers.back().first;
            const int top_y = row_centers.back().second;

            if (top_y > y_trigger || top_y > mem_.last_y_mid) {
                if ((signName == "no_left" || signName == "no_right") &&
                    !isStraightBlocked(row_centers, top_y)) return false;

                if (mem_.turn_start_tick < 0) mem_.turn_start_tick = (int)tick;
                if ((int)tick - mem_.turn_start_tick > cfg_.max_turn_ticks) {
                    completeTurn();
                    return false;
                }

                Opening ghost;
                ghost.side       = mem_.side;
                ghost.y_mid      = top_y;
                ghost.y_start    = top_y;
                ghost.y_end      = top_y;
                ghost.x_baseline = top_x;
                ghost.x_extreme  = (mem_.side == "left")
                    ? (top_x - mem_.turn_width) : (top_x + mem_.turn_width);
                modified = buildPathWithTurn(row_centers, cleaned, ghost);
                return modified;
            }
        }
        return false;
    }

private:
    static cv::Mat morphClose(const cv::Mat& mask, int w, int h) {
        cv::Mat out;
        if (mask.empty()) return out;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(w, h));
        cv::morphologyEx(mask, out, cv::MORPH_CLOSE, kernel);
        return out;
    }

    static void computeEdgeProfiles(const cv::Mat& mask,
                                    std::vector<float>& xl,
                                    std::vector<float>& xr) {
        const int h = mask.rows, w = mask.cols;
        xl.assign(h, NAN); xr.assign(h, NAN);
        for (int y = 0; y < h; ++y) {
            const uchar* row = mask.ptr<uchar>(y);
            int a = -1, b = -1;
            for (int x = 0; x < w; ++x) if (row[x]) { if (a < 0) a = x; b = x; }
            if (a >= 0) { xl[y] = (float)a; xr[y] = (float)b; }
        }
    }

    static void movingAvgNaN(std::vector<float>& arr, int win) {
        const int h = (int)arr.size();
        const int half = win / 2;
        std::vector<float> out(h, NAN);
        for (int i = 0; i < h; ++i) {
            float sum = 0; int n = 0;
            for (int j = std::max(0, i-half); j <= std::min(h-1, i+half); ++j) {
                if (!std::isnan(arr[j])) { sum += arr[j]; ++n; }
            }
            if (n > 0) out[i] = sum / n;
        }
        arr = std::move(out);
    }

    static void rollingMedianNaN(std::vector<float>& arr, int win) {
        const int h = (int)arr.size();
        const int half = win / 2;
        std::vector<float> out(h, NAN);
        std::vector<float> buf;
        for (int i = 0; i < h; ++i) {
            buf.clear();
            for (int j = std::max(0, i-half); j <= std::min(h-1, i+half); ++j) {
                if (!std::isnan(arr[j])) buf.push_back(arr[j]);
            }
            if (buf.empty()) continue;
            std::sort(buf.begin(), buf.end());
            const size_t n = buf.size();
            out[i] = (n % 2) ? buf[n/2] : (buf[n/2-1] + buf[n/2]) / 2.0f;
        }
        arr = std::move(out);
    }

    std::vector<Opening> findOpeningsFromDev(
        const std::vector<float>& dev,
        const std::vector<float>& x_edge,
        const std::vector<float>& x_base,
        const std::string& side) const {
        std::vector<Opening> out;
        const int h = (int)dev.size();
        bool in_open = false;
        int y_start = 0;

        auto makeOp = [&](int y0, int y1) -> Opening {
            Opening op;
            op.side = side; op.y_start = y0; op.y_end = y1;
            op.y_mid = (y0 + y1) / 2;
            float extreme = NAN;
            for (int y = y0; y <= y1; ++y) {
                if (std::isnan(x_edge[y])) continue;
                if (std::isnan(extreme) ||
                    (side == "left"  && x_edge[y] < extreme) ||
                    (side == "right" && x_edge[y] > extreme)) {
                    extreme = x_edge[y];
                }
            }
            op.x_extreme  = std::isnan(extreme) ? 0 : (int)extreme;
            op.x_baseline = std::isnan(x_base[op.y_mid]) ? 0 : (int)x_base[op.y_mid];
            return op;
        };

        for (int y = 0; y < h; ++y) {
            if (std::isnan(dev[y])) {
                if (in_open) {
                    in_open = false;
                    if (y - y_start >= cfg_.min_height)
                        out.push_back(makeOp(y_start, y-1));
                }
                continue;
            }
            if (dev[y] > cfg_.open_thresh && !in_open) {
                in_open = true; y_start = y;
            } else if (dev[y] <= cfg_.open_thresh && in_open) {
                in_open = false;
                if (y - y_start >= cfg_.min_height)
                    out.push_back(makeOp(y_start, y-1));
            }
        }
        if (in_open && h - y_start >= cfg_.min_height)
            out.push_back(makeOp(y_start, h-1));
        return out;
    }

    std::vector<Opening> detectOpenings(const cv::Mat& mask,
                                        const std::string& side) const {
        std::vector<float> xl, xr;
        computeEdgeProfiles(mask, xl, xr);
        std::vector<float> xl_s = xl, xr_s = xr;
        movingAvgNaN(xl_s, cfg_.smooth_win);
        movingAvgNaN(xr_s, cfg_.smooth_win);
        std::vector<float> bl = xl_s, br = xr_s;
        rollingMedianNaN(bl, cfg_.base_win);
        rollingMedianNaN(br, cfg_.base_win);

        std::vector<float> dev_l(xl.size()), dev_r(xr.size());
        for (size_t i = 0; i < xl.size(); ++i) {
            dev_l[i] = bl[i] - xl_s[i];
            dev_r[i] = xr_s[i] - br[i];
        }
        std::vector<Opening> out;
        if (side == "left" || side == "both")
            for (auto& op : findOpeningsFromDev(dev_l, xl_s, bl, "left"))
                out.push_back(op);
        if (side == "right" || side == "both")
            for (auto& op : findOpeningsFromDev(dev_r, xr_s, br, "right"))
                out.push_back(op);
        return out;
    }

    bool isStraightBlocked(const std::vector<std::pair<int,int>>& cl,
                           int y_mid) const {
        if (cl.empty()) return true;
        return cl.back().second > (y_mid - cfg_.straight_margin);
    }

    bool buildPathWithTurn(std::vector<std::pair<int,int>>& row_centers,
                           const cv::Mat& mask,
                           const Opening& op) {
        if (row_centers.empty()) return false;

        int y_turn_raw = op.y_mid;
        if (y_turn_raw < 0 || y_turn_raw >= mask.rows) return false;
        const uchar* row = mask.ptr<uchar>(y_turn_raw);
        int xl = -1, xr = -1;
        for (int x = 0; x < mask.cols; ++x)
            if (row[x]) { if (xl < 0) xl = x; xr = x; }
        if (xl < 0) return false;

        int x_edge_outer = (op.side == "left")
            ? (xl + cfg_.pull_back) : (xr - cfg_.pull_back);
        int x_turn = (x_edge_outer + op.x_baseline) / 2;

        int idx = 0, best_dy = std::abs(row_centers[0].second - y_turn_raw);
        for (size_t i = 1; i < row_centers.size(); ++i) {
            int dy = std::abs(row_centers[i].second - y_turn_raw);
            if (dy < best_dy) { best_dy = dy; idx = (int)i; }
        }
        const int x_c = row_centers[idx].first;
        const int y_c = row_centers[idx].second;
        const int y_turn = y_c;

        if (y_turn_raw < row_centers.back().second - 20) return false;
        int d = std::abs(x_turn - x_c);
        if (d < cfg_.min_turn_dist) return false;

        const int sign = (x_turn < x_c) ? -1 : 1;
        const int d_bez = std::min(d, cfg_.corner_radius * 2);

        const int A_y = y_c + d_bez;
        int A_idx = idx;
        for (int i = idx; i >= 0; --i) {
            if (row_centers[i].second >= A_y) { A_idx = i; break; }
        }
        if (A_idx == idx) return false;

        const auto A = row_centers[A_idx];
        const int Bx = x_c + sign * d_bez;
        const int By = y_c;

        std::vector<std::pair<int,int>> new_path;
        new_path.reserve(row_centers.size() + 32);
        for (int i = 0; i <= A_idx; ++i) new_path.push_back(row_centers[i]);

        constexpr int n_bez = 10;
        for (int i = 0; i <= n_bez; ++i) {
            float t = (float)i / n_bez;
            float x = (1-t)*(1-t)*A.first  + 2*t*(1-t)*x_c + t*t*Bx;
            float y = (1-t)*(1-t)*A.second + 2*t*(1-t)*y_c + t*t*By;
            new_path.push_back({(int)x, (int)y});
        }

        int cx = x_turn, cy = y_turn;
        new_path.push_back({cx, cy});
        for (int s = 0; s < cfg_.extend_len; s += 5) {
            if (op.side == "left") cx = std::max(0, cx - 5);
            else                   cx = std::min(mask.cols - 1, cx + 5);
            if (cy < 0 || cy >= mask.rows) break;
            if (mask.at<uchar>(cy, cx) == 0) break;
            new_path.push_back({cx, cy});
        }

        row_centers = std::move(new_path);
        return true;
    }

    void completeTurn() {
        mem_.active = false;
        mem_.turn_start_tick = -1;
        mem_.last_y_mid = 9999;
        mem_.turn_width = 0;
        mem_.cooldown   = cfg_.cooldown_ticks;
    }

    TurnPlannerConfig cfg_;
    Memory            mem_;
};

}  // namespace v20
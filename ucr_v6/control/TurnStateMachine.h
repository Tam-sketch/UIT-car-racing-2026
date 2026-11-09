// ============================================================
// control/TurnStateMachine.h
//
// Turn state machine ported from pray.py, with the arming gate and
// path-planning replaced by the V21 opening-detector + quadratic
// bezier algorithm.
//
// Clock contract:
//   - UpdateVoter(classIds, ctrlTick) is called ONCE per sign-detect
//     (every SIGN_DETECT_EVERY control ticks).
//   - Update(ctrlTick, heading, rawMask, centerline_pts) is called ONCE
//     per control tick.
//   - SetTopo(topo) is called once per sign-detect, after SkeletonTopology.
//
// Steering output is in v20's out.angle convention (+ = right turn).
// Caller converts to real-angle left-positive by negating, then clamps.
//
// Arm gate:   opening detected on the force-direction side, with
//             y_mid > TURN_OPENING_PROXIMITY * mask.height.
// Steering:   proportional to (bezier_lookahead_x − mask_center_x).
//             Fallback to ±FORCE_ANGLE if bezier unavailable.
// Completion: measured heading delta from force start.
// ============================================================
#pragma once

#include "control/SignVoter.h"
#include "perception/SkeletonTopology.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace ucr {
namespace control {

struct TurnConfig {
    // --- Force steering fallback ---
    float force_angle               = 22.0f;

    // --- Adaptive force speed (by |Δheading| from force start) ---
    float force_approach_speed   = 8.0f;
    float force_turning_speed    = 5.0f;
    float force_exit_speed       = 12.0f;
    float force_turn_phase_deg   = 15.0f;
    float force_exit_phase_deg   = 50.0f;

    // --- Reset ---
    float heading_reset_deg        = 56.0f;
    int   reset_debounce_ticks     = 90;
    int   sign_snapshot_hold_ticks = 3;
    int   max_force_ticks          = 50;   // safety timeout ~5s at 10Hz

    // --- Speed caps ---
    float slow_speed      = 10.0f;
    float max_speed       = 20.0f;
    int   sign_hold_ticks = 18;

    // --- Opening detection ---
    int   open_thresh             = 15;
    int   open_min_height         = 10;
    int   open_smooth_win         = 5;
    int   open_base_win           = 50;
    float turn_opening_proximity  = 0.55f;
    int   turn_arm_confirm        = 2;

    // --- Bezier path planning ---
    int   bezier_corner_radius    = 20;
    int   bezier_min_turn_dist    = 4;
    int   bezier_extend_len       = 80;
    float bezier_lookahead_frac   = 0.85f;
    float bezier_steer_gain       = 0.30f;

    // --- Class indices (8-class engine) ---
    int idx_construction_ahead = 0;
    int idx_no_left            = 1;
    int idx_no_right           = 2;
    int idx_parking            = 3;
    int idx_roundabout         = 4;
    int idx_traffic_light      = 5;
    int idx_turn_left          = 6;
    int idx_turn_right         = 7;
};

class TurnStateMachine {
public:
    struct Point   { int x, y; };
    struct Opening {
        int  side       = 0;    // -1 = left, +1 = right
        int  y_start    = -1;
        int  y_end      = -1;
        int  y_mid      = -1;
        int  x_extreme  = 0;
        int  x_baseline = 0;
        bool found      = false;
    };

    struct Output {
        bool   in_force_turn     = false;
        int    force_dir         = 0;
        float  force_angle_deg   = 0.0f;
        float  force_speed_cms   = 0.0f;
        float  max_speed_cap_cms = 0.0f;
        bool   sign_committed    = false;
        int    sign_id           = -1;
        const char* phase        = "-";
        float  heading   = 0.0f;
        float  d_heading = 0.0f;
        int    topo_L = 0, topo_S = 1, topo_R = 0;
        bool   topo_reset_fired  = false;
        // Diagnostics
        int    n_openings        = 0;
        int    chosen_y_mid      = -1;
        float  bezier_error_px   = 0.0f;
    };

    TurnStateMachine(const TurnConfig& cfg,
                     int activateFrames,
                     int voterTimeoutTicks)
        : cfg_(cfg)
        , voter_(activateFrames, voterTimeoutTicks) {}

    // ---- Detect-rate calls ----
    void UpdateVoter(const std::vector<int>& classIds, uint32_t ctrlTick) {
        voter_.Update(classIds, ctrlTick);
    }
    void SetTopo(const perception::TopoResult& t) { lastTopo_ = t; }

    // ---- Control-rate call ----
    Output Update(uint32_t ctrlTick,
                  float headingDeg,
                  const cv::Mat& rawMask,
                  const std::vector<std::pair<int,int>>& centerline_pts) {
        Output o;
        o.heading        = headingDeg;
        o.sign_id        = voter_.GetCommitted();
        o.sign_committed = (o.sign_id >= 0);
        o.topo_L = (int)lastTopo_.left;
        o.topo_S = (int)lastTopo_.straight;
        o.topo_R = (int)lastTopo_.right;

        const bool signSeenRecently =
            (voter_.LastSeenTick() >= 0) &&
            (int)ctrlTick - voter_.LastSeenTick() <= cfg_.sign_hold_ticks;

        // --- Timeout / topo reset / snapshot ---
        if (voter_.JustTimedOut()) {
            ResetForce();
            lastCommit_ = -1;
            snapTopoValid_ = false;
        }
        if (o.sign_id >= 0 && o.sign_id != lastCommit_) {
            snapTopo_ = lastTopo_;
            snapTopoValid_ = true;
            lastCommit_ = o.sign_id;
        }
        if (topoReset_.Update(lastTopo_, cfg_.reset_debounce_ticks)) {
            voter_.Reset();
            ResetForce();
            lastCommit_ = -1;
            snapTopoValid_ = false;
            o.topo_reset_fired = true;
        }

        // --- Heading-based reset ---
        float dHeading = 0.0f;
        if (forceActive_ && forceHeading0Valid_) {
            dHeading = AngleDiffDeg(headingDeg, forceHeading0_);
            if (std::abs(dHeading) >= cfg_.heading_reset_deg) {
                voter_.Reset();
                ResetForce();
                lastCommit_ = -1;
                snapTopoValid_ = false;
                dHeading = 0.0f;
            }
        }
        o.d_heading = dHeading;

        // --- Safety timeout ---
        if (forceActive_) {
            if (++forceTicks_ > cfg_.max_force_ticks) {
                voter_.Reset();
                ResetForce();
                lastCommit_ = -1;
                snapTopoValid_ = false;
            }
        }

        // --- Sign snapshot for decision ---
        const int framesSinceSeen = (voter_.LastSeenTick() >= 0)
                                  ? (int)ctrlTick - voter_.LastSeenTick()
                                  : (1 << 30);
        const perception::TopoResult topoDecision =
            (o.sign_id >= 0 && snapTopoValid_ &&
             framesSinceSeen >= cfg_.sign_snapshot_hold_ticks)
            ? snapTopo_ : lastTopo_;

        // --- Arm condition: opening + sign + proximity ---
        const int forceCandidate = DecideForceDir(o.sign_id, topoDecision);
        if (!forceActive_ && forceCandidate != 0) {
            Opening best = FindBestOpening(rawMask, forceCandidate);
            o.n_openings   = lastOpeningsCount_;
            o.chosen_y_mid = best.y_mid;

            if (best.found &&
                best.y_mid > (int)(cfg_.turn_opening_proximity *
                                   (float)rawMask.rows)) {
                if (++armConfirmCount_ >= cfg_.turn_arm_confirm) {
                    forceActive_        = true;
                    forceDir_           = forceCandidate;
                    forceHeading0_      = headingDeg;
                    forceHeading0Valid_ = true;
                    forcePhase_         = "ENTRY";
                    forceTicks_         = 0;
                    armConfirmCount_    = 0;
                    currentOpening_     = best;
                    currentBezier_      = BuildBezier(centerline_pts, rawMask, best);
                }
            } else {
                armConfirmCount_ = 0;
            }
        } else {
            lastOpeningsCount_ = 0;
        }

        // --- Active turn: steer from bezier lookahead ---
        if (forceActive_) {
            // Refresh if opening still visible and centerline has enough points
            Opening fresh = FindBestOpening(rawMask, forceDir_);
            if (fresh.found &&
                centerline_pts.size() >= 5) {
                currentOpening_ = fresh;
                currentBezier_  = BuildBezier(centerline_pts, rawMask, fresh);
            }

            // Compute steering angle from bezier or fallback
            if (!currentBezier_.empty()) {
                const int n = (int)currentBezier_.size();
                int look_idx = (int)(n * cfg_.bezier_lookahead_frac);
                look_idx = std::clamp(look_idx, 0, n - 1);
                float look_x   = (float)currentBezier_[look_idx].x;
                float center_x = (float)rawMask.cols * 0.5f;
                float err_px   = look_x - center_x;
                o.bezier_error_px = err_px;
                o.force_angle_deg = std::clamp(cfg_.bezier_steer_gain * err_px,
                                               -cfg_.force_angle,
                                               +cfg_.force_angle);
            } else {
                o.force_angle_deg = (float)forceDir_ * cfg_.force_angle;
            }

            // Phase by heading delta
            const float absDh = std::abs(dHeading);
            if (absDh < cfg_.force_turn_phase_deg) {
                forcePhase_ = "ENTRY";
                o.force_speed_cms = cfg_.force_approach_speed;
            } else if (absDh < cfg_.force_exit_phase_deg) {
                forcePhase_ = "TURNING";
                o.force_speed_cms = cfg_.force_turning_speed;
            } else {
                forcePhase_ = "EXIT";
                o.force_speed_cms = cfg_.force_exit_speed;
            }

            o.in_force_turn     = true;
            o.force_dir         = forceDir_;
            o.phase             = forcePhase_;
            o.max_speed_cap_cms = cfg_.slow_speed;
        } else {
            o.max_speed_cap_cms =
                (o.sign_committed || signSeenRecently)
                ? cfg_.slow_speed : cfg_.max_speed;
        }
        return o;
    }

    void Reset() {
        voter_.Reset();
        ResetForce();
        topoReset_.Reset();
        lastCommit_ = -1;
        snapTopoValid_ = false;
    }

private:
    // ------------------------------------------------------
    // Sign direction
    // ------------------------------------------------------
    static float AngleDiffDeg(float a, float b) {
        return std::fmod(a - b + 540.0f, 360.0f) - 180.0f;
    }

    int DecideForceDir(int commit, const perception::TopoResult& topo) const {
        if (commit < 0) return 0;
        if (commit == cfg_.idx_turn_left)  return -1;
        if (commit == cfg_.idx_turn_right) return +1;
        const bool hasS = topo.straight;
        if (commit == cfg_.idx_no_left)  return hasS ? 0 : +1;
        if (commit == cfg_.idx_no_right) return hasS ? 0 : -1;
        return 0;
    }

    // ------------------------------------------------------
    // Opening detection (V21 port)
    // ------------------------------------------------------
    static std::vector<float> MovingAvg(const std::vector<float>& arr, int win) {
        const int h = (int)arr.size();
        std::vector<float> out(h, -1.0f);
        const int half = std::max(1, win / 2);
        for (int i = 0; i < h; ++i) {
            float sum = 0.0f; int cnt = 0;
            const int lo = std::max(0, i - half);
            const int hi = std::min(h - 1, i + half);
            for (int j = lo; j <= hi; ++j) {
                if (arr[j] >= 0.0f) { sum += arr[j]; ++cnt; }
            }
            if (cnt > 0) out[i] = sum / (float)cnt;
        }
        return out;
    }

    static std::vector<float> RollingMedian(const std::vector<float>& arr, int win) {
        const int h = (int)arr.size();
        std::vector<float> out(h, -1.0f);
        const int half = std::max(1, win / 2);
        std::vector<float> buf;
        for (int i = 0; i < h; ++i) {
            buf.clear();
            const int lo = std::max(0, i - half);
            const int hi = std::min(h - 1, i + half);
            for (int j = lo; j <= hi; ++j) {
                if (arr[j] >= 0.0f) buf.push_back(arr[j]);
            }
            if (!buf.empty()) {
                std::sort(buf.begin(), buf.end());
                out[i] = buf[buf.size() / 2];
            }
        }
        return out;
    }

    std::vector<Opening> DetectOpenings(const cv::Mat& mask, int side) const {
        std::vector<Opening> openings;
        if (side == 0 || mask.empty() || mask.type() != CV_8UC1) return openings;

        const int h = mask.rows, w = mask.cols;
        std::vector<float> xl(h, -1.0f), xr(h, -1.0f);
        for (int y = 0; y < h; ++y) {
            const uchar* row = mask.ptr<uchar>(y);
            int first = -1, last = -1;
            for (int x = 0; x < w; ++x) {
                if (row[x]) { if (first < 0) first = x; last = x; }
            }
            if (first >= 0) { xl[y] = (float)first; xr[y] = (float)last; }
        }

        const auto xl_s = MovingAvg(xl, cfg_.open_smooth_win);
        const auto xr_s = MovingAvg(xr, cfg_.open_smooth_win);
        const auto bl   = RollingMedian(xl_s, cfg_.open_base_win);
        const auto br   = RollingMedian(xr_s, cfg_.open_base_win);

        std::vector<float> dev_l(h, -1.0f), dev_r(h, -1.0f);
        for (int y = 0; y < h; ++y) {
            if (xl_s[y] >= 0.0f && bl[y] >= 0.0f) dev_l[y] = bl[y] - xl_s[y];
            if (xr_s[y] >= 0.0f && br[y] >= 0.0f) dev_r[y] = xr_s[y] - br[y];
        }

        auto findPeaks = [&](const std::vector<float>& dev,
                             const std::vector<float>& edges,
                             const std::vector<float>& base,
                             int s) {
            bool in_open = false;
            int y_start = 0;
            for (int y = 0; y < h; ++y) {
                if (dev[y] < 0.0f) {
                    if (in_open && y - y_start >= cfg_.open_min_height) {
                        const int y_mid = (y_start + y - 1) / 2;
                        openings.push_back({s, y_start, y - 1, y_mid,
                                            (int)edges[y_mid], (int)base[y_mid], true});
                    }
                    in_open = false;
                    continue;
                }
                if (dev[y] > (float)cfg_.open_thresh && !in_open) {
                    in_open = true; y_start = y;
                } else if (dev[y] <= (float)cfg_.open_thresh && in_open) {
                    if (y - y_start >= cfg_.open_min_height) {
                        const int y_mid = (y_start + y - 1) / 2;
                        openings.push_back({s, y_start, y - 1, y_mid,
                                            (int)edges[y_mid], (int)base[y_mid], true});
                    }
                    in_open = false;
                }
            }
            // Close any open run at the bottom edge
            if (in_open && h - y_start >= cfg_.open_min_height) {
                const int y_mid = (y_start + h - 1) / 2;
                openings.push_back({s, y_start, h - 1, y_mid,
                                    (int)edges[y_mid], (int)base[y_mid], true});
            }
        };

        if (side == -1) findPeaks(dev_l, xl_s, bl, -1);
        else            findPeaks(dev_r, xr_s, br, +1);
        return openings;
    }

    Opening FindBestOpening(const cv::Mat& mask, int side) {
        lastOpeningsCount_ = 0;
        Opening best;
        if (side == 0) return best;

        const auto openings = DetectOpenings(mask, side);
        lastOpeningsCount_ = (int)openings.size();

        for (const auto& op : openings) {
            if (!best.found || op.y_mid > best.y_mid) best = op;
        }
        return best;
    }

    // ------------------------------------------------------
    // Quadratic bezier (V21 port)
    // ------------------------------------------------------
    static std::vector<Point> QuadBezier(Point A, Point C, Point B, int n) {
        std::vector<Point> pts;
        pts.reserve(n + 1);
        for (int i = 0; i <= n; ++i) {
            const float t = (float)i / (float)n;
            const float u = 1.0f - t;
            const float x = u * u * A.x + 2.0f * u * t * C.x + t * t * B.x;
            const float y = u * u * A.y + 2.0f * u * t * C.y + t * t * B.y;
            pts.push_back({(int)std::lround(x), (int)std::lround(y)});
        }
        return pts;
    }

    std::vector<Point> BuildBezier(const std::vector<std::pair<int,int>>& centerline,
                                    const cv::Mat& mask,
                                    const Opening& opening) const {
        if (centerline.empty() || !opening.found || mask.empty()) return {};

        std::vector<Point> cl;
        cl.reserve(centerline.size());
        for (const auto& p : centerline) cl.push_back({p.first, p.second});

        // Point on centerline closest to opening.y_mid
        int idx = 0;
        int min_d = std::numeric_limits<int>::max();
        for (size_t i = 0; i < cl.size(); ++i) {
            const int d = std::abs(cl[i].y - opening.y_mid);
            if (d < min_d) { min_d = d; idx = (int)i; }
        }

        const int x_turn = (opening.x_extreme + opening.x_baseline) / 2;
        const int x_c    = cl[idx].x;
        const int y_c    = cl[idx].y;
        const int dist   = std::abs(x_turn - x_c);
        if (dist < cfg_.bezier_min_turn_dist) return cl;

        const int dir = (x_turn < x_c) ? -1 : +1;
        const int d_bez = std::min(dist, cfg_.bezier_corner_radius * 2);

        // Point A: on centerline just below the turn entry
        const int A_y = y_c + d_bez;
        int A_idx = idx;
        for (int i = idx; i >= 0; --i) {
            if (cl[i].y >= A_y) { A_idx = i; break; }
        }
        const Point A = cl[A_idx];

        const Point B    = {x_c + dir * d_bez, y_c};
        const Point ctrl = {x_c, y_c};

        const auto bez = QuadBezier(A, ctrl, B, 10);

        std::vector<Point> path(cl.begin(), cl.begin() + A_idx + 1);
        path.insert(path.end(), bez.begin(), bez.end());

        // Extension into the side road
        const int step = 5;
        int x_curr = B.x;
        int y_curr = B.y;
        for (int i = 0; i < cfg_.bezier_extend_len / step; ++i) {
            const int nx = x_curr + dir * step;
            if (nx < 0 || nx >= mask.cols) break;
            if (y_curr < 0 || y_curr >= mask.rows) break;
            if (mask.at<uchar>(y_curr, nx) == 0) break;
            x_curr = nx;
            path.push_back({x_curr, y_curr});
        }
        return path;
    }

    // ------------------------------------------------------
    // Topo reset detector
    // ------------------------------------------------------
    class TopoResetDetector {
    public:
        bool Update(const perception::TopoResult& t, int debounce) {
            const int nbits = (int)t.left + (int)t.straight + (int)t.right;
            if (nbits >= 2) { armed_ = true; stable_ = 0; return false; }
            if (armed_ && nbits == 1) {
                if (++stable_ >= debounce) {
                    armed_ = false; stable_ = 0; return true;
                }
            } else {
                stable_ = 0;
            }
            return false;
        }
        void Reset() { armed_ = false; stable_ = 0; }
    private:
        bool armed_  = false;
        int  stable_ = 0;
    };

    void ResetForce() {
        forceActive_        = false;
        forceDir_           = 0;
        forceHeading0Valid_ = false;
        forcePhase_         = "-";
        armConfirmCount_    = 0;
        forceTicks_         = 0;
        currentBezier_.clear();
        currentOpening_ = Opening{};
    }

    // ------------------------------------------------------
    // State
    // ------------------------------------------------------
    TurnConfig          cfg_;
    SignVoter           voter_;
    TopoResetDetector   topoReset_;

    perception::TopoResult lastTopo_{};
    perception::TopoResult snapTopo_{};
    bool snapTopoValid_ = false;
    int  lastCommit_    = -1;

    bool        forceActive_        = false;
    int         forceDir_           = 0;
    float       forceHeading0_      = 0.0f;
    bool        forceHeading0Valid_ = false;
    const char* forcePhase_         = "-";
    int         forceTicks_         = 0;
    int         armConfirmCount_    = 0;

    Opening            currentOpening_{};
    std::vector<Point> currentBezier_{};

    mutable int lastOpeningsCount_ = 0;
};

}  // namespace control
}  // namespace ucr
// ============================================================
// control/TurnController.h
// ============================================================
#pragma once

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace turn {

class LaneTurnController {
public:
    enum class State { IDLE, WAIT_CURVE, WAIT_STRAIGHT, EXECUTE, SETTLE, DONE };
    enum class RoundaboutPhase { NONE = 0, WAIT_ENTRY, ENTER_RIGHT, CIRCLE_LEFT, EXIT_RIGHT };

    double TURN_STEER_DEG      = 32.0;
    double TURN_RAMP_DEG_PER_S = 90.0;
    double MIN_TURN_TIME       = 0.8;
    double TURN_BIAS_PX        = 35.0;
    double turn_floor_current  = 0.0;

    double ROUNDABOUT_ENTER_DEG       = 16.0;
    double ROUNDABOUT_MAX_ENTER_DEG   = 22.0;
    double ROUNDABOUT_CIRCLE_DEG      = 22.0;
    double ROUNDABOUT_EXIT_DEG        = 20.0;
    double ROUNDABOUT_ENTER_BIAS_PX   = 18.0;
    double ROUNDABOUT_CIRCLE_BIAS_PX  = 28.0;
    double ROUNDABOUT_EXIT_BIAS_PX    = 28.0;
    double RB_ENTER_MIN_TIME          = 0.55;
    double RB_ENTER_MAX_TIME          = 0.85;
    double RB_CIRCLE_MIN_TIME         = 0.80;
    double RB_EXIT_MAX_TIME           = 2.40;
    int    RB_BRANCH_OPEN_WIDTH_PX    = 380;
    int    RB_BRANCH_CLOSE_WIDTH_PX   = 340;
    int    RB_ENTRY_STABLE_FRAMES     = 2;
    int    RB_ONROAD_MARGIN_PX        = 40;
    int    RB_OFFROAD_MARGIN_PX       = 20;
    int    rb_entry_stable_frames     = 0;

    int    ALIGN_EDGE_MARGIN        = 5;
    int    ALIGN_MIN_WIDTH_PX       = 60;
    double ALIGN_FAR_CENTER_PX      = 60.0;
    double ALIGN_FAR_OPPOSITE_PX    = 30.0;
    int    REQUIRED_ALIGNED_FRAMES  = 2;
    int    aligned_frames = 0;
    bool   far_swept = false;

    int    JUNCTION_FAR_Y           = 150;
    int    TJUNCTION_OPEN_WIDTH_PX  = 410;
    double TJUNCTION_COOLDOWN_S     = 2.0;

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
    double          tjunction_cooldown      = 0.0;
    int             t_junction_count        = 0;
    int             tjunction_detect_frames = 0;

    State       state          = State::IDLE;
    std::string direction      = "NONE";
    double      turn_elapsed   = 0.0;
    bool        curve_detected = false;
    bool        completed      = false;

    double get_turn_elapsed() const { return turn_elapsed; }

    struct TurnResult {
        State state;
        double steer_bias_px;
        std::string direction;
        double dev              = -1.0;
        bool   junction_opened  = false;
        int    straight_frames  = 0;
        int    roundabout_exit  = 0;
        int    t_junction_count = 0;
        double steer_floor_deg  = 0.0;
        double steer_cap_deg    = 0.0;
        bool   lock_opposite_err = false;
        cv::Point p0 = cv::Point(-1, -1);
        cv::Point p1 = cv::Point(-1, -1);
        cv::Point p2 = cv::Point(-1, -1);
    };

    std::vector<cv::Point> select_three_points(std::vector<cv::Point> points) {
        std::sort(points.begin(), points.end(),
                  [](const cv::Point& a, const cv::Point& b){ return a.y < b.y; });
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
        if (state != State::IDLE) return;

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

        if (is_roundabout_mode) {
            rb_phase = RoundaboutPhase::WAIT_ENTRY;
            std::cout << "[ROUNDABOUT] Mode Active -> Waiting for entry...\n";
        } else {
            rb_phase = RoundaboutPhase::NONE;
            std::cout << "[TURN] Requested: " << dir << " -> Waiting for junction...\n";
        }
    }

    TurnResult update(const std::vector<cv::Point>& left_pts,
                      const std::vector<cv::Point>& right_pts,
                      double dt, bool lane_valid,
                      int far_lx = -1, int far_rx = -1,
                      int mid_lx = -1, int mid_rx = -1,
                      int img_w = 480,
                      bool has_active_sign = false) {
        TurnResult res{state, 0.0, direction};
        res.t_junction_count = t_junction_count;

        const int far_width = (far_lx >= 0 && far_rx >= 0) ? (far_rx - far_lx) : 0;
        const int mid_width = (mid_lx >= 0 && mid_rx >= 0) ? (mid_rx - mid_lx) : 0;

        if (state == State::IDLE) {
            if (tjunction_cooldown > 0.0)
                tjunction_cooldown = std::max(0.0, tjunction_cooldown - dt);

            if (waiting_for_tjunction && tjunction_cooldown <= 0.0
                && !has_active_sign && t_junction_count < 2) {
                const std::string target_dir = (t_junction_count == 0) ? "LEFT" : "RIGHT";
                const auto& t_pts = (target_dir == "LEFT") ? left_pts : right_pts;

                bool open_near = false, open_far = false;
                for (const auto& p : t_pts) {
                    bool touch = (target_dir == "LEFT") ? (p.x <= 5) : (p.x >= img_w - 10);
                    if (!touch) continue;
                    if (p.y >= 160 && p.y <= 210) open_near = true;
                    if (p.y <= JUNCTION_FAR_Y)    open_far  = true;
                }
                const bool wide_t_mouth = (mid_width >= TJUNCTION_OPEN_WIDTH_PX) &&
                    ((target_dir == "LEFT") ? (far_lx >= 0 && far_lx <= 10)
                                            : (far_rx >= img_w - 15));

                if ((open_near && open_far) || wide_t_mouth)
                    tjunction_detect_frames++;
                else
                    tjunction_detect_frames = 0;

                if (tjunction_detect_frames >= 2) {
                    t_junction_count++;
                    waiting_for_tjunction   = false;
                    tjunction_detect_frames = 0;
                    std::cout << "[T-JUNCTION] #" << t_junction_count
                              << " DETECTED -> AUTO TURN " << target_dir << "\n";
                    start_turn(target_dir, false);
                    res.t_junction_count = t_junction_count;
                }
            }
            if (state == State::IDLE) return res;
        }

        if (is_roundabout_mode) {
            turn_elapsed += dt;

            const bool right_open = ((mid_rx >= img_w - 10) && (far_rx >= img_w - 15 || far_rx == -1)) ||
                                    (mid_width >= RB_BRANCH_OPEN_WIDTH_PX);
            const bool right_closed = (far_rx > 0 && far_rx < img_w - 25) &&
                                      (mid_width > 0 && mid_width < RB_BRANCH_CLOSE_WIDTH_PX);

            auto bottom_x = [](const std::vector<cv::Point>& pts) {
                int by = -1, bx = -1;
                for (const auto& p : pts) if (p.y > by) { by = p.y; bx = p.x; }
                return bx;
            };
            const int  bot_lx = bottom_x(left_pts);
            const int  bot_rx = bottom_x(right_pts);
            const bool road_all_right = (bot_lx >= 0) && (bot_lx > img_w / 2 - RB_OFFROAD_MARGIN_PX);
            const bool car_on_road    = lane_valid && bot_lx >= 0 && bot_rx >= 0 &&
                                        (bot_lx < img_w / 2 - RB_ONROAD_MARGIN_PX) &&
                                        (bot_rx > img_w / 2 + RB_ONROAD_MARGIN_PX);

            if (rb_phase == RoundaboutPhase::WAIT_ENTRY) {
                rb_entry_stable_frames = car_on_road ? rb_entry_stable_frames + 1 : 0;
                double r_dev = calculate_deviation(right_pts);
                if (rb_entry_stable_frames >= RB_ENTRY_STABLE_FRAMES &&
                    (right_open || r_dev >= 20.0 || far_width >= 350)) {
                    rb_phase     = RoundaboutPhase::ENTER_RIGHT;
                    turn_elapsed = 0.0;
                    std::cout << "[ROUNDABOUT] >> Phase 1 -> 2: ENTER_RIGHT\n";
                }
            }

            if (rb_phase == RoundaboutPhase::ENTER_RIGHT) {
                res.direction         = "RIGHT";
                res.steer_bias_px     = ROUNDABOUT_ENTER_BIAS_PX;
                res.steer_floor_deg   = ROUNDABOUT_ENTER_DEG;
                res.steer_cap_deg     = road_all_right ? 0.0 : ROUNDABOUT_MAX_ENTER_DEG;
                res.lock_opposite_err = true;

                const bool entered_enough = (turn_elapsed >= RB_ENTER_MIN_TIME) &&
                                            (right_open || (mid_lx > 100));
                if (!road_all_right && (entered_enough || turn_elapsed >= RB_ENTER_MAX_TIME)) {
                    rb_phase                = RoundaboutPhase::CIRCLE_LEFT;
                    turn_elapsed            = 0.0;
                    roundabout_exits_passed = 1;
                    branch1_cleared         = false;
                    rb_phase_Confirm_frames = 0;
                    std::cout << "[ROUNDABOUT] >> Phase 2 -> 3: CIRCLE_LEFT\n";
                }
            }
            else if (rb_phase == RoundaboutPhase::CIRCLE_LEFT) {
                if (road_all_right) {
                    res.direction         = "RIGHT";
                    res.steer_bias_px     = 0.0;
                    res.steer_floor_deg   = 0.0;
                    res.lock_opposite_err = false;
                    turn_elapsed          = std::max(0.0, turn_elapsed - dt);
                    rb_phase_Confirm_frames = 0;
                } else {
                    res.direction         = "LEFT";
                    res.steer_bias_px     = -ROUNDABOUT_CIRCLE_BIAS_PX;
                    res.steer_floor_deg   = -ROUNDABOUT_CIRCLE_DEG;
                    res.lock_opposite_err = true;

                    if (!branch1_cleared) {
                        if (turn_elapsed >= RB_CIRCLE_MIN_TIME &&
                            (right_closed || (far_lx >= 0 && far_lx < img_w / 2))) {
                            rb_phase_Confirm_frames++;
                            if (rb_phase_Confirm_frames >= 2) {
                                branch1_cleared = true;
                                rb_phase_Confirm_frames = 0;
                                std::cout << "[ROUNDABOUT] BRANCH 1 CLEARED\n";
                            }
                        } else rb_phase_Confirm_frames = 0;
                    } else {
                        rb_phase_Confirm_frames = right_open ? rb_phase_Confirm_frames + 1 : 0;
                        if (rb_phase_Confirm_frames >= 2) {
                            roundabout_exits_passed = 2;
                            rb_phase                = RoundaboutPhase::EXIT_RIGHT;
                            turn_elapsed            = 0.0;
                            aligned_frames          = 0;
                            far_swept               = false;
                            turn_bias_current       = ROUNDABOUT_EXIT_BIAS_PX;
                            turn_floor_current      = ROUNDABOUT_EXIT_DEG;
                            std::cout << "[ROUNDABOUT] BRANCH 2 DETECTED -> EXIT RIGHT\n";
                        }
                    }
                }
            }
            else if (rb_phase == RoundaboutPhase::EXIT_RIGHT) {
                res.direction         = "RIGHT";
                res.steer_bias_px     = ROUNDABOUT_EXIT_BIAS_PX;
                res.steer_floor_deg   = ROUNDABOUT_EXIT_DEG;
                res.lock_opposite_err = true;

                const bool far_bounded = (far_lx > ALIGN_EDGE_MARGIN) && (far_rx >= 0) &&
                                         (far_rx < img_w - 1 - ALIGN_EDGE_MARGIN) &&
                                         (far_rx - far_lx >= ALIGN_MIN_WIDTH_PX);
                if (!far_bounded) far_swept = true;

                const double far_center_off = far_bounded
                    ? (0.5 * (far_lx + far_rx) - 0.5 * img_w) : 9999.0;
                const bool far_centered = far_bounded &&
                                          (far_center_off <  ALIGN_FAR_CENTER_PX) &&
                                          (far_center_off > -ALIGN_FAR_OPPOSITE_PX);

                if (turn_elapsed >= 0.6 && lane_valid && far_swept && far_centered)
                    aligned_frames++;
                else
                    aligned_frames = 0;

                if (aligned_frames >= REQUIRED_ALIGNED_FRAMES || turn_elapsed >= RB_EXIT_MAX_TIME) {
                    is_roundabout_mode      = false;
                    rb_phase                = RoundaboutPhase::NONE;
                    roundabout_exits_passed = 0;
                    branch1_cleared         = false;
                    rb_phase_Confirm_frames = 0;

                    waiting_for_tjunction   = true;
                    tjunction_cooldown      = TJUNCTION_COOLDOWN_S;

                    state              = State::SETTLE;
                    direction          = "RIGHT";
                    settle_start_bias  = ROUNDABOUT_EXIT_BIAS_PX;
                    settle_start_floor = ROUNDABOUT_EXIT_DEG;
                    settle_elapsed     = 0.0;
                    std::cout << "[ROUNDABOUT] ENTERED NEW LANE, ARM T-JUNCTION COUNTER\n";
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
        bool junction_opened = side_open_near && side_open_far;

        bool pts_stuck_on_border = false;
        if (selected_pts.size() == 3) {
            if (direction == "LEFT" &&
                selected_pts[0].x <= 3 && selected_pts[1].x <= 3 && selected_pts[2].x <= 3)
                pts_stuck_on_border = true;
            else if (direction == "RIGHT" &&
                     selected_pts[0].x >= img_w - 4 &&
                     selected_pts[1].x >= img_w - 4 &&
                     selected_pts[2].x >= img_w - 4)
                pts_stuck_on_border = true;
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
                std::cout << "[TURN] JUNCTION DETECTED\n";
            }
        }
        else if (state == State::WAIT_STRAIGHT) {
            if (dev >= 0.0 && dev <= STRAIGHT_THRESHOLD && junction_opened &&
                (!pts_stuck_on_border || mid_width >= 360))
                straight_frames++;
            else
                straight_frames = 0;

            if (straight_frames >= REQUIRED_STRAIGHT_FRAMES) {
                state          = State::EXECUTE;
                turn_elapsed   = 0.0;
                aligned_frames = 0;
                far_swept      = false;
                const double dir_sign = (direction == "LEFT") ? -1.0 : 1.0;
                turn_bias_current  = dir_sign * TURN_BIAS_PX;
                turn_floor_current = dir_sign * std::min(TURN_STEER_DEG, TURN_RAMP_DEG_PER_S * dt);
                res.steer_bias_px     = turn_bias_current;
                res.steer_floor_deg   = turn_floor_current;
                res.lock_opposite_err = true;
                std::cout << "[TURN] >>> EXECUTING\n";
            }
        }
        else if (state == State::EXECUTE) {
            turn_elapsed += dt;
            const double dir_sign = (direction == "LEFT") ? -1.0 : 1.0;

            const double mag = std::min(TURN_STEER_DEG,
                std::abs(turn_floor_current) + TURN_RAMP_DEG_PER_S * dt);
            turn_floor_current = dir_sign * mag;

            const bool far_bounded = (far_lx > ALIGN_EDGE_MARGIN) && (far_rx >= 0) &&
                                     (far_rx < img_w - 1 - ALIGN_EDGE_MARGIN) &&
                                     (far_rx - far_lx >= ALIGN_MIN_WIDTH_PX);
            if (!far_bounded) far_swept = true;

            const double far_center_off = far_bounded
                ? (0.5 * (far_lx + far_rx) - 0.5 * img_w) : 9999.0;
            const double off_dir = dir_sign * far_center_off;
            const bool far_centered = far_bounded &&
                (off_dir <  ALIGN_FAR_CENTER_PX) &&
                (off_dir > -ALIGN_FAR_OPPOSITE_PX);

            const bool lane_aligned = (turn_elapsed >= MIN_TURN_TIME) &&
                                      lane_valid && far_swept && far_centered;

            aligned_frames = lane_aligned ? aligned_frames + 1 : 0;

            res.steer_bias_px     = turn_bias_current;
            res.steer_floor_deg   = turn_floor_current;
            res.lock_opposite_err = true;

            if (aligned_frames >= REQUIRED_ALIGNED_FRAMES) {
                state              = State::SETTLE;
                settle_start_bias  = turn_bias_current;
                settle_start_floor = turn_floor_current;
                settle_elapsed     = 0.0;
                std::cout << "[TURN] ALIGNED -> SETTLE\n";
            }

            if (turn_elapsed >= MAX_TURN_TIME && state == State::EXECUTE) {
                state              = State::SETTLE;
                settle_start_bias  = turn_bias_current;
                settle_start_floor = turn_floor_current;
                settle_elapsed     = 0.0;
                std::cout << "[TURN] TIMEOUT -> RELEASE\n";
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
                turn_bias_current   = 0.0;
                turn_floor_current  = 0.0;
                res.steer_bias_px   = 0.0;
                res.steer_floor_deg = 0.0;
                state     = State::DONE;
                completed = true;
                std::cout << "[TURN] BIAS RELEASED\n";
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

class CornerAssist {
public:
    enum class Phase { IDLE, ACTIVE, RELEASE };

    double CORNER_STEER_DEG      = 45.0;
    double CORNER_RAMP_DEG_PER_S = 60.0;
    double CORNER_RELEASE_TIME   = 0.4;
    double CORNER_MIN_TIME       = 0.6;
    double CORNER_MAX_TIME       = 3.0;
    double CORNER_COOLDOWN_S     = 1.0;
    int    CORNER_BEND_PX        = 180;
    int    CORNER_CONFIRM_FRAMES = 2;
    int    EDGE_TOUCH_PX         = 3;
    int    ALIGN_CENTER_PX       = 70;
    int    ALIGN_MIN_WIDTH_PX    = 60;
    int    OVERTURN_PX           = 40;

    void reset() {
        phase_ = Phase::IDLE; dir_ = 0; pending_dir_ = 0; confirm_ = 0;
        aligned_ = 0; elapsed_ = 0.0; floor_ = 0.0;
        release_start_ = 0.0; release_t_ = 0.0;
    }
    bool   active()    const { return phase_ != Phase::IDLE; }
    int    direction() const { return dir_; }
    double floor()     const { return floor_; }

    double update(int lx1, int rx1, int lx2, int rx2,
                  int lx_bot, int rx_bot, int w, double dt) {
        const int  edgeR    = w - 1 - EDGE_TOUCH_PX;
        const bool far_none = (lx1 < 0 || rx1 < 0);

        const bool right_sig = (rx2 >= edgeR) && (rx_bot >= edgeR) &&
                               (lx2 > EDGE_TOUCH_PX + 2) &&
                               (far_none || (rx1 >= edgeR && (lx1 - lx2) >= CORNER_BEND_PX));
        const bool left_sig  = (lx2 >= 0 && lx2 <= EDGE_TOUCH_PX) &&
                               (lx_bot >= 0 && lx_bot <= EDGE_TOUCH_PX) &&
                               (rx2 >= 0 && rx2 < edgeR - 2) &&
                               (far_none || (lx1 >= 0 && lx1 <= EDGE_TOUCH_PX &&
                                             (rx2 - rx1) >= CORNER_BEND_PX));

        if (cooldown_ > 0.0) cooldown_ = std::max(0.0, cooldown_ - dt);

        if (phase_ == Phase::IDLE) {
            floor_ = 0.0;
            if (cooldown_ > 0.0) { confirm_ = 0; pending_dir_ = 0; return 0.0; }
            const int sig = right_sig ? 1 : (left_sig ? -1 : 0);
            if (sig != 0 && sig == pending_dir_) confirm_++;
            else { pending_dir_ = sig; confirm_ = (sig != 0) ? 1 : 0; }

            if (confirm_ >= CORNER_CONFIRM_FRAMES) {
                phase_ = Phase::ACTIVE; dir_ = pending_dir_;
                elapsed_ = 0.0; aligned_ = 0; floor_ = 0.0;
                std::cout << "[CORNER] SHARP " << (dir_ > 0 ? "RIGHT" : "LEFT")
                          << " -> ASSIST ON\n";
            }
        }
        else if (phase_ == Phase::ACTIVE) {
            elapsed_ += dt;
            const double mag = std::min(CORNER_STEER_DEG,
                std::abs(floor_) + CORNER_RAMP_DEG_PER_S * dt);
            floor_ = dir_ * mag;

            const bool far_bounded = (lx1 > EDGE_TOUCH_PX) && (rx1 >= 0) && (rx1 < edgeR) &&
                                     (rx1 - lx1 >= ALIGN_MIN_WIDTH_PX);
            const double far_off = far_bounded ? (0.5 * (lx1 + rx1) - 0.5 * w) : 9999.0;
            const bool aligned = (elapsed_ >= CORNER_MIN_TIME) && far_bounded &&
                                 (std::abs(far_off) < ALIGN_CENTER_PX);
            aligned_ = aligned ? aligned_ + 1 : 0;

            const bool overturn = (dir_ > 0)
                ? (rx_bot >= 0 && rx_bot < w / 2 + OVERTURN_PX)
                : (lx_bot >= 0 && lx_bot > w / 2 - OVERTURN_PX);

            if (aligned_ >= 2 || overturn || elapsed_ >= CORNER_MAX_TIME) {
                phase_ = Phase::RELEASE;
                release_start_ = floor_; release_t_ = 0.0;
                std::cout << "[CORNER] RELEASE ("
                          << (aligned_ >= 2 ? "aligned" : (overturn ? "overturn" : "timeout"))
                          << ")\n";
            }
        }
        else {
            release_t_ += dt;
            const double r = std::clamp(1.0 - release_t_ / std::max(CORNER_RELEASE_TIME, 0.01),
                                        0.0, 1.0);
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
    double elapsed_ = 0.0, floor_ = 0.0;
    double release_start_ = 0.0, release_t_ = 0.0, cooldown_ = 0.0;
};

}  // namespace turn
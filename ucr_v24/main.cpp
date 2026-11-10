// ============================================================
// main.cpp — UCR 2026 V6.4
// (Roundabout bias split RIGHT/LEFT, UI bias removed)
// ============================================================
#include "hardware/Camera.h"
#include "hardware/Motor.h"
#include "hardware/Servo.h"
#include "common/Logger.h"
#include "control/LaneFollowerV21.h"
#include "control/SignVoter.h"
#include "perception/RoadSegmenter.h"
#include "perception/SignDetector.h"
#include "config/RobotConfig.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

volatile std::sig_atomic_t g_stopSignal = 0;
static void SignalHandler(int) { g_stopSignal = 1; }

// ============================================================
// Sign class → v21::SignState
// ============================================================
static v21::SignState MakeSignState(int committed) {
    v21::SignState s;
    if (committed < 0) return s;

    using namespace ucr::config::sign_class;
    s.class_id = committed;
    switch (committed) {
        case TURN_LEFT:  s.active = true; s.side = -1; break;
        case TURN_RIGHT: s.active = true; s.side = +1; break;
        case NO_LEFT:    s.active = true; s.side = +1; break;
        case NO_RIGHT:   s.active = true; s.side = -1; break;
        default:         s.active = false; s.side = 0; break;
    }
    return s;
}

// ============================================================
// Roundabout state
// ============================================================
enum class RbState {
    NONE, APPROACH, IN_CIRCLE, EXIT_PREP, EXIT_SEARCH, EXITING
};

static const char* RbStateName(RbState s) {
    switch (s) {
        case RbState::APPROACH:    return "APPROACH";
        case RbState::IN_CIRCLE:   return "IN_CIRCLE";
        case RbState::EXIT_PREP:   return "EXIT_PREP";
        case RbState::EXIT_SEARCH: return "EXIT_SEARCH";
        case RbState::EXITING:     return "EXITING";
        default:                   return "NONE";
    }
}

// ============================================================
// Traffic light classifier
// ============================================================
enum class LightColor { NONE, RED, YELLOW, GREEN };

static LightColor ClassifyTrafficLight(const cv::Mat& bgr, const cv::Rect2f& bbox) {
    using namespace ucr::config::tl;

    const float cx = bbox.x + bbox.width  * 0.5f;
    const float cy = bbox.y + bbox.height * 0.5f;
    const float w  = bbox.width  * ROI_EXPAND;
    const float h  = bbox.height * ROI_EXPAND;
    const int x1 = std::max(0, static_cast<int>(cx - w * 0.5f));
    const int y1 = std::max(0, static_cast<int>(cy - h * 0.5f));
    const int x2 = std::min(bgr.cols, static_cast<int>(cx + w * 0.5f));
    const int y2 = std::min(bgr.rows, static_cast<int>(cy + h * 0.5f));
    if (x2 - x1 < 4 || y2 - y1 < 4) return LightColor::NONE;

    cv::Mat hsv;
    cv::cvtColor(bgr(cv::Rect(x1, y1, x2 - x1, y2 - y1)), hsv, cv::COLOR_BGR2HSV);

    cv::Mat maskRed1, maskRed2, maskRed;
    cv::inRange(hsv, cv::Scalar(RED_H1_LO, S_MIN, V_MIN),
                     cv::Scalar(RED_H1_HI, 255, 255), maskRed1);
    cv::inRange(hsv, cv::Scalar(RED_H2_LO, S_MIN, V_MIN),
                     cv::Scalar(RED_H2_HI, 255, 255), maskRed2);
    cv::bitwise_or(maskRed1, maskRed2, maskRed);
    int nRed = cv::countNonZero(maskRed);

    cv::Mat maskYel;
    cv::inRange(hsv, cv::Scalar(YEL_H_LO, S_MIN, V_MIN),
                     cv::Scalar(YEL_H_HI, 255, 255), maskYel);
    int nYel = cv::countNonZero(maskYel);

    cv::Mat maskGrn;
    cv::inRange(hsv, cv::Scalar(GRN_H_LO, S_MIN, V_MIN),
                     cv::Scalar(GRN_H_HI, 255, 255), maskGrn);
    int nGrn = cv::countNonZero(maskGrn);

    if (nRed >= MIN_LIT_PIXELS && nRed > nYel && nRed > nGrn) return LightColor::RED;
    if (nYel >= MIN_LIT_PIXELS && nYel > nRed && nYel > nGrn) return LightColor::YELLOW;
    if (nGrn >= MIN_LIT_PIXELS && nGrn > nRed && nGrn > nYel) return LightColor::GREEN;
    return LightColor::NONE;
}

// ============================================================
// MAIN
// ============================================================
int main() {
    std::signal(SIGINT,  SignalHandler);
    std::signal(SIGTERM, SignalHandler);

    Camera camera;
    if (!camera.Init()) { std::cerr << "Camera init failed\n"; return -1; }

    Motor motor;
    if (!motor.Init()) { std::cerr << "Motor init failed\n"; return -1; }
    motor.SetUartLogging(false);

    Servo servo;
    if (!servo.Init() || !servo.SetPWMFreq(50)) {
        std::cerr << "Servo init failed\n"; motor.SetSpeedCms(0.0f); return -1;
    }
    motor.SetSpeedCms(0.0f);
    servo.SetAngle(0);

    ucr::perception::RoadSegmenter road;
    ucr::perception::RoadSegmenter::Config roadCfg;
    roadCfg.daChannel        = ucr::config::ROAD_DA_CHANNEL;
    roadCfg.confidenceMargin = ucr::config::ROAD_CONF_MARGIN;
    if (!road.Init(ucr::config::ROAD_ENGINE_PATH, roadCfg)) {
        std::cerr << "Road model init failed\n"; return -1;
    }

    ucr::perception::SignDetector sign;
    ucr::perception::SignDetector::Config signCfg;
    signCfg.conf_thresh = ucr::config::SIGN_CONF_THRESH;
    signCfg.iou_thresh  = ucr::config::SIGN_IOU_THRESH;
    signCfg.input_size  = ucr::config::SIGN_INPUT_SIZE;
    bool signReady = sign.Init(ucr::config::SIGN_ENGINE_PATH, signCfg);
    if (!signReady) std::cerr << "[Main] Sign detector init failed — turns DISABLED\n";

    ucr::control::SignVoter voter(
        ucr::config::SIGN_VOTER_ACTIVATE_FRAMES,
        ucr::config::SIGN_VOTER_TIMEOUT_FRAMES);

    v21::Config cfg;
    cfg.max_speed          = ucr::config::MAX_SPEED_CMS;
    cfg.min_speed          = ucr::config::MIN_SPEED_CMS;
    cfg.curve_scale        = ucr::config::CURVE_SCALE;
    cfg.post_pid_gain      = ucr::config::POST_PID_GAIN;
    cfg.min_rows           = ucr::config::MIN_ROWS;
    cfg.coverage_threshold = ucr::config::COVERAGE_THRESHOLD;
    cfg.open_thresh        = ucr::config::OPEN_THRESH;
    cfg.min_height         = ucr::config::OPEN_MIN_HEIGHT;
    cfg.smooth_win         = ucr::config::OPEN_SMOOTH_WIN;
    cfg.base_win           = ucr::config::OPEN_BASE_WIN;
    cfg.corner_radius      = ucr::config::BEZIER_CORNER_RADIUS;
    cfg.min_turn_dist      = ucr::config::BEZIER_MIN_TURN_DIST;
    cfg.extend_len         = ucr::config::BEZIER_EXTEND_LEN;
    cfg.turn_min_ticks         = ucr::config::TURN_MIN_TICKS;
    cfg.turn_max_ticks         = ucr::config::TURN_MAX_TICKS;
    cfg.turn_synth_width_px    = ucr::config::TURN_SYNTH_WIDTH_PX;
    cfg.turn_release_road_frac = ucr::config::TURN_RELEASE_ROAD_FRAC;
    cfg.turn_release_window_px = ucr::config::TURN_RELEASE_WINDOW_PX;
    cfg.turn_commit_proximity  = ucr::config::TURN_COMMIT_PROXIMITY;
    cfg.turn_complete_deg      = ucr::config::TURN_COMPLETE_DEG;
    cfg.bez_look_dist_base     = ucr::config::BEZ_LOOK_DIST_BASE;
    cfg.bez_look_dist_scale    = ucr::config::BEZ_LOOK_DIST_SCALE;
    cfg.bez_err_clamp_px       = ucr::config::BEZ_ERR_CLAMP_PX;
    cfg.bez_ctrl_offset_frac   = ucr::config::BEZ_CTRL_OFFSET_FRAC;
    cfg.rb_bias_ema            = ucr::config::RB_BIAS_EMA;
    v21::LaneFollower follower(cfg);

    cv::Mat frame, small_mask;
    v21::LaneFollower::Output out;
    using SignDetection = ucr::perception::SignDetector::Detection;
    std::vector<SignDetection> lastDets;

    double fps = 0.0;
    int    frameCount = 0;
    auto   tPrev = std::chrono::high_resolution_clock::now();

    auto     next_control  = std::chrono::steady_clock::now();
    float    applied_speed = 0.0f;
    uint32_t tickIdx       = 0;
    int lastPersonSeenTick = -1;
    int lastRedSeenTick    = -1;

    // --- Roundabout state ---
    RbState  rb_state            = RbState::NONE;
    uint32_t rb_enter_tick       = 0;
    int      rb_missing_sign     = 0;
    int      rb_exit_search_tick = 0;
    int      rb_cooldown_ticks   = 0;

    std::cout << "\n=== UCR2026 V6.4 (turns="
              << (signReady ? "yes" : "no") << ") ===\n"
              << "Road: " << ucr::config::ROAD_ENGINE_PATH << "\n";
    if (signReady)
        std::cout << "Sign: " << ucr::config::SIGN_ENGINE_PATH << "\n";
    std::cout << "Control: " << ucr::config::CONTROL_PERIOD_MS << " ms\n"
              << "Roundabout: bias R="
              << ucr::config::RB_BIAS_PX_RIGHT
              << " L=" << ucr::config::RB_BIAS_PX_LEFT << "\n"
              << "Press 'q' to quit\n\n";

    while (!g_stopSignal) {
        camera.GetFrame(frame);

        if (frame.empty()) {
            motor.SetSpeedCms(0.0f);
            servo.SetAngle(0);
            applied_speed = 0.0f;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        const bool inferOK = road.Predict(frame, small_mask);
        const auto now     = std::chrono::steady_clock::now();

        if (!inferOK) {
            motor.SetSpeedCms(0.0f);
            servo.SetAngle(0);
            follower.resetTracking();
            applied_speed = 0.0f;
        }
        else if (now >= next_control) {

            // =================================================
            // 1. Sign detection (throttled)
            // =================================================
            if (signReady && (tickIdx % ucr::config::SIGN_DETECT_EVERY) == 0) {
                sign.Detect(frame, lastDets);

                using namespace ucr::config::sign_class;
                std::vector<int> classIds;

                for (auto& d : lastDets) {
                    if (d.confidence < ucr::config::SIGN_CONF_THRESH) continue;

                    if (d.class_id == WALKING_PERSON) {
                        lastPersonSeenTick = static_cast<int>(tickIdx);
                        continue;
                    }

                    if (d.class_id == TRAFFIC_LIGHT) {
                        LightColor color = ClassifyTrafficLight(frame, d.bbox);
                        if (color == LightColor::RED) {
                            lastRedSeenTick = static_cast<int>(tickIdx);
                        } else if (color == LightColor::GREEN) {
                            lastRedSeenTick = -1;
                        }
                        continue;
                    }

                    int cls = (d.class_id == CONSTRUCTION_AHEAD) ? TURN_LEFT : d.class_id;
                    if (cls == TURN_LEFT || cls == TURN_RIGHT ||
                        cls == NO_LEFT   || cls == NO_RIGHT ||
                        cls == ROUNDABOUT) {
                        classIds.push_back(cls);
                    }
                }
                voter.Update(classIds, tickIdx);
            }

            // =================================================
            // 2. Roundabout state machine
            // =================================================
            bool rb_visible_now = false;
            for (const auto& d : lastDets) {
                if (d.class_id == ucr::config::sign_class::ROUNDABOUT &&
                    d.confidence >= ucr::config::SIGN_CONF_THRESH) {
                    rb_visible_now = true;
                    break;
                }
            }

            const bool rb_committed =
                (voter.GetCommitted() == ucr::config::sign_class::ROUNDABOUT);

            if (rb_cooldown_ticks > 0) --rb_cooldown_ticks;

            // Track "missing sign" for APPROACH & EXIT_PREP
            if (rb_state == RbState::APPROACH ||
                rb_state == RbState::EXIT_PREP) {
                if (rb_visible_now) rb_missing_sign = 0;
                else                ++rb_missing_sign;
            }

            switch (rb_state) {
                case RbState::NONE:
                    if (rb_committed && rb_cooldown_ticks <= 0) {
                        rb_state = RbState::APPROACH;
                        rb_missing_sign = 0;
                    }
                    break;

                case RbState::APPROACH:
                    if (rb_missing_sign >= ucr::config::RB_MISSING_SIGN_TICKS) {
                        rb_state = RbState::IN_CIRCLE;
                        rb_enter_tick = tickIdx;
                    }
                    break;

                case RbState::IN_CIRCLE: {
                    const bool timeout =
                        (int)(tickIdx - rb_enter_tick) >= ucr::config::RB_CIRCLE_TIMEOUT;
                    const bool rb_seen_again = rb_committed && rb_visible_now;
                    if (timeout || rb_seen_again) {
                        rb_state = RbState::EXIT_PREP;
                        rb_missing_sign = 0;
                    }
                    break;
                }

                case RbState::EXIT_PREP:
                    if (rb_missing_sign >= ucr::config::RB_MISSING_SIGN_TICKS) {
                        rb_state = RbState::EXIT_SEARCH;
                        rb_exit_search_tick = 0;
                    }
                    break;

                case RbState::EXIT_SEARCH:
                    ++rb_exit_search_tick;
                    if (rb_exit_search_tick >= ucr::config::RB_EXIT_SEARCH_TIMEOUT) {
                        rb_state = RbState::EXITING;
                    }
                    break;

                case RbState::EXITING:
                    // Post-step handles transition to NONE
                    break;
            }

            // =================================================
            // 3. Compute bias + force sign by state
            // =================================================
            v21::SignState signState = MakeSignState(voter.GetCommitted());
            float rb_bias_px = 0.0f;

            switch (rb_state) {
                case RbState::APPROACH:
                    signState.active = false; signState.side = 0;
                    rb_bias_px = +ucr::config::RB_BIAS_PX_RIGHT;
                    break;
                case RbState::IN_CIRCLE:
                    signState.active = false; signState.side = 0;
                    rb_bias_px = -ucr::config::RB_BIAS_PX_LEFT;
                    break;
                case RbState::EXIT_PREP:
                    signState.active = false; signState.side = 0;
                    rb_bias_px = +ucr::config::RB_BIAS_PX_RIGHT;
                    break;
                case RbState::EXIT_SEARCH:
                    signState.active   = true;
                    signState.side     = +1;
                    signState.class_id = 7;   // fake turn_right → Bezier
                    rb_bias_px = -ucr::config::RB_BIAS_PX_LEFT;
                    break;
                case RbState::EXITING:
                    signState.active = false; signState.side = 0;
                    rb_bias_px = 0.0f;
                    break;
                default:
                    rb_bias_px = 0.0f;
                    break;
            }

            // =================================================
            // 4. PID step with roundabout bias
            // =================================================
            out = follower.step(small_mask, signState, NAN, rb_bias_px);

            // =================================================
            // 5. Post-step transitions
            // =================================================
            if (rb_state == RbState::EXIT_SEARCH &&
                out.state_str.find("TURN") != std::string::npos) {
                rb_state = RbState::EXITING;
            }
            if (rb_state == RbState::EXITING &&
                out.state_str == "CENTERLINE") {
                rb_state = RbState::NONE;
                rb_cooldown_ticks = ucr::config::RB_COOLDOWN_TICKS;
                voter.Reset();
            }

            // =================================================
            // 6. Steering
            // =================================================
            float desired_real_left = -static_cast<float>(out.angle);

            desired_real_left = std::clamp(desired_real_left,
                                           -ucr::config::MAX_REAL_ANGLE_DEG,
                                            ucr::config::MAX_REAL_ANGLE_DEG);
            float servo_cmd = ucr::config::RealAngleToServo(desired_real_left);
            servo_cmd = std::clamp(servo_cmd,
                                   -static_cast<float>(ucr::config::MAX_SERVO_CMD),
                                    static_cast<float>(ucr::config::MAX_SERVO_CMD));
            const int final_angle = static_cast<int>(std::lround(servo_cmd));

            // =================================================
            // 7. Speed
            // =================================================
            float speed = static_cast<float>(out.speed);
            if (!out.valid && out.coverage > 0.02)
                speed = ucr::config::MIN_SPEED_CMS;

            // Roundabout speed cap
            if (rb_state != RbState::NONE) {
                speed = std::min(speed, ucr::config::RB_SPEED_CAP_CMS);
            }

            // Stop overrides (person priority > red light)
            const bool stopForPerson =
                (lastPersonSeenTick >= 0) &&
                ((int)tickIdx - lastPersonSeenTick
                 <= ucr::config::PERSON_STOP_HANGOVER_TICKS);
            const bool stopForRedLight =
                (lastRedSeenTick >= 0) &&
                ((int)tickIdx - lastRedSeenTick
                 <= ucr::config::RED_LIGHT_HANGOVER_TICKS);
            if (stopForPerson || stopForRedLight) speed = 0.0f;

            // Soft-start ramp
            const float max_delta = ucr::config::MAX_ACCEL_CMS2 *
                                    (ucr::config::CONTROL_PERIOD_MS / 1000.0f);
            if (speed > applied_speed)
                applied_speed = std::min(speed, applied_speed + max_delta);
            else
                applied_speed = speed;

            motor.SetSpeedCms(applied_speed);
            servo.SetAngle(final_angle);

            // =================================================
            // 8. Advance tick + log
            // =================================================
            ++tickIdx;
            next_control += std::chrono::milliseconds(ucr::config::CONTROL_PERIOD_MS);
            if (now - next_control > std::chrono::seconds(1))
                next_control = now + std::chrono::milliseconds(ucr::config::CONTROL_PERIOD_MS);

            static auto lastLog = std::chrono::steady_clock::now();
            if (now - lastLog >= std::chrono::seconds(1)) {
                const char* stop_str = stopForPerson   ? "STOP:person"
                                     : stopForRedLight ? "STOP:red"
                                     :                   "-";
                std::printf(
                    "[t=%u] sign=%-11s(%d) state=%-15s rb=%-11s "
                    "bias=%+5.1f near=%6.1f far=%6.1f %-11s "
                    "spd=%.1f pid=%.1f out=%.1f\n",
                    tickIdx,
                    ucr::config::SignClassName(voter.GetCommitted()),
                    voter.GetCommitted(),
                    out.state_str.c_str(),
                    RbStateName(rb_state),
                    rb_bias_px,
                    out.near_e, out.far_e,
                    stop_str,
                    applied_speed, out.angle_pre_gain, out.angle);
                lastLog = now;
            }
        }

        // --- FPS ---
        ++frameCount;
        auto tNow = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(tNow - tPrev).count();
        if (dt > 0.5) { fps = frameCount / dt; frameCount = 0; tPrev = tNow; }

        // =====================================================
        // UI  (bias line removed)
        // =====================================================
        if (ucr::config::SHOW_UI) {
            cv::Mat overlay = frame.clone();

            if (inferOK) {
                cv::Mat ui_mask;
                cv::resize(small_mask, ui_mask, frame.size(), 0, 0, cv::INTER_NEAREST);
                cv::Mat green(frame.size(), frame.type(), cv::Scalar(0, 255, 0));
                cv::Mat blended;
                cv::addWeighted(frame, 0.6, green, 0.4, 0, blended);
                blended.copyTo(overlay, ui_mask);

                if (!out.final_path.empty()
                    && out.proc_width > 0 && out.proc_height > 0) {
                    const float sx = float(frame.cols) / out.proc_width;
                    const float sy = float(frame.rows) / out.proc_height;
                    std::vector<cv::Point> pts;
                    for (const auto& p : out.final_path)
                        pts.emplace_back(int(p.x * sx), int(p.y * sy));

                    cv::Scalar color(0, 0, 255);
                    if (out.state_str.find("TURN")  != std::string::npos) color = cv::Scalar(255, 0, 255);
                    if (out.state_str.find("GHOST") != std::string::npos) color = cv::Scalar(0, 255, 255);
                    if (rb_state != RbState::NONE)                        color = cv::Scalar(0, 165, 255);

                    for (size_t i = 0; i + 1 < pts.size(); ++i)
                        cv::line(overlay, pts[i], pts[i + 1], color, 2);
                    if (!pts.empty()) {
                        cv::circle(overlay, pts.front(), 4, cv::Scalar(255, 0, 0), -1);
                        cv::circle(overlay, pts.back(),  4, cv::Scalar(0, 255, 255), -1);
                    }
                }
            }

            for (const auto& d : lastDets) {
                cv::rectangle(overlay, cv::Rect(d.bbox), cv::Scalar(0, 255, 255), 2);
                char label[64];
                std::snprintf(label, sizeof(label), "%s %.2f",
                              ucr::config::SignClassName(d.class_id), d.confidence);
                cv::putText(overlay, label,
                            {int(d.bbox.x), std::max(12, int(d.bbox.y) - 5)},
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
            }

            char text1[192], text2[192];
            std::snprintf(text1, sizeof(text1),
                "FPS %.1f | spd %.1f/%.1f | PID %.1f | out %d | kf %.2f%s",
                fps, applied_speed, out.speed, out.angle_pre_gain,
                (int)std::lround(out.angle), out.kept_frac,
                (!out.valid && inferOK) ? " INV" : "");
            std::snprintf(text2, sizeof(text2),
                "sign %s | %s | near %.1f far %.1f cov %.3f",
                ucr::config::SignClassName(voter.GetCommitted()),
                out.state_str.c_str(),
                out.near_e, out.far_e, out.coverage);
            cv::putText(overlay, text1, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.55,
                        cv::Scalar(0, 255, 0), 2);
            cv::putText(overlay, text2, {10, 55}, cv::FONT_HERSHEY_SIMPLEX, 0.55,
                        cv::Scalar(0, 255, 0), 2);

            cv::imshow("Lane Follower", overlay);
            const int key = cv::waitKey(1) & 0xFF;
            if (key == 27 || key == 'q') break;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    motor.SetSpeedCms(0.0f);
    servo.SetAngle(0);
    if (ucr::config::SHOW_UI) cv::destroyAllWindows();
    std::cout << "\n[System] Stopped safely\n";
    return 0;
}
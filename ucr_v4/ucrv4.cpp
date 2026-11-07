// ============================================================
// ucrv4.cpp  (was main.cpp / test_run.cpp)
//
// UCR 2026 — road following + turn manoeuvres on the Jetson Nano car.
//
// Pipeline:
//   Camera (BGR 640x384)
//     → RoadSegmenter (TwinLiteNet) → binary mask
//     → v20 PID (LaneFollower)
//     → TurnStateMachine (overrides PID during turns)
//     → real-angle → servo → motor
//
// SignDetector runs on the control tick (SIGN_DETECT_EVERY).
// BNO055 read on every control tick; heading is fed to the SM.
// Sign voter commits only after persistent evidence; SM has hysteresis.
// ============================================================
#include "hardware/Camera.h"
#include "hardware/Motor.h"
#include "hardware/Servo.h"
#include "common/Logger.h"
#include "control/LaneFollowerV20.h"
#include "control/SignVoter.h"
#include "control/TurnStateMachine.h"
#include "perception/RoadSegmenter.h"
#include "perception/SignDetector.h"
#include "sensor/BNO055.h"
#include "config/RobotConfig.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <iostream>
#include <thread>
#include <vector>

// ============================================================
// SIGNAL HANDLING
// ============================================================
volatile std::sig_atomic_t g_stopSignal = 0;
static void SignalHandler(int) { g_stopSignal = 1; }

// ============================================================
// HELPERS
// ============================================================
// Count nonzero pixels in the 35%..65% row band — matches v28 mask_mid_pixels().
static int MaskMidPixels(const cv::Mat& mask) {
    if (mask.empty()) return 0;
    const int h  = mask.rows;
    const int y0 = static_cast<int>(h * 0.35f);
    const int y1 = static_cast<int>(h * 0.65f);
    if (y1 <= y0) return 0;
    return cv::countNonZero(mask.rowRange(y0, y1));
}

static inline float WrapDeg360(float d) {
    d = std::fmod(d, 360.0f);
    if (d < 0.0f) d += 360.0f;
    return d;
}

// ============================================================
// MAIN
// ============================================================
int main() {
    std::signal(SIGINT,  SignalHandler);
    std::signal(SIGTERM, SignalHandler);

    // ---------------- Hardware ----------------
    Camera camera;
    if (!camera.Init()) { std::cerr << "Camera init failed\n"; return -1; }

    Motor motor;
    if (!motor.Init()) { std::cerr << "Motor init failed\n"; return -1; }
    motor.SetUartLogging(false);

    Servo servo;
    if (!servo.Init() || !servo.SetPWMFreq(50)) {
        std::cerr << "Servo init failed\n";
        motor.SetSpeedCms(0.0f);
        return -1;
    }
    motor.SetSpeedCms(0.0f);
    servo.SetAngle(0);

    // ---------------- Road perception ----------------
    ucr::perception::RoadSegmenter road;
    ucr::perception::RoadSegmenter::Config roadCfg;
    roadCfg.daChannel        = ucr::config::ROAD_DA_CHANNEL;
    roadCfg.confidenceMargin = ucr::config::ROAD_CONF_MARGIN;
    if (!road.Init(ucr::config::ROAD_ENGINE_PATH, roadCfg)) {
        std::cerr << "Road model init failed\n";
        return -1;
    }

    // ---------------- BNO055 ----------------
    BNO055 bno;
    const bool bno_ok = bno.initialize(std::chrono::milliseconds(1500));
    if (!bno_ok) {
        std::cerr << "BNO055 init failed: " << bno.lastError() << "\n"
                  << "  Turn state machine will be DISABLED.\n";
    }

    // ---------------- Sign detector ----------------
    ucr::perception::SignDetector sign_det;
    ucr::perception::SignDetector::Config signCfg;
    signCfg.conf_thresh = ucr::config::SIGN_CONF_THRESH;
    signCfg.iou_thresh  = ucr::config::SIGN_IOU_THRESH;
    signCfg.input_size  = ucr::config::SIGN_INPUT_SIZE;
    const bool sign_ok = sign_det.Init(ucr::config::SIGN_ENGINE_PATH, signCfg);
    if (!sign_ok) {
        std::cerr << "Sign detector init failed: "
                  << ucr::config::SIGN_ENGINE_PATH << "\n"
                  << "  Turn state machine will be DISABLED.\n";
    }

    // ---------------- Voter + SM ----------------
    ucr::control::SignVoter sign_voter(
        ucr::config::SIGN_VOTER_MIN_FRAMES,
        ucr::config::SIGN_VOTER_MAX_GAP,
        ucr::config::SIGN_VOTER_TIMEOUT,
        ucr::config::SIGN_VOTER_BLACKLIST_FRAMES);
    ucr::control::TurnStateMachine turn_sm(sign_voter);

    const bool turn_features_enabled = bno_ok && sign_ok;

    // ---------------- Control ----------------
    v20::Config cfg;
    cfg.max_speed          = ucr::config::MAX_SPEED_CMS;
    cfg.min_speed          = ucr::config::MIN_SPEED_CMS;
    cfg.curve_scale        = ucr::config::CURVE_SCALE;
    cfg.post_pid_gain      = ucr::config::POST_PID_GAIN;
    cfg.min_rows           = ucr::config::MIN_ROWS;
    cfg.coverage_threshold = ucr::config::COVERAGE_THRESHOLD;
    v20::LaneFollower follower(cfg);

    // ---------------- Runtime state ----------------
    cv::Mat frame, small_mask;
    v20::LaneFollower::Output out;
    double fps        = 0.0;
    int    frameCount = 0;
    auto   tPrev      = std::chrono::high_resolution_clock::now();

    auto    next_control  = std::chrono::steady_clock::now();
    float   applied_speed = 0.0f;
    uint32_t ctrl_tick    = 0;

    BNO055::Data imu_data;
    float        last_heading = 0.0f;
    auto         tPrevIMU     = std::chrono::steady_clock::now();

    std::vector<ucr::perception::SignDetection> detections;
    detections.reserve(8);

    std::cout << "\n=== UCR2026: Auto Run (turn-enabled=" 
              << (turn_features_enabled ? "yes" : "NO") << ") ===\n"
              << "Road engine : " << ucr::config::ROAD_ENGINE_PATH << "\n"
              << "Sign engine : " << ucr::config::SIGN_ENGINE_PATH << "\n"
              << "Control     : " << ucr::config::CONTROL_PERIOD_MS << " ms\n"
              << "Press 'q' to quit\n\n";

    // =========================================================
    // MAIN LOOP
    // =========================================================
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

        ucr::control::TurnStateMachine::Output turn_out;  // all-NaN default

        if (!inferOK) {
            motor.SetSpeedCms(0.0f);
            servo.SetAngle(0);
            follower.resetTracking();
            applied_speed = 0.0f;
        }
        else if (now >= next_control) {
            ++ctrl_tick;

            // ---- Print mask resolution once ----
            if (ctrl_tick == 1) {
                std::cout << "[main] mask resolution: "
                          << small_mask.cols << "x" << small_mask.rows
                          << " (TURN_BASELINE_FLOOR was tuned for 320x192)\n";
            }

            // ---- 1) BNO055 read ----
            float heading = last_heading;
            if (bno_ok) {
                if (bno.read(imu_data)) {
                    // keep relativeYawDeg current in case HEADING_SOURCE switches later
                    const auto   nowIMU = std::chrono::steady_clock::now();
                    const double dtIMU  = std::chrono::duration<double>(
                                              nowIMU - tPrevIMU).count();
                    tPrevIMU = nowIMU;
                    if (dtIMU > 0.001 && dtIMU < 1.0) {
                        bno.integrateYawRate(dtIMU);
                    }
                    // Fused yaw. Bench-verify sign per config note.
                    heading = WrapDeg360(
                        ucr::config::HEADING_SIGN
                            * static_cast<float>(imu_data.eulerDeg.x)
                        + ucr::config::HEADING_OFFSET_DEG);
                    last_heading = heading;
                } else {
                    // I2C glitch — keep last heading this tick, warn occasionally
                    if ((ctrl_tick % 50) == 0) {
                        std::cerr << "[main] BNO055 read failed: "
                                  << bno.lastError() << "\n";
                    }
                }
            }

            // ---- 2) Sign detection (every N control ticks) ----
            if (turn_features_enabled &&
                (ctrl_tick % ucr::config::SIGN_DETECT_EVERY) == 0)
            {
                detections.clear();
                if (sign_det.Detect(frame, detections)) {
                    std::sort(detections.begin(), detections.end(),
                        [](const ucr::perception::SignDetection& a,
                           const ucr::perception::SignDetection& b) {
                            return a.confidence > b.confidence;
                        });
                    std::vector<int> ids;
                    ids.reserve(detections.size());
                    for (const auto& d : detections) ids.push_back(d.class_id);
                    sign_voter.Update(ids);
                }
            }
            const int sign_committed = turn_features_enabled
                                     ? sign_voter.GetCommitted() : -1;

            // ---- 3) px_mid ----
            const int px_mid = MaskMidPixels(small_mask);

            // ---- 4) Step follower (always, so its internal state stays fresh) ----
            out = follower.step(small_mask);

            // ---- 5) Turn SM ----
            if (turn_features_enabled) {
                turn_out = turn_sm.Update(sign_committed, heading, px_mid,
                                          out.valid);
            }

            // ---- 6) Blend follower + turn SM ----
            float speed             = static_cast<float>(out.speed);
            float desired_real_left;

            if (turn_out.in_turn_ctrl) {
                // Turn angle is in PID/sim convention (positive = right).
                // Existing code maps PID→left-positive by negation.
                desired_real_left = -turn_out.angle_deg;
                speed             = turn_out.speed_cms;
            } else {
                desired_real_left = -static_cast<float>(out.angle);
                if (!std::isnan(turn_out.speed_cms)) {
                    // APPROACH braking cap
                    speed = std::min(speed, turn_out.speed_cms);
                }
            }

            // ---- 7) Bound + convert to servo ----
            desired_real_left = std::clamp(desired_real_left,
                                           -ucr::config::MAX_REAL_ANGLE_DEG,
                                            ucr::config::MAX_REAL_ANGLE_DEG);

            float servo_cmd = ucr::config::RealAngleToServo(desired_real_left);
            servo_cmd = std::clamp(servo_cmd,
                                   -static_cast<float>(ucr::config::MAX_SERVO_CMD),
                                    static_cast<float>(ucr::config::MAX_SERVO_CMD));
            const int final_angle = static_cast<int>(std::lround(servo_cmd));

            // ---- 8) Lost-mask creep (skip when turn controller owns speed) ----
            if (!out.valid && out.coverage > 0.02 && !turn_out.in_turn_ctrl) {
                speed = ucr::config::MIN_SPEED_CMS;
            }

            // ---- 9) Soft-start ----
            const float max_delta = ucr::config::MAX_ACCEL_CMS2 *
                                    (ucr::config::CONTROL_PERIOD_MS / 1000.0f);
            if (speed > applied_speed) {
                applied_speed = std::min(speed, applied_speed + max_delta);
            } else {
                applied_speed = speed;
            }

            motor.SetSpeedCms(applied_speed);
            servo.SetAngle(final_angle);

            // ---- 10) Tick scheduling ----
            next_control += std::chrono::milliseconds(ucr::config::CONTROL_PERIOD_MS);
            if (now - next_control > std::chrono::seconds(1)) {
                next_control = now + std::chrono::milliseconds(ucr::config::CONTROL_PERIOD_MS);
            }
        }

        // ---------------- FPS ----------------
        ++frameCount;
        const auto tNow = std::chrono::high_resolution_clock::now();
        const double dt = std::chrono::duration<double>(tNow - tPrev).count();
        if (dt > 0.5) {
            fps        = frameCount / dt;
            frameCount = 0;
            tPrev      = tNow;
        }

        // ---------------- UI ----------------
        if (ucr::config::SHOW_UI) {
            cv::Mat overlay = frame.clone();

            if (inferOK) {
                cv::Mat ui_mask;
                cv::resize(small_mask, ui_mask, frame.size(), 0, 0, cv::INTER_NEAREST);

                cv::Mat green(frame.size(), frame.type(), cv::Scalar(0, 255, 0));
                cv::Mat blended;
                cv::addWeighted(frame, 0.6, green, 0.4, 0, blended);
                blended.copyTo(overlay, ui_mask);

                if (!out.centerline_pts.empty() &&
                    out.proc_width > 0 && out.proc_height > 0) {
                    const float sx = static_cast<float>(frame.cols) / out.proc_width;
                    const float sy = static_cast<float>(frame.rows) / out.proc_height;
                    std::vector<cv::Point> pts;
                    pts.reserve(out.centerline_pts.size());
                    for (const auto& p : out.centerline_pts)
                        pts.emplace_back(static_cast<int>(p.first  * sx),
                                         static_cast<int>(p.second * sy));
                    for (size_t i = 0; i + 1 < pts.size(); ++i)
                        cv::line(overlay, pts[i], pts[i + 1], cv::Scalar(0, 0, 255), 2);
                    if (!pts.empty()) {
                        cv::circle(overlay, pts.front(), 4, cv::Scalar(255, 0, 0), -1);
                        cv::circle(overlay, pts.back(),  4, cv::Scalar(0, 255, 255), -1);
                    }
                }
            }

            // Sign boxes
            for (const auto& d : detections) {
                cv::rectangle(overlay, d.bbox, cv::Scalar(0, 255, 255), 2);
                char lbl[64];
                std::snprintf(lbl, sizeof(lbl), "%s %.2f",
                              ucr::perception::SignClassName(d.class_id),
                              d.confidence);
                cv::putText(overlay, lbl,
                            cv::Point(static_cast<int>(d.bbox.x),
                                      std::max(0, static_cast<int>(d.bbox.y) - 6)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            cv::Scalar(0, 255, 255), 1);
            }

            const char* sm_state_names[] = {"NORM", "APPR", "TURN"};
            const char* sm_state = sm_state_names[static_cast<int>(turn_sm.GetState())];
            const int   sign_commit = turn_features_enabled
                                    ? sign_voter.GetCommitted() : -1;
            const char* sign_name = (sign_commit >= 0)
                                  ? ucr::perception::SignClassName(sign_commit)
                                  : "-";

            char text[256];
            std::snprintf(text, sizeof(text),
                "FPS %.1f | spd %.1f/%.1f | PID %.1f | out %d | TURN=%s dir=%+d | hdg %.1f | sign=%s",
                fps, applied_speed, out.speed, out.angle_pre_gain,
                static_cast<int>(std::lround(out.angle)),
                sm_state, turn_sm.GetTurnDir(),
                last_heading, sign_name);
            cv::putText(overlay, text, {10, 30}, cv::FONT_HERSHEY_SIMPLEX,
                        0.55, cv::Scalar(0, 255, 0), 2);

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
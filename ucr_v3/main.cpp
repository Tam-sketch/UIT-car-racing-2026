// ============================================================
// main.cpp  (was test_run.cpp)
//
// UCR 2026 — autonomous road following on the Jetson Nano car.
//
// Pipeline:
//   Camera (BGR 640x384)
//     → RoadSegmenter (TwinLiteNet ONNX/TensorRT)
//     → binary mask at model output resolution
//     → LaneFollowerV20 (v20 PID)
//     → motor / servo commands
//
// Control loop runs at a FIXED rate (CONTROL_PERIOD_MS).
// Inference runs as fast as the hardware allows; the follower
// only ticks on the fixed boundary. This preserves the meaning
// of v20's per-step rate limits and EMA alphas.
// ============================================================
#include "hardware/Camera.h"
#include "hardware/Motor.h"
#include "hardware/Servo.h"
#include "common/Logger.h"
#include "control/LaneFollowerV20.h"
#include "perception/RoadSegmenter.h"
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
// MAIN
// ============================================================
int main() {
    std::signal(SIGINT,  SignalHandler);
    std::signal(SIGTERM, SignalHandler);

    // --- Hardware ---
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

    // --- Perception ---
    ucr::perception::RoadSegmenter road;
    ucr::perception::RoadSegmenter::Config roadCfg;
    roadCfg.daChannel        = ucr::config::ROAD_DA_CHANNEL;
    roadCfg.confidenceMargin = ucr::config::ROAD_CONF_MARGIN;
    if (!road.Init(ucr::config::ROAD_ENGINE_PATH, roadCfg)) {
        std::cerr << "Road model init failed\n";
        return -1;
    }

    // --- Control ---
    v20::Config cfg;
    cfg.max_speed          = ucr::config::MAX_SPEED_CMS;
    cfg.min_speed          = ucr::config::MIN_SPEED_CMS;
    cfg.curve_scale        = ucr::config::CURVE_SCALE;
    cfg.post_pid_gain      = ucr::config::POST_PID_GAIN;
    cfg.min_rows           = ucr::config::MIN_ROWS;
    cfg.coverage_threshold = ucr::config::COVERAGE_THRESHOLD;
    v20::LaneFollower follower(cfg);

    // --- Runtime state ---
    cv::Mat frame, small_mask;
    v20::LaneFollower::Output out;
    double fps = 0.0;
    int    frameCount = 0;
    auto   tPrev = std::chrono::high_resolution_clock::now();

    auto  next_control   = std::chrono::steady_clock::now();
    float applied_speed  = 0.0f;

    std::cout << "\n=== UCR2026: Auto Run ===\n"
              << "Engine: " << ucr::config::ROAD_ENGINE_PATH << "\n"
              << "Control period: " << ucr::config::CONTROL_PERIOD_MS << " ms\n"
              << "Press 'q' to quit\n\n";

    // =========================================================
    // MAIN LOOP
    // =========================================================
    while (!g_stopSignal) {
        camera.GetFrame(frame);

        // Camera failure → stop and abort this iteration.
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
            // TensorRT failure (exception, shape mismatch, ...).
            // Current behavior (kept per user request): stop + reset.
            motor.SetSpeedCms(0.0f);
            servo.SetAngle(0);
            follower.resetTracking();
            applied_speed = 0.0f;
        }
        else if (now >= next_control) {
            // --- Fixed-rate PID tick ---
            out = follower.step(small_mask);

            // Speed
            float speed = static_cast<float>(out.speed);

            // Angle (out.angle already includes post_pid_gain)
            int final_angle = static_cast<int>(std::lround(out.angle));
            final_angle = std::clamp(final_angle,
                                     -ucr::config::MAX_STEER_DEG,
                                      ucr::config::MAX_STEER_DEG);
            if (ucr::config::INVERT_STEER) final_angle = -final_angle;

            // Lost-mask creep (kept per user request)
            if (!out.valid && out.coverage > 0.02) {
                speed = ucr::config::MIN_SPEED_CMS;
            }

            // Soft-start: ramp up, snap down.
            const float max_delta = ucr::config::MAX_ACCEL_CMS2 *
                                    (ucr::config::CONTROL_PERIOD_MS / 1000.0f);
            if (speed > applied_speed)
                applied_speed = std::min(speed, applied_speed + max_delta);
            else
                applied_speed = speed;

            motor.SetSpeedCms(applied_speed);
            servo.SetAngle(final_angle);

            // Advance tick; resync if we fall too far behind.
            next_control += std::chrono::milliseconds(ucr::config::CONTROL_PERIOD_MS);
            if (now - next_control > std::chrono::seconds(1)) {
                next_control = now + std::chrono::milliseconds(ucr::config::CONTROL_PERIOD_MS);
            }
        }

        // --- FPS ---
        ++frameCount;
        auto tNow = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(tNow - tPrev).count();
        if (dt > 0.5) {
            fps = frameCount / dt;
            frameCount = 0;
            tPrev = tNow;
        }

        // =====================================================
        // UI
        // =====================================================
        if (ucr::config::SHOW_UI) {
            cv::Mat overlay = frame.clone();

            if (inferOK) {
                // Mask (small → camera size for display only)
                cv::Mat ui_mask;
                cv::resize(small_mask, ui_mask, frame.size(), 0, 0, cv::INTER_NEAREST);

                cv::Mat green(frame.size(), frame.type(), cv::Scalar(0, 255, 0));
                cv::Mat blended;
                cv::addWeighted(frame, 0.6, green, 0.4, 0, blended);
                blended.copyTo(overlay, ui_mask);

                // Centerline
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

            char text[192];
            std::snprintf(text, sizeof(text),
                "FPS %.1f | spd %.1f/%.1f | PID %.1f | out %d | kf %.2f%s",
                fps, applied_speed, out.speed, out.angle_pre_gain,
                static_cast<int>(std::lround(out.angle)), out.kept_frac,
                (!out.valid && inferOK) ? " INV" : "");
            cv::putText(overlay, text, {10, 30}, cv::FONT_HERSHEY_SIMPLEX,
                        0.6, cv::Scalar(0, 255, 0), 2);

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
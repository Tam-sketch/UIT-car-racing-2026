// ============================================================
// main.cpp — UCR 2026 (Path-based V24)
// ============================================================
#include "hardware/Camera.h"
#include "hardware/Motor.h"
#include "hardware/Servo.h"
#include "sensor/BNO055.h"
#include "common/Logger.h"
#include "control/LaneFollowerV24.h" // <-- SỬA Ở ĐÂY
#include "control/SignVoter.h"
#include "perception/RoadSegmenter.h"
#include "perception/SignDetector.h"
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

volatile std::sig_atomic_t g_stopSignal = 0;
static void SignalHandler(int) { g_stopSignal = 1; }

static inline float WrapDeg360(float d) {
    d = std::fmod(d, 360.0f);
    if (d < 0.0f) d += 360.0f;
    return d;
}

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
        std::cerr << "Servo init failed\n"; motor.SetSpeedCms(0.0f); return -1;
    }
    motor.SetSpeedCms(0.0f); servo.SetAngle(0);

    // --- Road perception ---
    ucr::perception::RoadSegmenter road;
    ucr::perception::RoadSegmenter::Config roadCfg;
    roadCfg.daChannel        = ucr::config::ROAD_DA_CHANNEL;
    roadCfg.confidenceMargin = ucr::config::ROAD_CONF_MARGIN;
    if (!road.Init(ucr::config::ROAD_ENGINE_PATH, roadCfg)) {
        std::cerr << "Road model init failed\n"; return -1;
    }

    // --- Sign perception ---
    ucr::perception::SignDetector signDet;
    ucr::perception::SignDetector::Config signCfg;
    signCfg.conf_thresh = ucr::config::SIGN_CONF_THRESH;
    signCfg.iou_thresh  = ucr::config::SIGN_IOU_THRESH;
    signCfg.input_size  = ucr::config::SIGN_INPUT_SIZE;
    const bool signOk = signDet.Init(ucr::config::SIGN_ENGINE_PATH, signCfg);
    if (!signOk) {
        std::cerr << "[main] Sign detector init failed; turns DISABLED\n";
    }

    // --- BNO055 (chỉ để log heading) ---
    BNO055 bno;
    const bool bnoOk = bno.initialize(std::chrono::milliseconds(1500));
    if (!bnoOk) std::cerr << "[main] BNO055 init failed (OK, not needed)\n";

    // --- Control (SỬA SANG V24) ---
    v24::Config cfg;
    cfg.max_speed          = ucr::config::MAX_SPEED_CMS;
    cfg.min_speed          = ucr::config::MIN_SPEED_CMS;
    cfg.curve_scale        = ucr::config::CURVE_SCALE;
    cfg.post_pid_gain      = ucr::config::POST_PID_GAIN; // 2.2
    cfg.min_rows           = ucr::config::MIN_ROWS;
    cfg.coverage_threshold = ucr::config::COVERAGE_THRESHOLD;
    v24::LaneFollower follower(cfg);

    // --- Sign voter ---
    ucr::control::SignVoter voter(
        ucr::config::SIGN_VOTER_ACTIVATE_FRAMES,
        ucr::config::SIGN_VOTER_TIMEOUT_FRAMES);

    // --- State ---
    cv::Mat frame, small_mask;
    v24::LaneFollower::Output out;
    std::vector<ucr::perception::SignDetector::Detection> detections;
    std::vector<int> detectedIds;

    uint32_t ctrlTick = 0;
    float    appliedSpeed = 0.0f;
    float    lastHeading = 0.0f;
    BNO055::Data imu;
    double   fps = 0.0;
    int      frameCount = 0;
    auto     tPrev = std::chrono::high_resolution_clock::now();
    auto     next_control = std::chrono::steady_clock::now();
    auto     tPrevIMU = std::chrono::steady_clock::now();

    std::cout << "\n=== UCR2026 Auto Run (V24 Combined) ===\n";

    while (!g_stopSignal) {
        camera.GetFrame(frame);
        if (frame.empty()) {
            motor.SetSpeedCms(0.0f); servo.SetAngle(0);
            appliedSpeed = 0.0f;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        const bool inferOK = road.Predict(frame, small_mask);
        const auto now = std::chrono::steady_clock::now();

        if (!inferOK) {
            motor.SetSpeedCms(0.0f); servo.SetAngle(0);
            follower.resetTracking();
            appliedSpeed = 0.0f;
        } else if (now >= next_control) {
            ++ctrlTick;

            // BNO055
            if (bnoOk) {
                if (bno.read(imu)) {
                    const auto nowIMU = std::chrono::steady_clock::now();
                    const double dtIMU = std::chrono::duration<double>(
                        nowIMU - tPrevIMU).count();
                    tPrevIMU = nowIMU;
                    if (dtIMU > 0.001 && dtIMU < 1.0)
                        bno.integrateYawRate(dtIMU);
                    lastHeading = WrapDeg360(
                        ucr::config::HEADING_SIGN * (float)imu.eulerDeg.x +
                        ucr::config::HEADING_OFFSET_DEG);
                }
            }

            // Sign detect
            const bool detectTick = (ctrlTick % ucr::config::SIGN_DETECT_EVERY) == 0;
            if (signOk && detectTick) {
                detections.clear(); detectedIds.clear();
                if (signDet.Detect(frame, detections)) {
                    std::sort(detections.begin(), detections.end(),
                        [](const auto& a, const auto& b) {
                            return a.confidence > b.confidence;
                        });
                    for (const auto& d : detections)
                        detectedIds.push_back(d.class_id);
                }
                voter.Update(detectedIds, ctrlTick);
                
                // Set Sign cho V24
                const int cid = voter.GetCommitted();
                if (cid >= 0) {
                    follower.setSign(ucr::config::SignClassName(cid), 150); // 150 là trigger_y
                } else {
                    follower.setSign("", 150);
                }
            }

            // PID + Bézier
            out = follower.step(small_mask);

            // Speed + steering
            float desired_real_left = -static_cast<float>(out.angle);
            float speed = out.speed;

            desired_real_left = std::clamp(desired_real_left,
                -ucr::config::MAX_REAL_ANGLE_DEG,
                 ucr::config::MAX_REAL_ANGLE_DEG);
            float servo_cmd = ucr::config::RealAngleToServo(desired_real_left);
            servo_cmd = std::clamp(servo_cmd,
                -static_cast<float>(ucr::config::MAX_SERVO_CMD),
                 static_cast<float>(ucr::config::MAX_SERVO_CMD));
            int final_angle = static_cast<int>(std::lround(servo_cmd));

            // Soft-start
            const float maxDelta = ucr::config::MAX_ACCEL_CMS2 *
                (ucr::config::CONTROL_PERIOD_MS / 1000.0f);
            if (speed > appliedSpeed)
                appliedSpeed = std::min(speed, appliedSpeed + maxDelta);
            else
                appliedSpeed = speed;

            motor.SetSpeedCms(appliedSpeed);
            servo.SetAngle(final_angle);

            // Schedule
            next_control += std::chrono::milliseconds(
                ucr::config::CONTROL_PERIOD_MS);
            if (now - next_control > std::chrono::seconds(1))
                next_control = now + std::chrono::milliseconds(
                    ucr::config::CONTROL_PERIOD_MS);
        }

        // FPS
        ++frameCount;
        auto tNow = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(tNow - tPrev).count();
        if (dt > 0.5) { fps = frameCount / dt; frameCount = 0; tPrev = tNow; }

        // UI
        if (ucr::config::SHOW_UI) {
            cv::Mat overlay = frame.clone();
            if (inferOK) {
                cv::Mat ui_mask;
                cv::resize(small_mask, ui_mask, frame.size(), 0, 0, cv::INTER_NEAREST);
                cv::Mat green(frame.size(), frame.type(), cv::Scalar(0, 255, 0));
                cv::Mat blended;
                cv::addWeighted(frame, 0.6, green, 0.4, 0, blended);
                blended.copyTo(overlay, ui_mask);
                
                if (!out.centerline_pts.empty() && out.proc_width > 0) {
                    const float sx = (float)frame.cols / out.proc_width;
                    const float sy = (float)frame.rows / out.proc_height;
                    std::vector<cv::Point> pts;
                    for (const auto& p : out.centerline_pts)
                        pts.emplace_back((int)(p.first * sx), (int)(p.second * sy));
                    
                    // Đổi màu Magenta khi đang Turn
                    cv::Scalar color = (out.state_str != "CENTERLINE") 
                        ? cv::Scalar(255, 0, 255) : cv::Scalar(0, 0, 255);
                        
                    for (size_t i = 0; i + 1 < pts.size(); ++i)
                        cv::line(overlay, pts[i], pts[i + 1], color, 2);
                }
            }
            for (const auto& d : detections) {
                cv::rectangle(overlay, d.bbox, cv::Scalar(0, 255, 255), 2);
                char lbl[64];
                std::snprintf(lbl, sizeof(lbl), "%s %.2f",
                              ucr::config::SignClassName(d.class_id), d.confidence);
                cv::putText(overlay, lbl, cv::Point((int)d.bbox.x, std::max(0, (int)d.bbox.y - 6)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
            }
            
            char text[256];
            std::snprintf(text, sizeof(text),
                "FPS %.1f | spd %.1f/%.1f | PID %.1f | sign=%s | state=%s | hdg %.1f",
                fps, appliedSpeed, out.speed, out.angle_pre_gain,
                voter.GetCommitted() >= 0 ? ucr::config::SignClassName(voter.GetCommitted()) : "-",
                out.state_str.c_str(), lastHeading);
            cv::putText(overlay, text, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            
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
// ============================================================
// main.cpp  (ucrv5)
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

// Thư viện cảm biến
#include "sensor/TOF.h"
#include "sensor/BNO055.h"

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
        case PARKING:    s.active = true; s.side = 0;  break; // Thêm Parking
        default:         s.active = false; s.side = 0; break;
    }
    return s;
}

enum class LightColor { NONE, RED };

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
    cv::Mat roi_bgr = bgr(cv::Rect(x1, y1, x2 - x1, y2 - y1));

    int nRed = 0;
    int nGrnYel = 0;

    for (int y = 0; y < hsv.rows; ++y) {
        const uchar* hsv_p = hsv.ptr<uchar>(y);
        const uchar* bgr_p = roi_bgr.ptr<uchar>(y);
        for (int x = 0; x < hsv.cols; ++x) {
            const uchar H = hsv_p[x*3+0];
            const uchar S = hsv_p[x*3+1];
            const uchar V = hsv_p[x*3+2];
            const uchar B = bgr_p[x*3+0];
            const uchar G = bgr_p[x*3+1];
            const uchar R = bgr_p[x*3+2];

            if (S >= S_MIN && V >= V_MIN) {
                if ((H >= RED_H1_LO && H <= RED_H1_HI) || (H >= RED_H2_LO && H <= RED_H2_HI)) ++nRed;
                else if ((H >= YEL_H_LO && H <= YEL_H_HI) || (H >= GRN_H_LO && H <= GRN_H_HI)) ++nGrnYel;
            }
            else if (V >= 220 && S < 50) {
                if (R > G + 10 && R > B + 10 && R >= 200) ++nRed;
                else if (G > R + 10 && G > B + 10 && G >= 200) ++nGrnYel;
            }
        }
    }
    if (nRed >= MIN_LIT_PIXELS && nRed > nGrnYel) return LightColor::RED;
    return LightColor::NONE;
}

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
        std::cerr << "Servo init failed\n";
        motor.SetSpeedCms(0.0f);
        return -1;
    }
    motor.SetSpeedCms(0.0f);
    servo.SetAngle(0);

    // --- Khởi tạo TOF và BNO055 ---
    sensor::TOF tof;
    if (!tof.Init("/dev/i2c-1")) { // Sửa lại port I2C của bạn
        std::cerr << "[Main] TOF init failed\n";
    }
    
    BNO055 bno;
    if (!bno.initialize(std::chrono::milliseconds(1000))) {
        std::cerr << "[Main] BNO055 init failed\n";
    }

    sensor::TOFReadings tof_readings;
    BNO055::Data bno_data;

    // --- Road segmentation ---
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
    if (!signReady) {
        std::cerr << "[Main] Sign detector init failed — turns DISABLED\n";
    }

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
    
    cfg.turn_min_ticks     = ucr::config::TURN_MIN_TICKS;
    cfg.turn_max_ticks     = ucr::config::TURN_MAX_TICKS;
    cfg.turn_straight_ticks= ucr::config::TURN_STRAIGHT_TICKS;
    cfg.turn_synth_width_px= ucr::config::TURN_SYNTH_WIDTH_PX;
    cfg.turn_lookahead_y_ratio = ucr::config::TURN_LOOKAHEAD_Y_RATIO;
    cfg.turn_near_y_ratio  = ucr::config::TURN_NEAR_Y_RATIO;
    cfg.turn_far_weight    = ucr::config::TURN_FAR_WEIGHT;
    cfg.turn_ramp_ticks    = ucr::config::TURN_RAMP_TICKS;
    cfg.turn_error_cap     = ucr::config::TURN_ERROR_CAP;

    v21::LaneFollower follower(cfg);

    cv::Mat frame, small_mask;
    v21::LaneFollower::Output out;
    using SignDetection = ucr::perception::SignDetector::Detection;
    std::vector<SignDetection> lastDets;

    double fps = 0.0;
    int    frameCount = 0;
    auto   tPrev = std::chrono::high_resolution_clock::now();

    auto    next_control  = std::chrono::steady_clock::now();
    float   applied_speed = 0.0f;
    uint32_t tickIdx      = 0;
    int lastPersonSeenTick = -1;
    int lastRedSeenTick    = -1;

    std::cout << "\n=== UCR2026: Auto Run ===\nPress 'q' to quit\n\n";

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
            // 1. Sign detection
            if (signReady && (tickIdx % ucr::config::SIGN_DETECT_EVERY) == 0) {
                sign.Detect(frame, lastDets);

                using namespace ucr::config::sign_class;
                std::vector<int> classIds;
                classIds.reserve(lastDets.size());

                for (auto& d : lastDets) {
                    if (d.confidence < ucr::config::SIGN_CONF_THRESH) continue;

                    if (d.class_id == WALKING_PERSON) {
                        lastPersonSeenTick = static_cast<int>(tickIdx);
                        continue;
                    }
                    if (d.class_id == TRAFFIC_LIGHT) {
                        if (ClassifyTrafficLight(frame, d.bbox) == LightColor::RED) {
                            lastRedSeenTick = static_cast<int>(tickIdx);
                        }
                        continue;
                    }

                    int cls = (d.class_id == CONSTRUCTION_AHEAD) ? TURN_LEFT : d.class_id;

                    if (cls == TURN_LEFT || cls == TURN_RIGHT ||
                        cls == NO_LEFT   || cls == NO_RIGHT ||
                        cls == PARKING) { // Thêm Parking vào voter
                        classIds.push_back(cls);
                    }
                }
                voter.Update(classIds, tickIdx);
            }

            // 2. Build SignState
            v21::SignState signState = MakeSignState(voter.GetCommitted());

            // 3. Đọc cảm biến TOF và BNO055
            float tof_left = -1.0f;
            float tof_right = -1.0f;
            if (tof.Read(tof_readings, std::chrono::milliseconds(10))) {
                // TODO: Kiểm tra lại index [2] và [3] cho sau trái/phải
                if (tof_readings[2].valid) tof_left  = tof_readings[2].distanceMm;
                if (tof_readings[3].valid) tof_right = tof_readings[3].distanceMm;
            }

            float current_heading = 0.0f;
            if (bno.read(bno_data)) {
                current_heading = bno_data.eulerDeg.x; 
            }

            // 4. Control tick
            out = follower.step(small_mask, signState, tof_left, tof_right, current_heading);

            // 5. Steering
            float desired_real_left = -static_cast<float>(out.angle);
            desired_real_left = std::clamp(desired_real_left,
                                           -ucr::config::MAX_REAL_ANGLE_DEG,
                                            ucr::config::MAX_REAL_ANGLE_DEG);

            float servo_cmd = ucr::config::RealAngleToServo(desired_real_left);
            servo_cmd = std::clamp(servo_cmd,
                                   -static_cast<float>(ucr::config::MAX_SERVO_CMD),
                                    static_cast<float>(ucr::config::MAX_SERVO_CMD));

            const int final_angle = static_cast<int>(std::lround(servo_cmd));

            // 6. Speed
            float speed = static_cast<float>(out.speed);
            if (!out.valid && out.coverage > 0.02) speed = ucr::config::MIN_SPEED_CMS;

            const bool stopForPerson = (lastPersonSeenTick >= 0) &&
                (static_cast<int>(tickIdx) - lastPersonSeenTick <= ucr::config::PERSON_STOP_HANGOVER_TICKS);

            const bool stopForRedLight = (lastRedSeenTick >= 0) &&
                (static_cast<int>(tickIdx) - lastRedSeenTick <= ucr::config::RED_LIGHT_HANGOVER_TICKS);

            if (stopForPerson || stopForRedLight) speed = 0.0f;

            const float max_delta = ucr::config::MAX_ACCEL_CMS2 * (ucr::config::CONTROL_PERIOD_MS / 1000.0f);
            if (speed > applied_speed) applied_speed = std::min(speed, applied_speed + max_delta);
            else                       applied_speed = speed;

            motor.SetSpeedCms(applied_speed);
            servo.SetAngle(final_angle);

            // 7. Advance tick
            ++tickIdx;
            next_control += std::chrono::milliseconds(ucr::config::CONTROL_PERIOD_MS);
            if (now - next_control > std::chrono::seconds(1)) {
                next_control = now + std::chrono::milliseconds(ucr::config::CONTROL_PERIOD_MS);
            }

            // 8. Log
            static auto lastLog = std::chrono::steady_clock::now();
            if (now - lastLog >= std::chrono::seconds(1)) {
                std::printf("[t=%u] sign=%-11s(%d) state=%-20s spd=%.1f out=%.1f\n",
                    tickIdx, ucr::config::SignClassName(voter.GetCommitted()),
                    voter.GetCommitted(), out.state_str.c_str(), applied_speed, out.angle);
                lastLog = now;
            }
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

                if (!out.final_path.empty() && out.proc_width > 0 && out.proc_height > 0) {
                    const float sx = float(frame.cols) / out.proc_width;
                    const float sy = float(frame.rows) / out.proc_height;
                    std::vector<cv::Point> pts;
                    for (const auto& p : out.final_path) pts.emplace_back(int(p.x * sx), int(p.y * sy));

                    cv::Scalar color(0, 0, 255);
                    if (out.state_str.find("TURN") != std::string::npos) color = cv::Scalar(255, 0, 255);
                    if (out.state_str.find("PARK") != std::string::npos) color = cv::Scalar(255, 255, 0); // Cyan for parking

                    for (size_t i = 0; i + 1 < pts.size(); ++i) cv::line(overlay, pts[i], pts[i + 1], color, 2);
                }
            }

            for (const auto& d : lastDets) {
                cv::rectangle(overlay, cv::Rect(d.bbox), cv::Scalar(0, 255, 255), 2);
                cv::putText(overlay, ucr::config::SignClassName(d.class_id),
                            {int(d.bbox.x), std::max(12, int(d.bbox.y) - 5)},
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
            }

            char text1[192];
            std::snprintf(text1, sizeof(text1),
                "FPS %.1f | spd %.1f | sign %s | %s",
                fps, applied_speed, ucr::config::SignClassName(voter.GetCommitted()), out.state_str.c_str());
            cv::putText(overlay, text1, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 2);

            cv::imshow("Lane Follower", overlay);
            if ((cv::waitKey(1) & 0xFF) == 'q') break;
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
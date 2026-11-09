// ============================================================
// main.cpp  (ucrv5)
//
// UCR 2026 — autonomous road following with sign-triggered
// quadratic-bezier turns.
//
// Pipeline:
//   Camera (BGR 640x384)
//     → RoadSegmenter (TwinLiteNet TensorRT)
//     → binary mask at model output resolution
//     → SignDetector (YOLO 416x416, 8-class)  [throttled]
//     → SignVoter (temporal commit)
//     → v21::LaneFollower (V20 PID + V21 opening detector +
//                          quadratic bezier turn path)
//     → motor / servo commands
//
// The control loop ticks at a fixed rate (CONTROL_PERIOD_MS=100).
// Sign inference runs every SIGN_DETECT_EVERY control ticks.
// The voter's timeout/activate counters are in control-tick units.
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

// ============================================================
// SIGNAL HANDLING
// ============================================================
volatile std::sig_atomic_t g_stopSignal = 0;
static void SignalHandler(int) { g_stopSignal = 1; }

// ============================================================
// SIGN CLASS → v21::SignState
//
// side convention (matches LaneFollowerV21::detectOpenings):
//   -1 = look left, +1 = look right, 2 = both sides, 0 = disabled
//
// active=false disables the whole turn block inside v21::LaneFollower,
// including the ghost-turn branch. Non-turn classes (parking,
// roundabout, traffic_light) are detected but not acted on yet.
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
// TRAFFIC LIGHT: dominant lit color in the bbox ROI
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

    int nRed = 0, nYel = 0, nGrn = 0;
    for (int y = 0; y < hsv.rows; ++y) {
        const uchar* p = hsv.ptr<uchar>(y);
        for (int x = 0; x < hsv.cols; ++x) {
            const uchar H = p[x*3+0], S = p[x*3+1], V = p[x*3+2];
            if (S < S_MIN || V < V_MIN) continue;
            if ((H >= RED_H1_LO && H <= RED_H1_HI) ||
                (H >= RED_H2_LO && H <= RED_H2_HI))   { ++nRed; }
            else if (H >= YEL_H_LO && H <= YEL_H_HI)  { ++nYel; }
            else if (H >= GRN_H_LO && H <= GRN_H_HI)  { ++nGrn; }
        }
    }

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

    // --- Road segmentation ---
    ucr::perception::RoadSegmenter road;
    ucr::perception::RoadSegmenter::Config roadCfg;
    roadCfg.daChannel        = ucr::config::ROAD_DA_CHANNEL;
    roadCfg.confidenceMargin = ucr::config::ROAD_CONF_MARGIN;
    if (!road.Init(ucr::config::ROAD_ENGINE_PATH, roadCfg)) {
        std::cerr << "Road model init failed\n";
        return -1;
    }

    // --- Sign detector (non-fatal on failure — lane following still works) ---
    ucr::perception::SignDetector sign;
    ucr::perception::SignDetector::Config signCfg;
    signCfg.conf_thresh = ucr::config::SIGN_CONF_THRESH;
    signCfg.iou_thresh  = ucr::config::SIGN_IOU_THRESH;
    signCfg.input_size  = ucr::config::SIGN_INPUT_SIZE;
    bool signReady = sign.Init(ucr::config::SIGN_ENGINE_PATH, signCfg);
    if (!signReady) {
        std::cerr << "[Main] Sign detector init failed — turns DISABLED\n";
    }

    // --- Sign voter ---
    ucr::control::SignVoter voter(
        ucr::config::SIGN_VOTER_ACTIVATE_FRAMES,
        ucr::config::SIGN_VOTER_TIMEOUT_FRAMES);

    // --- Lane follower (V21 = V20 + opening detector + bezier turn) ---
    v21::Config cfg;
    cfg.max_speed          = ucr::config::MAX_SPEED_CMS;
    cfg.min_speed          = ucr::config::MIN_SPEED_CMS;
    cfg.curve_scale        = ucr::config::CURVE_SCALE;
    cfg.post_pid_gain      = ucr::config::POST_PID_GAIN;
    cfg.min_rows           = ucr::config::MIN_ROWS;
    cfg.coverage_threshold = ucr::config::COVERAGE_THRESHOLD;
    v21::LaneFollower follower(cfg);

    // --- Runtime state ---
    cv::Mat frame, small_mask;
    v21::LaneFollower::Output out;
    using SignDetection = ucr::perception::SignDetector::Detection;
    std::vector<SignDetection> lastDets;   // retained for UI drawing

    double fps = 0.0;
    int    frameCount = 0;
    auto   tPrev = std::chrono::high_resolution_clock::now();

    auto    next_control  = std::chrono::steady_clock::now();
    float   applied_speed = 0.0f;
    uint32_t tickIdx      = 0;
    int lastPersonSeenTick = -1;
    int lastRedSeenTick    = -1;

    std::cout << "\n=== UCR2026: Auto Run (turns="
              << (signReady ? "yes" : "no") << ") ===\n"
              << "Road engine: " << ucr::config::ROAD_ENGINE_PATH << "\n";
    if (signReady)
        std::cout << "Sign engine: " << ucr::config::SIGN_ENGINE_PATH << "\n";
    std::cout << "Control period: " << ucr::config::CONTROL_PERIOD_MS << " ms\n"
              << "Sign detect every " << ucr::config::SIGN_DETECT_EVERY
              << " ticks\n"
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

        if (!inferOK) {
            motor.SetSpeedCms(0.0f);
            servo.SetAngle(0);
            follower.resetTracking();
            applied_speed = 0.0f;
        }
        else if (now >= next_control) {
            // -------------------------------------------------
            // 1. Sign detection (throttled)
            // -------------------------------------------------
            if (signReady && (tickIdx % ucr::config::SIGN_DETECT_EVERY) == 0) {
                sign.Detect(frame, lastDets);

                using namespace ucr::config::sign_class;
                std::vector<int> classIds;
                classIds.reserve(lastDets.size());

                for (auto& d : lastDets) {
                    if (d.confidence < ucr::config::SIGN_CONF_THRESH) continue;

                    // Walking person → stop signal (bypasses the voter)
                    if (d.class_id == WALKING_PERSON) {
                        lastPersonSeenTick = static_cast<int>(tickIdx);
                        continue;
                    }

                    // Traffic light → HSV color check on the bbox ROI (bypasses the voter)
                    if (d.class_id == TRAFFIC_LIGHT) {
                        if (ClassifyTrafficLight(frame, d.bbox) == LightColor::RED) {
                            lastRedSeenTick = static_cast<int>(tickIdx);
                        }
                        continue;
                    }

                    // Construction ahead is physically a turn-left sign on this map
                    int cls = (d.class_id == CONSTRUCTION_AHEAD) ? TURN_LEFT : d.class_id;

                    // Only turn-related signs feed the voter
                    if (cls == TURN_LEFT || cls == TURN_RIGHT ||
                        cls == NO_LEFT   || cls == NO_RIGHT) {
                        classIds.push_back(cls);
                    }
                }
                voter.Update(classIds, tickIdx);
            }

            // -------------------------------------------------
            // 2. Build the SignState from the voter's committed class
            // -------------------------------------------------
            v21::SignState signState = MakeSignState(voter.GetCommitted());

            // -------------------------------------------------
            // 3. Control tick — v21::LaneFollower does the rest:
            //    • v20 PID for straight/curve
            //    • detectOpenings on the sign's side when active
            //    • buildPathWithTurn → quadratic bezier on turn
            //    • lookahead-based final_error during turn
            // -------------------------------------------------
            out = follower.step(small_mask, signState);

            // -------------------------------------------------
            // 4. Steering: PID (+ = right) → real-angle (+ = left)
            //    → bound → servo cmd via calibration
            // -------------------------------------------------
            float desired_real_left = -static_cast<float>(out.angle);
            desired_real_left = std::clamp(desired_real_left,
                                           -ucr::config::MAX_REAL_ANGLE_DEG,
                                            ucr::config::MAX_REAL_ANGLE_DEG);

            float servo_cmd = ucr::config::RealAngleToServo(desired_real_left);
            servo_cmd = std::clamp(servo_cmd,
                                   -static_cast<float>(ucr::config::MAX_SERVO_CMD),
                                    static_cast<float>(ucr::config::MAX_SERVO_CMD));

            const int final_angle = static_cast<int>(std::lround(servo_cmd));

            // -------------------------------------------------
            // 5. Speed (soft-start + lost-mask creep)
            // -------------------------------------------------
            // --- Speed ---
            float speed = static_cast<float>(out.speed);
            if (!out.valid && out.coverage > 0.02) {
                speed = ucr::config::MIN_SPEED_CMS;
            }

            // --- Stop overrides (highest priority) ---
            const bool stopForPerson =
                (lastPersonSeenTick >= 0) &&
                (static_cast<int>(tickIdx) - lastPersonSeenTick <=
                 ucr::config::PERSON_STOP_HANGOVER_TICKS);

            const bool stopForRedLight =
                (lastRedSeenTick >= 0) &&
                (static_cast<int>(tickIdx) - lastRedSeenTick <=
                 ucr::config::RED_LIGHT_HANGOVER_TICKS);

            const bool stopRequested = stopForPerson || stopForRedLight;
            if (stopRequested) {
                speed = 0.0f;
            }

            // --- Soft-start ramp ---
            const float max_delta = ucr::config::MAX_ACCEL_CMS2 *
                                    (ucr::config::CONTROL_PERIOD_MS / 1000.0f);
            if (speed > applied_speed)
                applied_speed = std::min(speed, applied_speed + max_delta);
            else
                applied_speed = speed;   // immediate decel to 0 when stopped

            motor.SetSpeedCms(applied_speed);
            servo.SetAngle(final_angle);

            // -------------------------------------------------
            // 6. Advance tick
            // -------------------------------------------------
            ++tickIdx;
            next_control += std::chrono::milliseconds(ucr::config::CONTROL_PERIOD_MS);
            if (now - next_control > std::chrono::seconds(1)) {
                next_control = now +
                    std::chrono::milliseconds(ucr::config::CONTROL_PERIOD_MS);
            }

            // -------------------------------------------------
            // 7. Periodic debug log (1 Hz)
            // -------------------------------------------------
            static auto lastLog = std::chrono::steady_clock::now();
            if (now - lastLog >= std::chrono::seconds(1)) {
                const char* stopStr = stopForPerson   ? "STOP:person"
                                    : stopForRedLight ? "STOP:red"
                                    :                   "-";
                std::printf(
                    "[t=%u] sign=%-11s(%d) state=%-20s %-11s "
                    "near=%6.1f far=%6.1f eff=%6.1f cov=%.3f "
                    "spd=%.1f pid=%.1f out=%.1f\n",
                    tickIdx,
                    ucr::config::SignClassName(voter.GetCommitted()),
                    voter.GetCommitted(),
                    out.state_str.c_str(),
                    stopStr,
                    out.near_e, out.far_e, out.eff_error, out.coverage,
                    applied_speed, out.angle_pre_gain, out.angle);
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
                // Mask overlay
                cv::Mat ui_mask;
                cv::resize(small_mask, ui_mask, frame.size(), 0, 0, cv::INTER_NEAREST);
                cv::Mat green(frame.size(), frame.type(), cv::Scalar(0, 255, 0));
                cv::Mat blended;
                cv::addWeighted(frame, 0.6, green, 0.4, 0, blended);
                blended.copyTo(overlay, ui_mask);

                // Path overlay: final_path is the centerline, or the
                // bezier turn path when v21::LaneFollower is turning.
                if (!out.final_path.empty() &&
                    out.proc_width > 0 && out.proc_height > 0) {
                    const float sx = float(frame.cols) / out.proc_width;
                    const float sy = float(frame.rows) / out.proc_height;

                    std::vector<cv::Point> pts;
                    pts.reserve(out.final_path.size());
                    for (const auto& p : out.final_path)
                        pts.emplace_back(int(p.x * sx), int(p.y * sy));

                    cv::Scalar color(0, 0, 255);                     // red = centerline
                    if (out.state_str.find("TURN") != std::string::npos)
                        color = cv::Scalar(255, 0, 255);             // magenta = bezier turn
                    if (out.state_str.find("GHOST") != std::string::npos)
                        color = cv::Scalar(0, 255, 255);             // yellow  = ghost turn

                    for (size_t i = 0; i + 1 < pts.size(); ++i)
                        cv::line(overlay, pts[i], pts[i + 1], color, 2);
                    if (!pts.empty()) {
                        cv::circle(overlay, pts.front(), 4, cv::Scalar(255, 0, 0), -1);
                        cv::circle(overlay, pts.back(),  4, cv::Scalar(0, 255, 255), -1);
                    }
                }
            }

            // Sign detections
            for (const auto& d : lastDets) {
                cv::rectangle(overlay, cv::Rect(d.bbox), cv::Scalar(0, 255, 255), 2);
                char label[64];
                std::snprintf(label, sizeof(label), "%s %.2f",
                              ucr::config::SignClassName(d.class_id), d.confidence);
                cv::putText(overlay, label,
                            {int(d.bbox.x), std::max(12, int(d.bbox.y) - 5)},
                            cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            cv::Scalar(0, 255, 255), 1);
            }

            // HUD
            char text1[192], text2[192];
            std::snprintf(text1, sizeof(text1),
                "FPS %.1f | spd %.1f/%.1f | PID %.1f | out %d | kf %.2f%s",
                fps, applied_speed, out.speed, out.angle_pre_gain,
                static_cast<int>(std::lround(out.angle)), out.kept_frac,
                (!out.valid && inferOK) ? " INV" : "");
            std::snprintf(text2, sizeof(text2),
                "sign %s | %s | near %.1f far %.1f cov %.3f",
                ucr::config::SignClassName(voter.GetCommitted()),
                out.state_str.c_str(),
                out.near_e, out.far_e, out.coverage);

            cv::putText(overlay, text1, {10, 30},
                        cv::FONT_HERSHEY_SIMPLEX, 0.55,
                        cv::Scalar(0, 255, 0), 2);
            cv::putText(overlay, text2, {10, 55},
                        cv::FONT_HERSHEY_SIMPLEX, 0.55,
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
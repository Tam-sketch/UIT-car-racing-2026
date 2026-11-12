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
        case PARKING:    s.active = true; s.side = 0;  break; 
        default:         s.active = false; s.side = 0; break;
    }
    return s;
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
    if (!servo.Init() || !servo.SetPWMFreq(50)) { std::cerr << "Servo init failed\n"; motor.SetSpeedCms(0.0f); return -1; }
    motor.SetSpeedCms(0.0f);
    servo.SetAngle(0);

    sensor::TOF tof;
    if (!tof.Init("/dev/i2c-1")) { std::cerr << "[Main] TOF init failed\n"; }
    BNO055 bno;
    if (!bno.initialize(std::chrono::milliseconds(1000))) { std::cerr << "[Main] BNO055 init failed\n"; }

    sensor::TOFReadings tof_readings;
    BNO055::Data bno_data;

    ucr::perception::RoadSegmenter road;
    ucr::perception::RoadSegmenter::Config roadCfg;
    roadCfg.daChannel = ucr::config::ROAD_DA_CHANNEL; roadCfg.confidenceMargin = ucr::config::ROAD_CONF_MARGIN;
    if (!road.Init(ucr::config::ROAD_ENGINE_PATH, roadCfg)) { std::cerr << "Road model init failed\n"; return -1; }

    ucr::perception::SignDetector sign;
    ucr::perception::SignDetector::Config signCfg;
    signCfg.conf_thresh = ucr::config::SIGN_CONF_THRESH; signCfg.iou_thresh = ucr::config::SIGN_IOU_THRESH; signCfg.input_size = ucr::config::SIGN_INPUT_SIZE;
    bool signReady = sign.Init(ucr::config::SIGN_ENGINE_PATH, signCfg);

    ucr::control::SignVoter voter(ucr::config::SIGN_VOTER_ACTIVATE_FRAMES, ucr::config::SIGN_VOTER_TIMEOUT_FRAMES);
    v21::Config cfg;
    cfg.max_speed = ucr::config::MAX_SPEED_CMS; cfg.min_speed = ucr::config::MIN_SPEED_CMS;
    cfg.curve_scale = ucr::config::CURVE_SCALE; cfg.post_pid_gain = ucr::config::POST_PID_GAIN;
    cfg.min_rows = ucr::config::MIN_ROWS; cfg.coverage_threshold = ucr::config::COVERAGE_THRESHOLD;
    cfg.open_thresh = ucr::config::OPEN_THRESH; cfg.min_height = ucr::config::OPEN_MIN_HEIGHT;
    cfg.smooth_win = ucr::config::OPEN_SMOOTH_WIN; cfg.base_win = ucr::config::OPEN_BASE_WIN;
    cfg.corner_radius = ucr::config::BEZIER_CORNER_RADIUS; cfg.min_turn_dist = ucr::config::BEZIER_MIN_TURN_DIST;
    cfg.extend_len = ucr::config::BEZIER_EXTEND_LEN; cfg.turn_min_ticks = ucr::config::TURN_MIN_TICKS;
    cfg.turn_max_ticks = ucr::config::TURN_MAX_TICKS; cfg.turn_straight_ticks = ucr::config::TURN_STRAIGHT_TICKS;
    cfg.turn_synth_width_px = ucr::config::TURN_SYNTH_WIDTH_PX; cfg.turn_lookahead_y_ratio = ucr::config::TURN_LOOKAHEAD_Y_RATIO;
    cfg.turn_near_y_ratio = ucr::config::TURN_NEAR_Y_RATIO; cfg.turn_far_weight = ucr::config::TURN_FAR_WEIGHT;
    cfg.turn_ramp_ticks = ucr::config::TURN_RAMP_TICKS; cfg.turn_error_cap = ucr::config::TURN_ERROR_CAP;

    v21::LaneFollower follower(cfg);

    cv::Mat frame, small_mask;
    v21::LaneFollower::Output out;
    std::vector<ucr::perception::SignDetector::Detection> lastDets;
    auto next_control = std::chrono::steady_clock::now();
    float applied_speed = 0.0f;
    uint32_t tickIdx = 0;
    
    float tof_vals[4] = {-1.0f, -1.0f, -1.0f, -1.0f};
    float current_heading = 0.0f;
    double fps = 0.0; int frameCount = 0; auto tPrev = std::chrono::high_resolution_clock::now();

    while (!g_stopSignal) {
        camera.GetFrame(frame);
        if (frame.empty()) { motor.SetSpeedCms(0.0f); servo.SetAngle(0); applied_speed = 0.0f; continue; }

        const bool inferOK = road.Predict(frame, small_mask);
        const auto now = std::chrono::steady_clock::now();

        if (!inferOK) {
            motor.SetSpeedCms(0.0f); servo.SetAngle(0); follower.resetTracking(); applied_speed = 0.0f;
        }
        else if (now >= next_control) {
            if (signReady && (tickIdx % ucr::config::SIGN_DETECT_EVERY) == 0) {
                sign.Detect(frame, lastDets);
                std::vector<int> classIds;
                for (auto& d : lastDets) {
                    if (d.confidence < ucr::config::SIGN_CONF_THRESH) continue;
                    int cls = (d.class_id == ucr::config::sign_class::CONSTRUCTION_AHEAD) ? ucr::config::sign_class::TURN_LEFT : d.class_id;
                    if (cls == ucr::config::sign_class::TURN_LEFT || cls == ucr::config::sign_class::TURN_RIGHT ||
                        cls == ucr::config::sign_class::NO_LEFT || cls == ucr::config::sign_class::NO_RIGHT ||
                        cls == ucr::config::sign_class::PARKING) {
                        classIds.push_back(cls);
                    }
                }
                voter.Update(classIds, tickIdx);
            }

            v21::SignState signState = MakeSignState(voter.GetCommitted());

            if (tof.Read(tof_readings, std::chrono::milliseconds(10))) {
                for (int i = 0; i < 4; i++) {
                    if (tof_readings[i].valid) tof_vals[i] = tof_readings[i].distanceMm;
                    else tof_vals[i] = -1.0f;
                }
            }
            if (bno.read(bno_data)) current_heading = bno_data.eulerDeg.x; 

            float tof_left  = tof_vals[ucr::config::TOF_IDX_REAR_LEFT];
            float tof_right = tof_vals[ucr::config::TOF_IDX_REAR_RIGHT];

            out = follower.step(small_mask, signState, tof_left, tof_right, current_heading);

            float desired_real_left = -static_cast<float>(out.angle);
            desired_real_left = std::clamp(desired_real_left, -ucr::config::MAX_REAL_ANGLE_DEG, ucr::config::MAX_REAL_ANGLE_DEG);
            float servo_cmd = ucr::config::RealAngleToServo(desired_real_left);
            servo_cmd = std::clamp(servo_cmd, -static_cast<float>(ucr::config::MAX_SERVO_CMD), static_cast<float>(ucr::config::MAX_SERVO_CMD));
            const int final_angle = static_cast<int>(std::lround(servo_cmd));

            float speed = static_cast<float>(out.speed);
            if (!out.valid && out.coverage > 0.02) speed = ucr::config::MIN_SPEED_CMS;
            const float max_delta = ucr::config::MAX_ACCEL_CMS2 * (ucr::config::CONTROL_PERIOD_MS / 1000.0f);
            if (speed > applied_speed) applied_speed = std::min(speed, applied_speed + max_delta);
            else                       applied_speed = speed;

            motor.SetSpeedCms(applied_speed);
            servo.SetAngle(final_angle);

            ++tickIdx;
            next_control += std::chrono::milliseconds(ucr::config::CONTROL_PERIOD_MS);
            if (now - next_control > std::chrono::seconds(1)) next_control = now + std::chrono::milliseconds(ucr::config::CONTROL_PERIOD_MS);
        }

        ++frameCount;
        auto tNow = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(tNow - tPrev).count();
        if (dt > 0.5) { fps = frameCount / dt; frameCount = 0; tPrev = tNow; }

        if (ucr::config::SHOW_UI) {
            cv::Mat overlay = frame.clone();
            if (inferOK) {
                cv::Mat ui_mask; cv::resize(small_mask, ui_mask, frame.size(), 0, 0, cv::INTER_NEAREST);
                cv::Mat green(frame.size(), frame.type(), cv::Scalar(0, 255, 0));
                cv::Mat blended; cv::addWeighted(frame, 0.6, green, 0.4, 0, blended);
                blended.copyTo(overlay, ui_mask);

                if (!out.final_path.empty() && out.proc_width > 0) {
                    const float sx = float(frame.cols) / out.proc_width; const float sy = float(frame.rows) / out.proc_height;
                    std::vector<cv::Point> pts;
                    for (const auto& p : out.final_path) pts.emplace_back(int(p.x * sx), int(p.y * sy));
                    cv::Scalar color(0, 0, 255); 
                    if (out.state_str.find("PARK") != std::string::npos) color = cv::Scalar(255, 255, 0);
                    for (size_t i = 0; i + 1 < pts.size(); ++i) cv::line(overlay, pts[i], pts[i + 1], color, 2);
                }
            }

            char text1[192], text2[192], text3[192];
            std::snprintf(text1, sizeof(text1), "FPS %.1f | spd %.1f | %s", fps, applied_speed, out.state_str.c_str());
            std::snprintf(text2, sizeof(text2), "HDG:%.1f | T1(RL):%.0f T3(RR):%.0f", current_heading, tof_vals[1], tof_vals[3]);
            
            cv::Scalar hud_color = (out.state_str.find("PARK") != std::string::npos) ? cv::Scalar(255, 255, 0) : cv::Scalar(0, 255, 0);

            cv::putText(overlay, text1, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.55, hud_color, 2);
            cv::putText(overlay, text2, {10, 55}, cv::FONT_HERSHEY_SIMPLEX, 0.55, hud_color, 2);

            if (out.state_str.find("PARK") != std::string::npos) {
                std::snprintf(text3, sizeof(text3), "L:%s R:%s | Base:%.1f Tgt:%.1f",
                    out.dbg_left_blocked ? "BLK" : (out.dbg_left_free ? "FRE" : "---"),
                    out.dbg_right_blocked ? "BLK" : (out.dbg_right_free ? "FRE" : "---"),
                    out.dbg_heading_base, out.dbg_heading_target);
                cv::putText(overlay, text3, {10, 80}, cv::FONT_HERSHEY_SIMPLEX, 0.55, hud_color, 2);
            }

            cv::imshow("Lane Follower", overlay);
            if ((cv::waitKey(1) & 0xFF) == 'q') break;
        }
    }

    motor.SetSpeedCms(0.0f);
    servo.SetAngle(0);
    if (ucr::config::SHOW_UI) cv::destroyAllWindows();
    return 0;
}
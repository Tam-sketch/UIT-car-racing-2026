// ============================================================
// main.cpp  (a.k.a. ucrv6)
//
// UCR 2026 — autonomous road following on the Jetson Nano car.
//
// Pipeline:
//   Camera (BGR 640×384)
//     → RoadSegmenter (TwinLiteNet ONNX/TensorRT)
//     → LaneFollowerV20 (v20 PID) ──┐
//                                   ├─→ combine → servo/motor
//   Camera → SignDetector (thread) ─┤
//     → SignVoter → TurnStateMachine┘
//
// Turn arming uses the V21 opening-detector + quadratic bezier.
// Steering during turn follows the bezier lookahead.
// Fixed 10 Hz control tick, rate-limited handoff between v20 and turn.
// ============================================================
#include "hardware/Camera.h"
#include "hardware/Motor.h"
#include "hardware/Servo.h"
#include "sensor/BNO055.h"
#include "common/Logger.h"
#include "control/LaneFollowerV20.h"
#include "control/TurnStateMachine.h"
#include "perception/RoadSegmenter.h"
#include "perception/SignDetector.h"
#include "perception/SkeletonTopology.h"
#include "config/RobotConfig.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

// ============================================================
// SIGNAL HANDLING
// ============================================================
volatile std::sig_atomic_t g_stopSignal = 0;
static void SignalHandler(int) { g_stopSignal = 1; }

// ============================================================
// SIGN WORKER — runs SignDetector off the control loop.
// ============================================================
struct SignWorker {
    ucr::perception::SignDetector& detector;
    std::mutex mtx;
    std::condition_variable cv;
    cv::Mat latest_frame;
    std::vector<ucr::perception::SignDetector::Detection> current_detections;
    bool new_frame_ready = false;
    bool stop_thread     = false;
    std::thread thread;

    explicit SignWorker(ucr::perception::SignDetector& d) : detector(d) {}

    void Start() { thread = std::thread(&SignWorker::Run, this); }

    void Stop() {
        { std::lock_guard<std::mutex> lock(mtx); stop_thread = true; }
        cv.notify_all();
        if (thread.joinable()) thread.join();
    }

    void PushFrame(const cv::Mat& frame) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            latest_frame = frame;
            new_frame_ready = true;
        }
        cv.notify_one();
    }

    std::vector<ucr::perception::SignDetector::Detection> GetLatest() {
        std::lock_guard<std::mutex> lock(mtx);
        return current_detections;
    }

private:
    void Run() {
        while (true) {
            cv::Mat frame_to_process;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [this] { return new_frame_ready || stop_thread; });
                if (stop_thread) break;
                frame_to_process = latest_frame;
                new_frame_ready = false;
            }
            std::vector<ucr::perception::SignDetector::Detection> dets;
            detector.Detect(frame_to_process, dets);
            { std::lock_guard<std::mutex> lock(mtx); current_detections = dets; }
        }
    }
};

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

    BNO055 imu;
    imu.initialize();
    auto last_imu_time = std::chrono::steady_clock::now();

    // --- Perception: road ---
    ucr::perception::RoadSegmenter road;
    ucr::perception::RoadSegmenter::Config roadCfg;
    roadCfg.daChannel        = ucr::config::ROAD_DA_CHANNEL;
    roadCfg.confidenceMargin = ucr::config::ROAD_CONF_MARGIN;
    if (!road.Init(ucr::config::ROAD_ENGINE_PATH, roadCfg)) {
        std::cerr << "Road model init failed\n";
        return -1;
    }

    // --- Perception: sign ---
    ucr::perception::SignDetector sign_det;
    if (!sign_det.Init(ucr::config::SIGN_ENGINE_PATH)) {
        std::cerr << "Sign model init failed\n";
        return -1;
    }
    SignWorker sign_worker(sign_det);
    sign_worker.Start();

    // --- Topo latch ---
    ucr::perception::TopoLatch topo_latch(ucr::config::TOPO_CONFIRM_FRAMES);
    ucr::perception::SkeletonConfig topo_cfg;

    // --- Control: v20 PID ---
    v20::Config cfg;
    cfg.max_speed          = ucr::config::MAX_SPEED_CMS;
    cfg.min_speed          = ucr::config::MIN_SPEED_CMS;
    cfg.curve_scale        = ucr::config::CURVE_SCALE;
    cfg.post_pid_gain      = ucr::config::POST_PID_GAIN;
    cfg.min_rows           = ucr::config::MIN_ROWS;
    cfg.coverage_threshold = ucr::config::COVERAGE_THRESHOLD;
    v20::LaneFollower follower(cfg);

    // --- Control: turn SM ---
    ucr::control::TurnConfig turnCfg;
    turnCfg.force_angle             = ucr::config::FORCE_ANGLE;
    turnCfg.force_approach_speed    = ucr::config::FORCE_APPROACH_SPEED;
    turnCfg.force_turning_speed     = ucr::config::FORCE_TURNING_SPEED;
    turnCfg.force_exit_speed        = ucr::config::FORCE_EXIT_SPEED;
    turnCfg.force_turn_phase_deg    = ucr::config::FORCE_TURN_PHASE_DEG;
    turnCfg.force_exit_phase_deg    = ucr::config::FORCE_EXIT_PHASE_DEG;
    turnCfg.heading_reset_deg       = ucr::config::HEADING_RESET_DEG;
    turnCfg.reset_debounce_ticks    = ucr::config::RESET_DEBOUNCE_FRAMES;
    turnCfg.sign_snapshot_hold_ticks= ucr::config::SIGN_SNAPSHOT_HOLD_FRAMES;
    turnCfg.max_force_ticks         = ucr::config::MAX_FORCE_TICKS;
    turnCfg.slow_speed              = ucr::config::SLOW_SPEED_CMS;
    turnCfg.max_speed               = ucr::config::MAX_SPEED_CMS;
    turnCfg.sign_hold_ticks         = ucr::config::SIGN_HOLD_FRAMES;
    turnCfg.open_thresh             = ucr::config::OPEN_THRESH;
    turnCfg.open_min_height         = ucr::config::OPEN_MIN_HEIGHT;
    turnCfg.open_smooth_win         = ucr::config::OPEN_SMOOTH_WIN;
    turnCfg.open_base_win           = ucr::config::OPEN_BASE_WIN;
    turnCfg.turn_opening_proximity  = ucr::config::TURN_OPENING_PROXIMITY;
    turnCfg.turn_arm_confirm        = ucr::config::TURN_ARM_CONFIRM;
    turnCfg.bezier_corner_radius    = ucr::config::BEZIER_CORNER_RADIUS;
    turnCfg.bezier_min_turn_dist    = ucr::config::BEZIER_MIN_TURN_DIST;
    turnCfg.bezier_extend_len       = ucr::config::BEZIER_EXTEND_LEN;
    turnCfg.bezier_lookahead_frac   = ucr::config::BEZIER_LOOKAHEAD_FRAC;
    turnCfg.bezier_steer_gain       = ucr::config::BEZIER_STEER_GAIN;
    turnCfg.idx_construction_ahead  = ucr::config::sign_class::CONSTRUCTION_AHEAD;
    turnCfg.idx_no_left             = ucr::config::sign_class::NO_LEFT;
    turnCfg.idx_no_right            = ucr::config::sign_class::NO_RIGHT;
    turnCfg.idx_parking             = ucr::config::sign_class::PARKING;
    turnCfg.idx_roundabout          = ucr::config::sign_class::ROUNDABOUT;
    turnCfg.idx_traffic_light       = ucr::config::sign_class::TRAFFIC_LIGHT;
    turnCfg.idx_turn_left           = ucr::config::sign_class::TURN_LEFT;
    turnCfg.idx_turn_right          = ucr::config::sign_class::TURN_RIGHT;

    ucr::control::TurnStateMachine turn_sm(
        turnCfg,
        ucr::config::SIGN_VOTER_ACTIVATE_FRAMES,
        ucr::config::SIGN_VOTER_TIMEOUT_FRAMES);

    // --- Runtime state ---
    cv::Mat frame, small_mask;
    v20::LaneFollower::Output out;
    ucr::control::TurnStateMachine::Output turn_out;

    double fps = 0.0;
    int    frameCount = 0;
    auto   tPrev = std::chrono::high_resolution_clock::now();

    auto  next_control     = std::chrono::steady_clock::now();
    auto  last_loop_time   = std::chrono::steady_clock::now();
    float applied_speed    = 0.0f;
    float last_pid_angle   = 0.0f;   // for rate-limited handoff
    uint32_t ctrl_tick     = 0;
    float last_heading     = 0.0f;

    std::cout << "\n=== UCR2026: Auto Run (v20 + opening/bezier turn) ===\n"
              << "Road engine: " << ucr::config::ROAD_ENGINE_PATH << "\n"
              << "Sign engine: " << ucr::config::SIGN_ENGINE_PATH << "\n"
              << "Control period: " << ucr::config::CONTROL_PERIOD_MS << " ms\n"
              << "Press 'q' to quit\n\n";

    while (!g_stopSignal) {
        const auto loop_now = std::chrono::steady_clock::now();
        const double loop_dt =
            std::chrono::duration<double>(loop_now - last_loop_time).count();
        last_loop_time = loop_now;

        // Watchdog: if the loop is running too slowly, stop safely.
        if (loop_dt > 0.30) {
            std::cerr << "\n[WATCHDOG] loop_dt = " << loop_dt
                      << "s. EMERGENCY STOP.\n";
            motor.SetSpeedCms(0.0f);
            servo.SetAngle(0);
            applied_speed = 0.0f;
            next_control  = std::chrono::steady_clock::now();
            continue;
        }

        if (loop_now < next_control) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        next_control += std::chrono::milliseconds(ucr::config::CONTROL_PERIOD_MS);
        ++ctrl_tick;

        camera.GetFrame(frame);
        if (frame.empty()) {
            motor.SetSpeedCms(0.0f);
            servo.SetAngle(0);
            applied_speed = 0.0f;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // --- IMU heading (best effort) ---
        {
            BNO055::Data imu_data;
            if (imu.read(imu_data)) {
                const auto now_i = std::chrono::steady_clock::now();
                const double dt_i =
                    std::chrono::duration<double>(now_i - last_imu_time).count();
                last_imu_time = now_i;
                imu.integrateYawRate(dt_i);
                last_heading = static_cast<float>(imu.relativeYawDeg());
            }
        }
        const float heading = last_heading;

        // --- Road inference ---
        const bool inferOK = road.Predict(frame, small_mask);
        const auto now     = std::chrono::steady_clock::now();
        if (!inferOK) {
            motor.SetSpeedCms(0.0f);
            servo.SetAngle(0);
            follower.resetTracking();
            turn_sm.Reset();
            applied_speed = 0.0f;
            last_pid_angle = 0.0f;
            continue;
        }

        // --- Push frame to sign worker ---
        sign_worker.PushFrame(frame);
        const auto dets = sign_worker.GetLatest();

        // --- Sign detect + topo, every SIGN_DETECT_EVERY ticks ---
        if ((ctrl_tick % ucr::config::SIGN_DETECT_EVERY) == 0) {
            std::vector<int> class_ids;
            class_ids.reserve(dets.size());
            for (const auto& d : dets) class_ids.push_back(d.class_id);
            turn_sm.UpdateVoter(class_ids, ctrl_tick);

            ucr::perception::TopoResult raw_topo =
                ucr::perception::ComputeTopology(small_mask, topo_cfg);
            turn_sm.SetTopo(topo_latch.Update(raw_topo));
        }

        // --- v20 lane following ---
        out = follower.step(small_mask);

        // --- Turn state machine ---
        turn_out = turn_sm.Update(ctrl_tick, heading, small_mask, out.centerline_pts);

        // --- Combine angle + speed ---
        float target_pid_angle = turn_out.in_force_turn
                               ? turn_out.force_angle_deg
                               : (float)out.angle;

        // Rate-limited handoff
        {
            float delta = target_pid_angle - last_pid_angle;
            delta = std::clamp(delta,
                               -ucr::config::PID_RATE_LIMIT_DEG,
                               +ucr::config::PID_RATE_LIMIT_DEG);
            last_pid_angle += delta;
        }
        const float final_angle_pid = last_pid_angle;

        float target_speed;
        if (turn_out.in_force_turn) {
            target_speed = turn_out.force_speed_cms;
        } else {
            target_speed = std::min((float)out.speed, turn_out.max_speed_cap_cms);
        }
        // Lost-mask creep
        if (!out.valid && out.coverage > 0.02f && !turn_out.in_force_turn) {
            target_speed = ucr::config::MIN_SPEED_CMS;
        }

        // --- Servo command via calibration ---
        float desired_real_left = -final_angle_pid;
        desired_real_left = std::clamp(desired_real_left,
                                        -ucr::config::MAX_REAL_ANGLE_DEG,
                                        +ucr::config::MAX_REAL_ANGLE_DEG);
        float servo_cmd = ucr::config::RealAngleToServo(desired_real_left);
        servo_cmd = std::clamp(servo_cmd,
                                -(float)ucr::config::MAX_SERVO_CMD,
                                +(float)ucr::config::MAX_SERVO_CMD);
        const int final_servo = (int)std::lround(servo_cmd);

        // --- Soft-start ramp ---
        const float max_delta = ucr::config::MAX_ACCEL_CMS2 *
                                (ucr::config::CONTROL_PERIOD_MS / 1000.0f);
        if (target_speed > applied_speed) {
            applied_speed = std::min(target_speed, applied_speed + max_delta);
        } else {
            applied_speed = target_speed;
        }

        motor.SetSpeedCms(applied_speed);
        servo.SetAngle(final_servo);

        // --- FPS ---
        ++frameCount;
        const auto tNow = std::chrono::high_resolution_clock::now();
        const double dt = std::chrono::duration<double>(tNow - tPrev).count();
        if (dt > 0.5) {
            fps = frameCount / dt;
            frameCount = 0;
            tPrev = tNow;
        }

        // --- UI ---
        if (ucr::config::SHOW_UI) {
            cv::Mat overlay = frame.clone();

            // Mask overlay
            cv::Mat ui_mask;
            cv::resize(small_mask, ui_mask, frame.size(), 0, 0, cv::INTER_NEAREST);
            cv::Mat green(frame.size(), frame.type(), cv::Scalar(0, 255, 0));
            cv::Mat blended;
            cv::addWeighted(frame, 0.6, green, 0.4, 0, blended);
            blended.copyTo(overlay, ui_mask);

            // Centerline (from v20)
            if (!out.centerline_pts.empty() &&
                out.proc_width > 0 && out.proc_height > 0) {
                const float sx = (float)frame.cols / (float)out.proc_width;
                const float sy = (float)frame.rows / (float)out.proc_height;
                std::vector<cv::Point> pts;
                pts.reserve(out.centerline_pts.size());
                for (const auto& p : out.centerline_pts) {
                    pts.emplace_back((int)(p.first * sx), (int)(p.second * sy));
                }
                for (size_t i = 0; i + 1 < pts.size(); ++i) {
                    cv::line(overlay, pts[i], pts[i + 1],
                             cv::Scalar(0, 0, 255), 2);
                }
            }

            // Sign boxes
            for (const auto& det : dets) {
                cv::rectangle(overlay, det.bbox, cv::Scalar(0, 255, 255), 2);
                cv::putText(overlay,
                    ucr::config::SignClassName(det.class_id),
                    cv::Point((int)det.bbox.x, (int)det.bbox.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 255, 255), 1);
            }

            char text[256];
            std::snprintf(text, sizeof(text),
                "FPS %.1f | spd %.1f/%.1f | PID %.1f | srv %d | kf %.2f%s%s",
                fps, applied_speed, target_speed,
                out.angle_pre_gain, final_servo, out.kept_frac,
                (!out.valid && inferOK) ? " INV" : "",
                turn_out.in_force_turn ? " TURN" : "");
            cv::putText(overlay, text, {10, 30},
                        cv::FONT_HERSHEY_SIMPLEX, 0.55,
                        cv::Scalar(0, 255, 0), 2);

            char line2[256];
            std::snprintf(line2, sizeof(line2),
                "sign=%s | phase=%s | dH %.1f | openings=%d | y_mid=%d",
                ucr::config::SignClassName(turn_out.sign_id),
                turn_out.phase,
                turn_out.d_heading,
                turn_out.n_openings,
                turn_out.chosen_y_mid);
            cv::putText(overlay, line2, {10, 58},
                        cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 200, 255), 1);

            cv::imshow("Lane Follower", overlay);
            const int key = cv::waitKey(1) & 0xFF;
            if (key == 27 || key == 'q') break;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    sign_worker.Stop();
    motor.SetSpeedCms(0.0f);
    servo.SetAngle(0);
    if (ucr::config::SHOW_UI) cv::destroyAllWindows();
    std::cout << "\n[System] Stopped safely\n";
    return 0;
}
#include "config/RobotConfig.h"
#include "perception/RoadSegmenter.h"
#include "perception/SignDetector.h"
#include "control/LaneFollowerV22.h" 

#include "hardware/Camera.h"
#include "hardware/Motor.h"
#include "hardware/Servo.h"
#include "sensor/BNO055.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <cmath>
#include <algorithm>
#include <csignal>
#include <mutex>
#include <condition_variable>

volatile std::sig_atomic_t g_running = 1;
void SignalHandler(int) { g_running = 0; }

struct SignWorker {
    ucr::perception::SignDetector& detector;
    std::mutex mtx;
    std::condition_variable cv;
    cv::Mat latest_frame;
    std::vector<ucr::perception::Detection> current_detections;
    bool new_frame_ready = false;
    bool stop_thread = false;
    std::thread thread;

    SignWorker(ucr::perception::SignDetector& d) : detector(d) {}
    void Start() { thread = std::thread(&SignWorker::Run, this); }
    void Stop() {
        { std::lock_guard<std::mutex> lock(mtx); stop_thread = true; }
        cv.notify_all();
        if (thread.joinable()) thread.join();
    }
    void PushFrame(const cv::Mat& frame) {
        { std::lock_guard<std::mutex> lock(mtx); latest_frame = frame; new_frame_ready = true; }
        cv.notify_one();
    }
    std::vector<ucr::perception::Detection> GetLatest() {
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
            std::vector<ucr::perception::Detection> dets;
            detector.Predict(frame_to_process, dets);
            { std::lock_guard<std::mutex> lock(mtx); current_detections = dets; }
        }
    }
};

std::string GetSignName(int id) {
    switch(id) {
        case 0: return "construction_ahead";
        case 1: return "no_left";
        case 2: return "no_right";
        case 6: return "turn_left";
        case 7: return "turn_right";
        default: return "unknown";
    }
}

// Đã sửa: Trả lại Gain 2.2 thay vì công thức hồi quy
int16_t ApplySteeringCalibration(float pid_angle) {
    if (std::isnan(pid_angle) || std::isinf(pid_angle)) pid_angle = 0.0f;
    float cmd = pid_angle * 2.2f; 
    if (ucr::config::INVERT_STEER) cmd = -cmd;
    return static_cast<int16_t>(std::clamp(cmd, -ucr::config::MAX_STEER_DEG, ucr::config::MAX_STEER_DEG));
}

int main() {
    std::cout << "=== UCR2026 V22 + TurnPlanner ===\n";
    std::signal(SIGINT, SignalHandler);

    Camera camera;
    if (!camera.Init()) return -1;
    Motor motor("/dev/ttyTHS1");
    if (!motor.Init()) return -1;
    Servo servo(0x40);
    if (!servo.Init()) return -1;
    BNO055 imu;
    imu.initialize();

    ucr::perception::RoadSegmenter road_seg;
    road_seg.Init(ucr::config::ROAD_ENGINE_PATH);
    
    ucr::perception::SignDetector sign_det;
    sign_det.Init(ucr::config::SIGN_ENGINE_PATH);

    SignWorker sign_worker(sign_det);
    sign_worker.Start();

    v22::Config lane_cfg;
    lane_cfg.post_pid_gain = 1.0f; 
    lane_cfg.max_speed = ucr::config::MAX_SPEED_CMS;
    lane_cfg.min_speed = ucr::config::MIN_SPEED_CMS;
    v22::LaneFollower lane_follower(lane_cfg);

    int frame_idx = 0;
    float current_speed = 0.0f;
    auto next_tick = std::chrono::steady_clock::now();
    
    // Cờ báo hiệu đã kích hoạt TurnPlanner chưa
    bool sign_processed = false; 

    std::cout << "[Main] Bắt đầu vòng lặp 10 Hz...\n";

    while (g_running) {
        auto now = std::chrono::steady_clock::now();
        if (now < next_tick) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); continue; }
        next_tick += std::chrono::milliseconds(ucr::config::CONTROL_PERIOD_MS);
        frame_idx++;

        cv::Mat frame;
        camera.GetFrame(frame);
        if (frame.empty()) continue;

        cv::Mat mask;
        road_seg.Predict(frame, mask);

        sign_worker.PushFrame(frame);
        auto signs = sign_worker.GetLatest();

        // ============================================================
        // KÍCH HOẠT TURN PLANNER TỪ BIỂN BÁO
        // ============================================================
        if (!signs.empty() && !sign_processed) {
            auto best_sign = signs[0];
            std::string sign_name = GetSignName(best_sign.class_id);
            
            if (sign_name != "unknown" && sign_name != "traffic_light" && sign_name != "parking" && sign_name != "roundabout") {
                std::cout << "\n[AI] Phat hien bien bao: " << sign_name << "\n";
                // Kích hoạt bẻ lái tại tọa độ Y = 180 (gần giữa ảnh)
                lane_follower.setSign(sign_name, 180); 
                sign_processed = true; 
            }
        }

        auto out = lane_follower.step(mask);

        // ============================================================
        // UI MASK OVERLAY
        // ============================================================
        cv::Mat ui_frame = frame.clone();
        if (!mask.empty()) {
            cv::Mat mask_resized;
            cv::resize(mask, mask_resized, frame.size(), 0, 0, cv::INTER_NEAREST);
            cv::Mat color_mask = cv::Mat::zeros(frame.size(), CV_8UC3);
            color_mask.setTo(cv::Scalar(0, 255, 0), mask_resized > 0);
            cv::addWeighted(ui_frame, 0.6, color_mask, 0.4, 0, ui_frame);
        }

        // Vẽ centerline (đã bị bẻ cong nếu có biển báo)
        if (!out.centerline_pts.empty() && out.proc_width > 0) {
            float sx = static_cast<float>(frame.cols) / out.proc_width;
            float sy = static_cast<float>(frame.rows) / out.proc_height;
            for (size_t i = 0; i < out.centerline_pts.size() - 1; ++i) {
                cv::Point p1(out.centerline_pts[i].first * sx, out.centerline_pts[i].second * sy);
                cv::Point p2(out.centerline_pts[i+1].first * sx, out.centerline_pts[i+1].second * sy);
                cv::line(ui_frame, p1, p2, cv::Scalar(0, 0, 255), 2); 
            }
        }

        cv::putText(ui_frame, "State: " + out.state_str, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        cv::putText(ui_frame, "near_e: " + std::to_string((int)out.near_e), cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        
        cv::imshow("Camera & Mask Overlay", ui_frame);
        if (cv::waitKey(1) == 'q') g_running = 0;

        float target_speed = out.speed;
        float max_delta = ucr::config::MAX_ACCEL_CMS2 * (ucr::config::CONTROL_PERIOD_MS / 1000.0f);
        current_speed = (target_speed > current_speed) ? std::min(target_speed, current_speed + max_delta) : target_speed;

        int16_t servo_angle = ApplySteeringCalibration(out.angle);
        servo.SetAngle(servo_angle);
        motor.SetSpeedCms(current_speed);
        
        std::cout << "\r[" << frame_idx << "] " << out.state_str 
                  << " | nE=" << (int)out.near_e << " fE=" << (int)out.far_e 
                  << " | PID=" << out.angle << " Srv=" << servo_angle 
                  << " | Spd=" << current_speed << "   " << std::flush;
    }

    sign_worker.Stop();
    motor.SetSpeedCms(0);
    servo.SetAngle(0);
    cv::destroyAllWindows();
    return 0;
}

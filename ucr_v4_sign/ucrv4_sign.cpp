#include "config/RobotConfig.h"
#include "perception/RoadSegmenter.h"
#include "perception/SignDetector.h"
#include "mission/MissionStateMachine.h"

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

// ============================================================
// MULTITHREADING CHO SIGN DETECTOR
// ============================================================
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
        case 0: return "construction";
        case 1: return "no_left";
        case 2: return "no_right";
        case 6: return "turn_left";
        case 7: return "turn_right";
        default: return "sign_" + std::to_string(id);
    }
}

// Phương trình hồi quy góc lái
int16_t ApplySteeringCalibration(float pid_angle) {
    float cmd = 0.0f;
    if (pid_angle < 0.0f) cmd = (pid_angle + 6.2f) / 0.96f;
    else if (pid_angle > 0.0f) cmd = (pid_angle - 2.5f) / 0.71f;
    if (ucr::config::INVERT_STEER) cmd = -cmd;
    return static_cast<int16_t>(std::clamp(cmd, -ucr::config::MAX_STEER_DEG, ucr::config::MAX_STEER_DEG));
}

int main() {
    std::cout << "=== UCR2026 V4 SIGN (Multithreading) ===\n";
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

    ucr::mission::MissionStateMachine mission;

    int frame_idx = 0;
    float current_speed = 0.0f;
    auto next_tick = std::chrono::steady_clock::now();
    
    std::cout << "[Main] Bắt đầu vòng lặp 10 Hz...\n";

    while (g_running) {
        auto now = std::chrono::steady_clock::now();
        if (now < next_tick) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); continue; }
        next_tick += std::chrono::milliseconds(ucr::config::CONTROL_PERIOD_MS);
        frame_idx++;

        cv::Mat frame;
        camera.GetFrame(frame);
        if (frame.empty()) continue;

        float heading = 0.0f;
        BNO055::Data imu_data;
        if (imu.read(imu_data)) heading = static_cast<float>(imu_data.eulerDeg.y); 

        cv::Mat mask;
        road_seg.Predict(frame, mask);

        sign_worker.PushFrame(frame);
        auto signs = sign_worker.GetLatest();

        // ============================================================
        // UI
        // ============================================================
        cv::Mat ui_frame = frame.clone();
        cv::Mat sign_panel = cv::Mat(150, 400, CV_8UC3, cv::Scalar(40, 40, 40));
        int ui_x_offset = 10;

        for (const auto& det : signs) {
            std::string name = GetSignName(det.class_id);
            cv::rectangle(ui_frame, det.bbox, cv::Scalar(0, 255, 0), 2);
            
            if (det.bbox.x >= 0 && det.bbox.y >= 0 && det.bbox.x + det.bbox.width <= frame.cols && det.bbox.y + det.bbox.height <= frame.rows) {
                cv::Mat crop = frame(det.bbox).clone();
                cv::resize(crop, crop, cv::Size(128, 128));
                if (ui_x_offset + 128 <= sign_panel.cols) {
                    crop.copyTo(sign_panel(cv::Rect(ui_x_offset, 10, 128, 128)));
                    cv::putText(sign_panel, name, cv::Point(ui_x_offset, 145), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
                    ui_x_offset += 135;
                }
            }
        }
        cv::putText(ui_frame, "Spd: " + std::to_string(current_speed), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        cv::imshow("Camera", ui_frame);
        cv::imshow("Signs", sign_panel);
        if (!mask.empty()) cv::imshow("Mask", mask);
        
        if (cv::waitKey(1) == 'q') g_running = 0;
        // ============================================================

        auto cmd = mission.Update(mask, signs, heading, frame_idx);

        float target_speed = cmd.speed;
        float max_delta = ucr::config::MAX_ACCEL_CMS2 * (ucr::config::CONTROL_PERIOD_MS / 1000.0f);
        current_speed = (target_speed > current_speed) ? std::min(target_speed, current_speed + max_delta) : target_speed;

        int16_t servo_angle = ApplySteeringCalibration(cmd.angle);
        servo.SetAngle(servo_angle);
        motor.SetSpeedCms(current_speed);
        
        std::cout << "\r[" << frame_idx << "] Spd=" << current_speed << " PID=" << cmd.angle << " Srv=" << servo_angle << " | Signs=" << signs.size() << "   " << std::flush;
    }

    sign_worker.Stop();
    motor.SetSpeedCms(0);
    servo.SetAngle(0);
    cv::destroyAllWindows();
    return 0;
}
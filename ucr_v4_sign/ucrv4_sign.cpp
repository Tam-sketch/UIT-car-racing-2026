#include "config/RobotConfig.h"
#include "perception/RoadSegmenter.h"
#include "perception/SignDetector.h"
#include "mission/MissionStateMachine.h"

// Include Hardware (Đã sửa lại đường dẫn chuẩn theo repo)
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

// Biến toàn cục để xử lý Ctrl+C an toàn
volatile std::sig_atomic_t g_running = 1;
void SignalHandler(int) { g_running = 0; }

// Hàm hiệu chuẩn góc lái từ dữ liệu hồi quy
int16_t ApplySteeringCalibration(float pid_angle) {
    float cmd = 0.0f;
    if (pid_angle < 0.0f) {
        // Bánh trái: x = (y + 6.2) / 0.96
        cmd = (pid_angle + 6.2f) / 0.96f;
    } else if (pid_angle > 0.0f) {
        // Bánh phải: x = (y - 2.5) / 0.71
        cmd = (pid_angle - 2.5f) / 0.71f;
    }
    
    // Đảo chiều servo nếu cần (kiểm tra thực tế trên xe)
    if (ucr::config::INVERT_STEER) cmd = -cmd;
    
    // Clamp và ép kiểu về int16_t cho Servo
    cmd = std::clamp(cmd, -ucr::config::MAX_STEER_DEG, ucr::config::MAX_STEER_DEG);
    return static_cast<int16_t>(cmd);
}

int main() {
    std::cout << "=== UCR2026 V3: Sign & Mission ===\n";
    std::signal(SIGINT, SignalHandler);

    // 1. Khởi tạo phần cứng
    Camera camera;
    if (!camera.Init()) {
        std::cerr << "Camera Init Failed\n";
        return -1;
    }

    Motor motor("/dev/ttyTHS1");
    if (!motor.Init()) {
        std::cerr << "Motor Init Failed\n";
        return -1;
    }

    Servo servo(0x40);
    if (!servo.Init()) {
        std::cerr << "Servo Init Failed\n";
        return -1;
    }

    BNO055 imu;
    if (!imu.initialize()) {
        std::cerr << "[WARN] IMU Init Failed. Fallback heading = 0\n";
    }

    // 2. Khởi tạo Perception
    ucr::perception::RoadSegmenter road_seg;
    if (!road_seg.Init(ucr::config::ROAD_ENGINE_PATH)) {
        std::cerr << "RoadSegmenter Init Failed\n";
        return -1;
    }
    
    ucr::perception::SignDetector sign_det;
    if (!sign_det.Init(ucr::config::SIGN_ENGINE_PATH)) {
        std::cerr << "SignDetector Init Failed\n";
        return -1;
    }

    // 3. Cấu hình LaneFollower (Tắt post_pid_gain)
    v20::Config lane_cfg;
    lane_cfg.post_pid_gain = 1.0f; // Tắt gain cũ để dùng hồi quy
    lane_cfg.max_speed = ucr::config::MAX_SPEED_CMS;
    lane_cfg.min_speed = ucr::config::MIN_SPEED_CMS;
    lane_cfg.min_rows = ucr::config::MIN_ROWS;
    lane_cfg.coverage_threshold = ucr::config::COVERAGE_THRESHOLD;
    lane_cfg.curve_scale = ucr::config::CURVE_SCALE;

    // 4. Khởi tạo Mission
    ucr::mission::MissionStateMachine mission(lane_cfg);

    // 5. Vòng lặp chính
    int frame_idx = 0;
    float current_speed = 0.0f;
    auto next_tick = std::chrono::steady_clock::now();
    
    std::cout << "[Main] Bắt đầu vòng lặp 10 Hz...\n";

    while (g_running) {
        auto now = std::chrono::steady_clock::now();
        if (now < next_tick) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        next_tick += std::chrono::milliseconds(ucr::config::CONTROL_PERIOD_MS);
        frame_idx++;

        // A. Lấy dữ liệu
        cv::Mat frame;
        camera.GetFrame(frame);
        if (frame.empty()) continue;

        float heading = 0.0f;
        BNO055::Data imu_data;
        if (imu.read(imu_data)) {
            // CHỈ SỬ DỤNG TRỤC Y CHO HEADING THEO YÊU CẦU
            heading = static_cast<float>(imu_data.eulerDeg.y); 
        }

        // B. Chạy AI
        cv::Mat mask;
        road_seg.Predict(frame, mask);
        
        std::vector<ucr::perception::Detection> signs;
        sign_det.Predict(frame, signs);

        // C. Ra quyết định
        auto cmd = mission.Update(mask, signs, heading, frame_idx);

        // D. Soft-start (Giới hạn gia tốc)
        float target_speed = cmd.speed;
        float max_delta = ucr::config::MAX_ACCEL_CMS2 * (ucr::config::CONTROL_PERIOD_MS / 1000.0f);
        if (target_speed > current_speed) {
            current_speed = std::min(target_speed, current_speed + max_delta);
        } else {
            current_speed = target_speed;
        }

        // E. Hiệu chuẩn góc lái
        int16_t servo_angle = ApplySteeringCalibration(cmd.angle);

        // F. Gửi lệnh xuống phần cứng
        servo.SetAngle(servo_angle);
        motor.SetSpeedCms(current_speed);

        // Log debug
        std::cout << "[" << frame_idx << "] "
                  << "Spd=" << current_speed << " "
                  << "PID=" << cmd.angle << " "
                  << "Servo=" << servo_angle << " "
                  << "Head=" << heading << " "
                  << "Signs=" << signs.size() << "\n";
    }

    // Dừng xe an toàn khi thoát
    std::cout << "\n[Main] Dừng xe an toàn...\n";
    motor.SetSpeedCms(0);
    servo.SetAngle(0);
    
    return 0;
}

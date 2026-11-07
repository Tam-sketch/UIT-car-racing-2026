#include "config/RobotConfig.h"
#include "perception/RoadSegmenter.h"
#include "perception/SignDetector.h"
#include "mission/MissionStateMachine.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <cmath>
#include <algorithm>

// TODO: Include các header phần cứng của bạn
// #include "hardware/Camera.h"
// #include "hardware/Motor.h"
// #include "hardware/Servo.h"
// #include "hardware/IMU.h"

float ApplySteeringCalibration(float pid_angle) {
    float cmd = 0.0f;
    if (pid_angle < 0.0f) {
        // Bánh trái: y = 0.96x - 6.2  =>  x = (y + 6.2) / 0.96
        cmd = (pid_angle + 6.2f) / 0.96f;
    } else if (pid_angle > 0.0f) {
        // Bánh phải: y = 0.71x + 2.5  =>  x = (y - 2.5) / 0.71
        cmd = (pid_angle - 2.5f) / 0.71f;
    }
    
    if (ucr::config::INVERT_STEER) cmd = -cmd;
    return std::clamp(cmd, -ucr::config::MAX_STEER_DEG, ucr::config::MAX_STEER_DEG);
}

int main() {
    std::cout << "=== UCR2026 V3: Sign & Mission ===\n";

    // 1. Khởi tạo phần cứng (Mở comment và điền API thật)
    // ucr::hardware::Camera camera;
    // ucr::hardware::Motor motor;
    // ucr::hardware::Servo servo;
    // ucr::hardware::IMU imu;

    // 2. Khởi tạo Perception
    ucr::perception::RoadSegmenter road_seg;
    if (!road_seg.Init(ucr::config::ROAD_ENGINE_PATH)) {
        std::cerr << "Failed to init RoadSegmenter\n";
        return -1;
    }
    
    ucr::perception::SignDetector sign_det;
    if (!sign_det.Init(ucr::config::SIGN_ENGINE_PATH)) {
        std::cerr << "Failed to init SignDetector\n";
        return -1;
    }

    // 3. Cấu hình LaneFollower (Tắt post_pid_gain cũ)
    v20::Config lane_cfg;
    lane_cfg.post_pid_gain = 1.0f;  // QUAN TRỌNG: Tắt gain 2.2 cũ
    lane_cfg.max_speed = ucr::config::MAX_SPEED_CMS;
    lane_cfg.min_speed = ucr::config::MIN_SPEED_CMS;
    // lane_cfg.min_rows = ucr::config::MIN_ROWS; (nếu cần)
    // lane_cfg.coverage_threshold = ucr::config::COVERAGE_THRESHOLD;

    // 4. Khởi tạo Mission với cấu hình mới
    ucr::mission::MissionStateMachine mission(lane_cfg);

    // 5. Biến vòng lặp
    int frame_idx = 0;
    float current_speed = 0.0f;
    auto next_tick = std::chrono::steady_clock::now();
    
    std::cout << "[Main] Bắt đầu vòng lặp 10 Hz...\n";

    while (true) {
        auto now = std::chrono::steady_clock::now();
        if (now < next_tick) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        next_tick += std::chrono::milliseconds(ucr::config::CONTROL_PERIOD_MS);
        frame_idx++;

        // A. Lấy dữ liệu (Thay bằng code thật)
        // cv::Mat frame = camera.GetFrame();
        // float heading = imu.GetHeading();
        
        cv::Mat frame(384, 640, CV_8UC3, cv::Scalar(0,0,0)); 
        float heading = 0.0f;

        // B. Chạy AI
        cv::Mat mask;
        road_seg.Predict(frame, mask);
        
        std::vector<ucr::perception::Detection> signs;
        sign_det.Predict(frame, signs);

        // C. Ra quyết định
        auto cmd = mission.Update(mask, signs, heading, frame_idx);

        // D. Soft-start cho tốc độ (Giới hạn gia tốc)
        float target_speed = cmd.speed;
        float max_delta = ucr::config::MAX_ACCEL_CMS2 * (ucr::config::CONTROL_PERIOD_MS / 1000.0f);
        if (target_speed > current_speed) {
            current_speed = std::min(target_speed, current_speed + max_delta);
        } else {
            current_speed = target_speed;
        }

        // E. Hiệu chuẩn góc lái (Áp dụng phương trình hồi quy)
        float servo_angle = ApplySteeringCalibration(cmd.angle);

        // F. Gửi lệnh xuống phần cứng (Thay bằng code thật)
        // servo.SetAngle(servo_angle);
        // motor.SetSpeedCms(current_speed);

        // Log debug
        std::cout << "[" << frame_idx << "] "
                  << "Spd=" << current_speed << " "
                  << "PID=" << cmd.angle << " "
                  << "Servo=" << servo_angle << " "
                  << "Heading=" << heading << "\n";

        // G. Thoát (Thay bằng cv::waitKey)
        // if (cv::waitKey(1) == 'q') break;
    }

    // Dừng xe an toàn
    // motor.SetSpeedCms(0);
    // servo.SetAngle(0);
    return 0;
}
#include "config/RobotConfig.h"
#include "perception/RoadSegmenter.h"
#include "perception/SignDetector.h"
#include "mission/MissionStateMachine.h"

// Include Hardware
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

// Hàm lấy tên biển báo từ Class ID
std::string GetSignName(int id) {
    switch(id) {
        case 0: return "construction_ahead";
        case 1: return "no_left";
        case 2: return "no_right";
        case 3: return "parking";
        case 4: return "roundabout";
        case 5: return "traffic_light";
        case 6: return "turn_left";
        case 7: return "turn_right";
        default: return "unknown";
    }
}

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
    
    if (ucr::config::INVERT_STEER) cmd = -cmd;
    cmd = std::clamp(cmd, -ucr::config::MAX_STEER_DEG, ucr::config::MAX_STEER_DEG);
    return static_cast<int16_t>(cmd);
}

int main() {
    std::cout << "=== UCR2026 V3: Sign & Mission ===\n";
    std::signal(SIGINT, SignalHandler);

    // 1. Khởi tạo phần cứng
    Camera camera;
    if (!camera.Init()) return -1;

    Motor motor("/dev/ttyTHS1");
    if (!motor.Init()) return -1;

    Servo servo(0x40);
    if (!servo.Init()) return -1;

    BNO055 imu;
    if (!imu.initialize()) {
        std::cerr << "[WARN] IMU Init Failed. Fallback heading = 0\n";
    }

    // 2. Khởi tạo Perception
    ucr::perception::RoadSegmenter road_seg;
    if (!road_seg.Init(ucr::config::ROAD_ENGINE_PATH)) return -1;
    
    ucr::perception::SignDetector sign_det;
    if (!sign_det.Init(ucr::config::SIGN_ENGINE_PATH)) return -1;

    // 3. Cấu hình LaneFollower (Tắt post_pid_gain)
    v20::Config lane_cfg;
    lane_cfg.post_pid_gain = 1.0f; 
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
    
    std::cout << "[Main] Bắt đầu vòng lặp 10 Hz... Nhấn 'q' hoặc ESC trên cửa sổ UI để thoát.\n";

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
            heading = static_cast<float>(imu_data.eulerDeg.y); 
        }

        // B. Chạy AI
        cv::Mat mask;
        road_seg.Predict(frame, mask);
        
        std::vector<ucr::perception::Detection> signs;
        sign_det.Predict(frame, signs);

        // ============================================================
        // [UI DEBUGGING] HIỂN THỊ BIỂN BÁO VÀ MASK
        // ============================================================
        cv::Mat ui_frame = frame.clone();
        
        // Vẽ khung lên ảnh Camera
        cv::Mat sign_panel = cv::Mat(150, 400, CV_8UC3, cv::Scalar(40, 40, 40)); // Panel đen để chứa ảnh crop
        int ui_x_offset = 10;

        for (size_t i = 0; i < signs.size(); ++i) {
            const auto& det = signs[i];
            std::string name = GetSignName(det.class_id);
            
            // Vẽ Bounding Box
            cv::rectangle(ui_frame, det.bbox, cv::Scalar(0, 255, 0), 2);
            
            // Ghi text lên Bounding Box
            std::string label = name + " " + std::to_string(static_cast<int>(det.confidence * 100)) + "%";
            cv::putText(ui_frame, label, cv::Point(det.bbox.x, det.bbox.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

            // Lấy ảnh crop của biển báo và hiển thị sang Panel riêng
            if (det.bbox.x >= 0 && det.bbox.y >= 0 && 
                det.bbox.x + det.bbox.width <= frame.cols && 
                det.bbox.y + det.bbox.height <= frame.rows) {
                
                cv::Mat crop = frame(det.bbox).clone();
                cv::resize(crop, crop, cv::Size(128, 128)); // Resize cho vuông vắn
                
                // Chỉ hiển thị tối đa 3 biển báo lên panel để không bị tràn
                if (ui_x_offset + 128 <= sign_panel.cols) {
                    crop.copyTo(sign_panel(cv::Rect(ui_x_offset, 10, 128, 128)));
                    cv::putText(sign_panel, name, cv::Point(ui_x_offset, 145),
                                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
                    ui_x_offset += 135;
                }
            }
        }

        // Vẽ text Debug lên ảnh Camera
        cv::putText(ui_frame, "Speed: " + std::to_string(current_speed), cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        cv::putText(ui_frame, "Heading: " + std::to_string(heading), cv::Point(10, 60),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

        // Hiển thị các cửa sổ
        cv::imshow("Camera & Signs", ui_frame);
        cv::imshow("Detected Signs", sign_panel);
        if (!mask.empty()) {
            cv::imshow("Road Mask", mask);
        }

        // Lắng nghe phím bấm (1ms) để không chặn vòng lặp 10Hz
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) { // 'q' hoặc ESC
            g_running = 0;
        }
        // ============================================================

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

        // Log debug ra console
        std::cout << "[" << frame_idx << "] "
                  << "Spd=" << current_speed << " "
                  << "PID=" << cmd.angle << " "
                  << "Servo=" << servo_angle << " "
                  << "Head=" << heading << " "
                  << "Signs=" << signs.size() << "\r" << std::flush;
    }

    // Dừng xe an toàn khi thoát
    std::cout << "\n[Main] Dừng xe an toàn...\n";
    motor.SetSpeedCms(0);
    servo.SetAngle(0);
    cv::destroyAllWindows();
    
    return 0;
}

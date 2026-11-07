// ============================================================
// main_run.cpp — UCR2026 Full Integrated (LaneFollower + SignDetector + NavigationBrain)
// ============================================================
#include "hardware/Camera.h"
#include "hardware/Motor.h"
#include "hardware/Servo.h"
#include "common/Logger.h"

#include "control/LaneFollowerV20.h"
#include "SignDetector.h"
#include "NavigationBrain.h"

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <csignal>
#include <algorithm>
#include <stdexcept>
#include <cstdio>
#include <chrono>
#include <thread>
#include <ctime>
#include <iomanip>
#include <sstream>

static const std::string ENGINE_PATH      = "/home/car4/Downloads/road/kga2.engine";
static const std::string SIGN_ENGINE_PATH = "/home/car4/Downloads/sign/huy_sign.engine";

static constexpr float MAX_SPEED_CMS         = 18.0f;
static constexpr int   MAX_STEER_DEG        = 50;
static constexpr bool  INVERT_STEER         = true;
static constexpr bool  SHOW_UI              = true;
static constexpr bool  DA_CHANNEL           = 1; // Giữ nguyên kênh phân loại mặt đường 2-channel
static constexpr float MIN_SPEED_CMS        = 11.0f;
static constexpr int   SIGN_INTERVAL_FRAMES = 5;
static constexpr float PX_TO_CM             = 0.25f;

volatile std::sig_atomic_t g_stopSignal = 0;
static void SignalHandler(int) { g_stopSignal = 1; }

static void CheckCuda(cudaError_t s, const char* op) {
    if (s != cudaSuccess)
        throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(s));
}

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity sev, const char* msg) noexcept override {
        if (sev <= Severity::kWARNING) std::cerr << "[TRT] " << msg << "\n";
    }
};
static TrtLogger gTrtLogger;

// ============================================================
// HÀM ÁNH XẠ GÓC LÁI TỪ THỰC NGHIỆM
// ============================================================
// Dựa trên thực nghiệm: Bánh trái y_L = 0.96x - 6.2 | Bánh phải y_R = 0.71x + 2.5
// Trung bình 2 bánh: y_avg = 0.835x - 1.85 => Điểm 0 thực tế tại x0 = +2.22 độ
static int MapWheelAngleToServoCmd(double desiredWheelDeg, bool invertSteer) {
    // Đảo chiều trước nếu hệ quy ước PID và Servo ngược nhau
    double targetY = invertSteer ? -desiredWheelDeg : desiredWheelDeg;

    // Bù bất đối xứng độ lợi cơ khí (0.96 bên âm vs 0.71 bên dương) + Offset điểm 0 (+2.22)
    // Khi targetY = 0 -> x = +2.22 độ (giúp 2 bánh cân bằng ở -4.07 và +4.08 độ)
    double servoCmdX = 0.0;
    if (targetY >= 0.0) {
        // Phía bánh phải chịu tải chính (gain cơ khí yếu 0.71 -> trung bình hiệu dụng ~0.78)
        servoCmdX = 2.22 + (targetY / 0.78);
    } else {
        // Phía bánh trái chịu tải chính (gain cơ khí mạnh 0.96 -> trung bình hiệu dụng ~0.90)
        servoCmdX = 2.22 + (targetY / 0.90);
    }

    int finalCmd = static_cast<int>(std::lround(servoCmdX));
    return std::clamp(finalCmd, -MAX_STEER_DEG, MAX_STEER_DEG);
}

// ============================================================
// MODEL TENSORRT ĐƯỜNG LÀN
// ============================================================
class TrtModel {
public:
    ~TrtModel() { Cleanup(); }

    bool Init(const std::string& enginePath) {
        try {
            std::cout << "[TRT] Loading Engine: " << enginePath << "\n";
            std::ifstream f(enginePath, std::ios::binary);
            if (!f) throw std::runtime_error("Cannot open engine");
            f.seekg(0, std::ios::end);
            size_t size = f.tellg();
            f.seekg(0, std::ios::beg);
            std::vector<char> data(size);
            f.read(data.data(), size);

            runtime_ = nvinfer1::createInferRuntime(gTrtLogger);
            if (!runtime_) throw std::runtime_error("createInferRuntime failed");

            engine_ = runtime_->deserializeCudaEngine(data.data(), data.size());
            if (!engine_) throw std::runtime_error("deserializeCudaEngine failed");

            context_ = engine_->createExecutionContext();
            if (!context_) throw std::runtime_error("createExecutionContext failed");

            const int nb = engine_->getNbBindings();
            buffers_.assign(nb, nullptr);
            inputIndex_ = -1; outputIndex_ = -1;

            for (int i = 0; i < nb; ++i) {
                const bool isIn = engine_->bindingIsInput(i);
                nvinfer1::Dims d = engine_->getBindingDimensions(i);
                size_t elems = 1;
                for (int j = 0; j < d.nbDims; ++j) elems *= (size_t)d.d[j];

                if (isIn) {
                    inputIndex_ = i;
                    inputH_ = d.d[2]; inputW_ = d.d[3];
                    inputData_.resize(elems);
                } else if (outputIndex_ == -1) {
                    outputIndex_ = i;
                    outputH_ = d.d[2]; outputW_ = d.d[3];
                    outputData_.resize(elems);
                }
                CheckCuda(cudaMalloc(&buffers_[i], elems * sizeof(float)), "cudaMalloc");
            }
            if (inputIndex_ < 0 || outputIndex_ < 0)
                throw std::runtime_error("Missing binding");

            CheckCuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
            std::cout << "[TRT] Road Ready. Input: " << inputW_ << "x" << inputH_ << "\n";
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[Init Error] " << e.what() << "\n";
            Cleanup();
            return false;
        }
    }

    bool Predict(const cv::Mat& frame, cv::Mat& daMask) {
        if (frame.empty() || !context_ || !stream_) return false;
        try {
            cv::Mat resized;
            if (frame.cols != inputW_ || frame.rows != inputH_)
                cv::resize(frame, resized, cv::Size(inputW_, inputH_), 0, 0, cv::INTER_LINEAR);
            else resized = frame;

            cv::Mat rgb;
            cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
            cv::Mat fimg;
            rgb.convertTo(fimg, CV_32FC3, 1.0 / 255.0);

            const size_t plane = (size_t)inputW_ * inputH_;
            const float* ptr = fimg.ptr<float>();
            for (size_t i = 0; i < plane; ++i) {
                inputData_[i]             = ptr[i * 3 + 0];
                inputData_[i + plane]     = ptr[i * 3 + 1];
                inputData_[i + 2 * plane] = ptr[i * 3 + 2];
            }

            CheckCuda(cudaMemcpyAsync(buffers_[inputIndex_], inputData_.data(),
                                      inputData_.size() * sizeof(float),
                                      cudaMemcpyHostToDevice, stream_), "H2D");
            if (!context_->enqueueV2(buffers_.data(), stream_, nullptr))
                throw std::runtime_error("enqueueV2 failed");
            CheckCuda(cudaMemcpyAsync(outputData_.data(), buffers_[outputIndex_],
                                      outputData_.size() * sizeof(float),
                                      cudaMemcpyDeviceToHost, stream_), "D2H");
            CheckCuda(cudaStreamSynchronize(stream_), "sync");

            const size_t outPlane = (size_t)outputW_ * outputH_;
            const float* ch0 = outputData_.data();
            const float* ch1 = outputData_.data() + outPlane;
            const float* da  = (DA_CHANNEL == 1) ? ch1 : ch0;
            const float* bg  = (DA_CHANNEL == 1) ? ch0 : ch1;

            cv::Mat small(outputH_, outputW_, CV_8UC1);
            for (int y = 0; y < outputH_; ++y) {
                uchar* row = small.ptr<uchar>(y);
                for (int x = 0; x < outputW_; ++x) {
                    row[x] = (da[y * outputW_ + x] > bg[y * outputW_ + x]) ? 255 : 0;
                }
            }
            cv::resize(small, daMask, frame.size(), 0, 0, cv::INTER_NEAREST);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[Predict Error] " << e.what() << "\n";
            daMask.release();
            return false;
        }
    }

private:
    void Cleanup() {
        if (stream_) cudaStreamSynchronize(stream_);
        for (void*& b : buffers_) if (b) { cudaFree(b); b = nullptr; }
        buffers_.clear();
        if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
        if (context_) { delete context_; context_ = nullptr; }
        if (engine_) { delete engine_; engine_ = nullptr; }
        if (runtime_) { delete runtime_; runtime_ = nullptr; }
    }
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    cudaStream_t stream_ = nullptr;
    std::vector<void*> buffers_;
    std::vector<float> inputData_, outputData_;
    int inputIndex_ = -1, outputIndex_ = -1;
    int inputW_ = 0, inputH_ = 0, outputW_ = 0, outputH_ = 0;
};

// ============================================================
// BỘ GHI FILE LOG MỞ RỘNG
// ============================================================
class RunFileLogger {
public:
    bool Init() {
        std::time_t now = std::time(nullptr);
        std::tm tmNow{};
        localtime_r(&now, &tmNow);

        std::ostringstream oss;
        oss << "autocontrol_run_"
            << std::put_time(&tmNow, "%Y%m%d_%H%M%S")
            << ".log";
        filename_ = oss.str();

        file_.open(filename_, std::ios::out | std::ios::trunc);
        if (!file_.is_open()) return false;

        file_ << "# AUTO CONTROL & TELEMETRY LOG (UCR2026 LaneFollowerV2.0 + NavigationBrain)\n";
        file_ << "# Columns: Frame,Timestamp_ms,FPS,Speed_cms,Servo_Angle,PID_Angle,"
                 "Near_Error,Far_Error,Dist_Left_px,Dist_Right_px,Active_Sign,Steer_Bias_px,IsValid,CurveAhead\n";
        file_.flush();
        return true;
    }

    void LogStep(long frameIdx, double elapsedMs, double fps, float speed, 
                 int finalAngle, const v20::LaneFollower::Output& out, 
                 const std::string& activeSign, float steerBias) {
        if (!file_.is_open()) return;

        file_ << frameIdx << ","
              << std::fixed << std::setprecision(1) << elapsedMs << ","
              << std::setprecision(1) << fps << ","
              << std::setprecision(2) << speed << ","
              << finalAngle << ","
              << std::setprecision(2) << out.angle << ","
              << std::setprecision(2) << out.near_e << ","
              << std::setprecision(2) << out.far_e << ","
              << std::setprecision(1) << out.dist_to_left << ","
              << std::setprecision(1) << out.dist_to_right << ","
              << activeSign << ","
              << std::setprecision(1) << steerBias << ","
              << (out.valid ? 1 : 0) << ","
              << (out.is_curve_ahead ? 1 : 0) << "\n";

        if (frameIdx % 30 == 0) file_.flush();
    }

    std::string GetFilename() const { return filename_; }

private:
    std::ofstream file_;
    std::string filename_;
};

int main() {
    std::signal(SIGINT, SignalHandler);
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

    TrtModel roadModel;
    if (!roadModel.Init(ENGINE_PATH)) {
        std::cerr << "Road Model init failed\n";
        return -1;
    }

    SignDetector signDetector;
    if (!signDetector.Init(SIGN_ENGINE_PATH)) {
        std::cerr << "Sign Detector init failed\n";
        return -1;
    }

    NavigationBrain navBrain;

    // Cấu hình PID V2.0 tối ưu đường hẹp / 1 làn
    v20::Config cfg;
    cfg.max_speed = MAX_SPEED_CMS;
    cfg.min_speed = MIN_SPEED_CMS;
    cfg.curve_scale = 160.0;
    cfg.min_rows = 5;
    cfg.coverage_threshold = 0.05;

    // --- HIỆU CHỈNH PID CHO TẦN SỐ THẤP (~6-10 FPS) ---
    cfg.kp = 0.28;               // Hạ Kp để tránh vọt lố do trễ frame (~150ms)
    cfg.kd = 0.45;               // Hạ Kd để không khuếch đại nhiễu do sampling
    cfg.derivative_alpha = 0.25; // Lọc đạo hàm mượt hơn
    cfg.output_limit = 28.0;     // Cho phép góc bánh xe mục tiêu tối đa +-28 độ
    cfg.rate_limit = 8.0;        // Giới hạn thay đổi góc bánh xe tối đa 8 độ/frame (~50 độ/giây)
    cfg.emergency_rate_limit = 12.0; 
    cfg.deadband = 2.5;

    // --- HIỆU CHỈNH ĐÓN CUA SỚM & LÀN ---
    cfg.lead_rate_gain = 0.60;   
    cfg.far_delta_clip = 35.0;   
    cfg.early_curve_gain = 0.45; 
    cfg.early_curve_thresh = 10.0; 

    // --- HIỆU CHỈNH WHEEL CLEARANCE & BẢO VỆ BÁNH XE ---
    cfg.narrow_road_width = 190.0;
    cfg.wheel_clearance_px = 110.0; 
    cfg.wheel_guard_gain = 1.10;    

    v20::LaneFollower follower(cfg);

    RunFileLogger runLogger;
    if (runLogger.Init()) {
        std::cout << "[Logger] Writing extended telemetry to: " << runLogger.GetFilename() << "\n";
    }

    cv::Mat frame, mask;
    long frameCount = 0;
    long totalFrames = 0;
    double currentFps = 0.0;
    auto tPrev = std::chrono::high_resolution_clock::now();
    const double appStartTick = static_cast<double>(cv::getTickCount());
    std::vector<SignObject> cachedSigns;

    std::cout << "\n=== UCR2026: Auto Run (Full Integrated) ===\n"
              << "Press 'q' or ESC to quit\n\n";

    while (!g_stopSignal) {
        camera.GetFrame(frame);
        if (frame.empty()) {
            motor.SetSpeedCms(0.0f);
            servo.SetAngle(0);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        totalFrames++;

        if (totalFrames % SIGN_INTERVAL_FRAMES == 0) {
            cachedSigns = signDetector.Detect(frame, 0.70f);
            navBrain.UpdateSign(cachedSigns);
        }

        bool roadOK = roadModel.Predict(frame, mask);
        v20::LaneFollower::Output out;
        float speed = MIN_SPEED_CMS;
        int final_angle = 0;
        NavDecision dec;

        if (roadOK) {
            // Lấy quyết định từ NavigationBrain
            dec = navBrain.Process(mask, follower.speed(), 200.0);

            // Tích hợp steer_bias_px vào TRONG thuật toán PID của LaneFollower
            out = follower.step(mask, dec.steer_bias_px);

            if (dec.emergency_stop) {
                speed = 0.0f;
                final_angle = MapWheelAngleToServoCmd(0.0, INVERT_STEER);
            } else {
                // Ánh xạ góc bánh mong muốn (out.angle) qua phương trình Servo thực nghiệm
                final_angle = MapWheelAngleToServoCmd(out.angle, INVERT_STEER);
                speed = std::clamp(dec.target_speed, MIN_SPEED_CMS, MAX_SPEED_CMS);
            }

            motor.SetSpeedCms(speed);
            servo.SetAngle(final_angle);

        } else {
            motor.SetSpeedCms(0.0f);
            servo.SetAngle(MapWheelAngleToServoCmd(0.0, INVERT_STEER));
            follower.resetTracking();
        }

        frameCount++;
        auto tNow = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(tNow - tPrev).count();
        if (dt > 0.5) {
            currentFps = frameCount / dt;
            frameCount = 0;
            tPrev = tNow;
        }

        const double tCurrentTick = static_cast<double>(cv::getTickCount());
        const double elapsedMs = ((tCurrentTick - appStartTick) / cv::getTickFrequency()) * 1000.0;

        if (roadOK) {
            // Truyền currentFps và totalFrames để ghi log chuẩn xác
            runLogger.LogStep(totalFrames, elapsedMs, currentFps, speed, final_angle, out, dec.active_sign, dec.steer_bias_px);
        }

        if (SHOW_UI) {
            cv::Mat overlay = frame.clone();
            if (roadOK) {
                cv::Mat green(frame.size(), frame.type(), cv::Scalar(0, 255, 0));
                cv::Mat blended;
                cv::addWeighted(frame, 0.65, green, 0.35, 0, blended);
                blended.copyTo(overlay, mask);

                if (out.proc_width > 0 && out.proc_height > 0) {
                    float scale_x = (float)frame.cols / out.proc_width;
                    float scale_y = (float)frame.rows / out.proc_height;

                    for (size_t i = 0; i + 1 < out.left_line_pts.size(); ++i) {
                        cv::Point p1(out.left_line_pts[i].first * scale_x, out.left_line_pts[i].second * scale_y);
                        cv::Point p2(out.left_line_pts[i+1].first * scale_x, out.left_line_pts[i+1].second * scale_y);
                        cv::line(overlay, p1, p2, cv::Scalar(255, 220, 0), 2);
                    }
                    for (size_t i = 0; i + 1 < out.right_line_pts.size(); ++i) {
                        cv::Point p1(out.right_line_pts[i].first * scale_x, out.right_line_pts[i].second * scale_y);
                        cv::Point p2(out.right_line_pts[i+1].first * scale_x, out.right_line_pts[i+1].second * scale_y);
                        cv::line(overlay, p1, p2, cv::Scalar(0, 165, 255), 2);
                    }
                    for (size_t i = 0; i + 1 < out.centerline_pts.size(); ++i) {
                        cv::Point p1(out.centerline_pts[i].first * scale_x, out.centerline_pts[i].second * scale_y);
                        cv::Point p2(out.centerline_pts[i+1].first * scale_x, out.centerline_pts[i+1].second * scale_y);
                        cv::line(overlay, p1, p2, cv::Scalar(0, 0, 255), 2);
                    }

                    if (out.left_near_y > 0 && out.right_near_y > 0) {
                        cv::Point left_lane_pt(out.left_near_x * scale_x, out.left_near_y * scale_y);
                        cv::Point right_lane_pt(out.right_near_x * scale_x, out.right_near_y * scale_y);
                        
                        cv::Point bottom_left(0, frame.rows - 1);
                        cv::Point bottom_right(frame.cols - 1, frame.rows - 1);

                        cv::line(overlay, bottom_left, left_lane_pt, cv::Scalar(255, 0, 255), 3);
                        cv::line(overlay, bottom_right, right_lane_pt, cv::Scalar(255, 0, 255), 3);

                        float left_dist_cm  = std::max(0.0f, static_cast<float>(out.dist_to_left * scale_x) * PX_TO_CM);
                        float right_dist_cm = std::max(0.0f, static_cast<float>(out.dist_to_right * scale_x) * PX_TO_CM);

                        cv::putText(overlay, std::to_string((int)left_dist_cm) + "cm", 
                                    cv::Point(left_lane_pt.x + 10, left_lane_pt.y - 10), 
                                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
                        cv::putText(overlay, std::to_string((int)right_dist_cm) + "cm", 
                                    cv::Point(right_lane_pt.x - 60, right_lane_pt.y - 10), 
                                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
                    }
                }
            }

            signDetector.Draw(overlay, cachedSigns);

            char text1[128], text2[128];
            std::snprintf(text1, sizeof(text1), "FPS: %.1f | Spd: %.1f | Steer: %d | Sign: %s",
                          currentFps, speed, final_angle, dec.active_sign.c_str());
            std::snprintf(text2, sizeof(text2), "L_Dist: %.0fpx | R_Dist: %.0fpx | Bias: %.0fpx",
                          out.dist_to_left, out.dist_to_right, dec.steer_bias_px);

            cv::putText(overlay, text1, {15, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 0), 2);
            cv::putText(overlay, text2, {15, 60}, cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 255), 2);

            cv::imshow("Lane Follower", overlay);
            int key = cv::waitKey(1) & 0xFF;
            if (key == 27 || key == 'q') break;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    motor.SetSpeedCms(0.0f);
    servo.SetAngle(MapWheelAngleToServoCmd(0.0, INVERT_STEER));
    if (SHOW_UI) cv::destroyAllWindows();
    std::cout << "\n[System] Stopped safely\n";
    return 0;
}

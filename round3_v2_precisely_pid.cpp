// test_run.cpp
// Chạy thử nghiệm LaneFollowerV20 với TensorRT engine (TwinLiteNet / Kiet)
#include "hardware/Camera.h"
#include "hardware/Motor.h"
#include "hardware/Servo.h"
#include "common/Logger.h"
#include "control/LaneFollowerV20.h"

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

// ============================================================
// CẤU HÌNH CÓ THỂ CHỈNH SỬA (TUNE TẠI ĐÂY)
// ============================================================
static const std::string ENGINE_PATH = "/home/car4/Downloads/road/kga2.engine"; // Đường dẫn model

// --- Cấu hình phần cứng ---
static constexpr float MAX_SPEED_CMS = 15.0f;   // Tốc độ tối đa cho phép (cm/s)
static constexpr int   MAX_STEER_DEG = 50;      // Góc tối đa của Servo (theo repo là 60)
static constexpr bool  INVERT_STEER  = true;    // Đổi thành true nếu bánh xe rẽ ngược hướng
static constexpr bool  SHOW_UI       = true;    // Bật/tắt màn hình hiển thị
static constexpr bool  DA_CHANNEL    = 1;       // Kênh dự đoán đường (đổi thành 0 nếu mask bị đảo)

// --- Cấu hình Bù trừ Cơ khí ---
static constexpr float STEER_GAIN    = 2.2f;    // Tăng lên 2.5 hoặc 3.0 nếu muốn đánh lái gắt hơn nữa

// --- Cấu hình thuật toán PID (LaneFollower V20) ---
static constexpr float MIN_SPEED_CMS = 5.0f;    // Tốc độ tối thiểu khi vào cua hoặc bị che khuất
static constexpr float CURVE_SCALE   = 150.0f;  // Ngưỡng nhận diện cua (tăng lên để ít bị giảm tốc độ)

// ============================================================
// SIGNAL
// ============================================================
volatile std::sig_atomic_t g_stopSignal = 0;
static void SignalHandler(int) { g_stopSignal = 1; }

// ============================================================
// TENSORRT UTILS
// ============================================================
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
                const char* name = engine_->getBindingName(i);
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
            std::cout << "[TRT] Ready. Input: " << inputW_ << "x" << inputH_ << "\n";
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
            const float* da = (DA_CHANNEL == 1) ? ch1 : ch0;
            const float* bg = (DA_CHANNEL == 1) ? ch0 : ch1;

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
// MAIN
// ============================================================
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

    TrtModel model;
    if (!model.Init(ENGINE_PATH)) {
        std::cerr << "Model init failed\n";
        return -1;
    }

    v20::Config cfg;
    cfg.max_speed = MAX_SPEED_CMS;
    cfg.min_speed = MIN_SPEED_CMS;
    cfg.curve_scale = CURVE_SCALE;
    
    // --- BỎ LOGIC DỪNG KHI MẤT ĐIỂM XA ---
    // Hạ ngưỡng nhận diện để không bị dừng khi chỉ mất mask ở xa (điểm vàng)
    cfg.min_rows = 5;              // Cho phép nhận diện khi chỉ còn 5 hàng (thay vì 20)
    cfg.coverage_threshold = 0.05; // Chỉ cần 5% diện tích mask là vẫn chạy (thay vì 18%)
    
    v20::LaneFollower follower(cfg);

    cv::Mat frame, mask;
    double fps = 0.0;
    int frameCount = 0;
    auto tPrev = std::chrono::high_resolution_clock::now();

    std::cout << "\n=== UCR2026: Auto Run ===\n"
              << "Engine: " << ENGINE_PATH << "\n"
              << "Steer Gain: " << STEER_GAIN << "x\n"
              << "Press 'q' to quit\n\n";

    while (!g_stopSignal) {
        camera.GetFrame(frame);
        if (frame.empty()) {
            motor.SetSpeedCms(0.0f);
            servo.SetAngle(0);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        bool inferOK = model.Predict(frame, mask);
        v20::LaneFollower::Output out;
        
        if (inferOK) {
            out = follower.step(mask);
            
            // 1. Lấy tốc độ
            float speed = std::min((float)out.speed, MAX_SPEED_CMS);
            
            // 2. Khuếch đại góc lái
            float raw_angle = out.angle * STEER_GAIN;
            int final_angle = static_cast<int>(std::lround(raw_angle));
            
            // 3. Giới hạn góc
            final_angle = std::clamp(final_angle, -MAX_STEER_DEG, MAX_STEER_DEG);
            
            // 4. Đảo chiều
            if (INVERT_STEER) final_angle = -final_angle;
            
            // --- BỎ LOGIC DỪNG XE KHI MẤT MASK ---
            // Nếu bị mất mask ở xa (điểm vàng) hoặc bị vật cản che, valid = false.
            // Thay vì phanh gấp, ta vẫn cho xe bò tiếp với tốc độ tối thiểu.
            if (!out.valid && out.coverage > 0.02) {
                speed = MIN_SPEED_CMS;
            }
            
            // 5. Gửi lệnh
            motor.SetSpeedCms(speed);
            servo.SetAngle(final_angle);
            
        } else {
            // Chỉ dừng hẳn khi AI không chạy được (lỗi camera hoặc lỗi model)
            motor.SetSpeedCms(0.0f);
            servo.SetAngle(0);
            follower.resetTracking();
        }

        // Tính FPS
        frameCount++;
        auto tNow = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(tNow - tPrev).count();
        if (dt > 0.5) {
            fps = frameCount / dt;
            frameCount = 0;
            tPrev = tNow;
        }

        // =========================================================
        // VẼ UI & CENTERLINE
        // =========================================================
        if (SHOW_UI) {
            cv::Mat overlay = frame.clone();
            if (inferOK) {
                // Vẽ mask xanh mờ
                cv::Mat green(frame.size(), frame.type(), cv::Scalar(0, 255, 0));
                cv::Mat blended;
                cv::addWeighted(frame, 0.6, green, 0.4, 0, blended);
                blended.copyTo(overlay, mask);
                
                // --- VẼ CENTERLINE (ĐƯỜNG TÂM) ---
                if (!out.centerline_pts.empty() && out.proc_width > 0 && out.proc_height > 0) {
                    float scale_x = (float)frame.cols / out.proc_width;
                    float scale_y = (float)frame.rows / out.proc_height;
                    
                    std::vector<cv::Point> pts;
                    for (const auto& pt : out.centerline_pts) {
                        int cx = static_cast<int>(pt.first * scale_x);
                        int y  = static_cast<int>(pt.second * scale_y);
                        pts.push_back(cv::Point(cx, y));
                    }
                    
                    for (size_t i = 0; i < pts.size() - 1; ++i) {
                        cv::line(overlay, pts[i], pts[i+1], cv::Scalar(0, 0, 255), 2); // Màu đỏ
                    }
                    
                    if (!pts.empty()) {
                        cv::circle(overlay, pts.front(), 4, cv::Scalar(255, 0, 0), -1); // Điểm dưới cùng (Xanh dương)
                        cv::circle(overlay, pts.back(), 4, cv::Scalar(0, 255, 255), -1); // Điểm trên cùng (Vàng)
                    }
                }
            }
            
            char text[128];
            std::snprintf(text, sizeof(text),
                "FPS %.1f | spd %.1f | PID_ang %.1f | Out_ang %d",
                fps, out.speed, out.angle, (int)std::lround(out.angle * STEER_GAIN));
            cv::putText(overlay, text, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        cv::Scalar(0, 255, 0), 2);
                        
            cv::imshow("Lane Follower", overlay);
            int key = cv::waitKey(1) & 0xFF;
            if (key == 27 || key == 'q') break;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    motor.SetSpeedCms(0.0f);
    servo.SetAngle(0);
    if (SHOW_UI) cv::destroyAllWindows();
    std::cout << "\n[System] Stopped safely\n";
    return 0;
}

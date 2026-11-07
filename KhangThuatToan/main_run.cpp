// test_run.cpp
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

static const std::string ENGINE_PATH = "/home/car4/Downloads/road/kga2.engine";

static constexpr float MAX_SPEED_CMS = 18.0f;
static constexpr int   MAX_STEER_DEG = 50;
static constexpr bool  INVERT_STEER  = true;
static constexpr bool  SHOW_UI       = true;
static constexpr bool  DA_CHANNEL    = 1;
static constexpr float STEER_GAIN    = 2.2f;
static constexpr float MIN_SPEED_CMS = 11.0f;
static constexpr float CURVE_SCALE   = 160.0f;

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

        file_ << "# AUTO CONTROL & BOUNDARY TELEMETRY LOG (UCR2026 LaneFollowerV20)\n";
        file_ << "# Columns: Frame,Timestamp_ms,FPS,Speed_cms,Servo_Angle,PID_Angle,"
                 "Near_Error,Far_Error,Eff_Error,Coverage,IsValid,CurveAhead\n";
        file_.flush();
        return true;
    }

    void LogStep(long frameIdx, double elapsedMs, double fps, float speed, 
                 int finalAngle, const v20::LaneFollower::Output& out) {
        if (!file_.is_open()) return;

        file_ << frameIdx << ","
              << std::fixed << std::setprecision(1) << elapsedMs << ","
              << std::setprecision(1) << fps << ","
              << std::setprecision(2) << speed << ","
              << finalAngle << ","
              << std::setprecision(2) << out.angle << ","
              << std::setprecision(2) << out.near_e << ","
              << std::setprecision(2) << out.far_e << ","
              << std::setprecision(2) << out.eff_error << ","
              << std::setprecision(3) << out.coverage << ","
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

    TrtModel model;
    if (!model.Init(ENGINE_PATH)) {
        std::cerr << "Model init failed\n";
        return -1;
    }

    // --- CẤU HÌNH THUẬT TOÁN (ĐO ĐỘ UỐN MÉP ĐƯỜNG ĐÓN CUA SỚM) ---
// CẤU HÌNH THUẬT TOÁN KHOÁ 2 GÓC ĐÁY & BẢO VỆ BÁNH XE ĐƯỜNG HẸP
    v20::Config cfg;
    cfg.max_speed = MAX_SPEED_CMS;
    cfg.min_speed = MIN_SPEED_CMS;
    cfg.curve_scale = 140.0;
    cfg.min_rows = 5;
    cfg.coverage_threshold = 0.05;

    cfg.kp = 0.42;
    cfg.kd = 0.65;
    cfg.derivative_alpha = 0.30;
    cfg.lead_rate_gain = 0.80;
    cfg.far_delta_clip = 45.0;
    cfg.early_curve_gain = 0.55;      // Không để far_e đảo dấu near_e
    cfg.early_curve_thresh = 8.0;
    cfg.narrow_road_width = 190.0;    // Nhận diện sớm khúc đường hẹp trên khung 320px
    cfg.wheel_clearance_px = 120.0;   // Giữ 2 vạch biên cách tâm xe >= 120px (sát 2 góc cuối)
    cfg.wheel_guard_gain = 1.35;      // Lực đẩy mạnh khi 1 vạch tràn vào bánh xe
    cfg.rate_limit = 24.0;
    cfg.deadband = 1.0;

    v20::LaneFollower follower(cfg);

    RunFileLogger runLogger;
    if (runLogger.Init()) {
        std::cout << "[Logger] Writing telemetry to: " << runLogger.GetFilename() << "\n";
    }

    cv::Mat frame, mask;
    double fps = 0.0;
    long frameCount = 0;
    auto tPrev = std::chrono::high_resolution_clock::now();
    const double appStartTick = static_cast<double>(cv::getTickCount());

    std::cout << "\n=== UCR2026: Auto Run (Boundary Aware) ===\n"
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
        float speed = 0.0f;
        int final_angle = 0;

        if (inferOK) {
            out = follower.step(mask);

            speed = std::clamp((float)out.speed, MIN_SPEED_CMS, MAX_SPEED_CMS);

            float raw_angle = out.angle * STEER_GAIN;
            final_angle = static_cast<int>(std::lround(raw_angle));
            final_angle = std::clamp(final_angle, -MAX_STEER_DEG, MAX_STEER_DEG);

            if (INVERT_STEER) final_angle = -final_angle;

            if (!out.valid && out.coverage > 0.02) {
                speed = MIN_SPEED_CMS;
            }

            motor.SetSpeedCms(speed);
            servo.SetAngle(final_angle);

        } else {
            motor.SetSpeedCms(0.0f);
            servo.SetAngle(0);
            follower.resetTracking();
        }

        frameCount++;
        auto tNow = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(tNow - tPrev).count();
        if (dt > 0.5) {
            fps = frameCount / dt;
            frameCount = 0;
            tPrev = tNow;
        }

        const double tCurrentTick = static_cast<double>(cv::getTickCount());
        const double elapsedMs = ((tCurrentTick - appStartTick) / cv::getTickFrequency()) * 1000.0;

        if (inferOK) {
            runLogger.LogStep(frameCount, elapsedMs, fps, speed, final_angle, out);
        }

        // =========================================================
        // VẼ UI: 2 ĐƯỜNG MÉP BIÊN + ĐƯỜNG TIM
        // =========================================================
        if (SHOW_UI) {
            cv::Mat overlay = frame.clone();
            if (inferOK) {
                // Phủ mask xanh lá mờ lên mặt đường
                cv::Mat green(frame.size(), frame.type(), cv::Scalar(0, 255, 0));
                cv::Mat blended;
                cv::addWeighted(frame, 0.65, green, 0.35, 0, blended);
                blended.copyTo(overlay, mask);

                if (out.proc_width > 0 && out.proc_height > 0) {
                    float scale_x = (float)frame.cols / out.proc_width;
                    float scale_y = (float)frame.rows / out.proc_height;

                    // 1. Mép TRÁI (Cyan / Xanh biển nhạt)
                    if (!out.left_line_pts.empty()) {
                        for (size_t i = 0; i + 1 < out.left_line_pts.size(); ++i) {
                            cv::Point p1(out.left_line_pts[i].first * scale_x, out.left_line_pts[i].second * scale_y);
                            cv::Point p2(out.left_line_pts[i+1].first * scale_x, out.left_line_pts[i+1].second * scale_y);
                            cv::line(overlay, p1, p2, cv::Scalar(255, 220, 0), 2);
                        }
                    }

                    // 2. Mép PHẢI (Vàng cam / Magenta)
                    if (!out.right_line_pts.empty()) {
                        for (size_t i = 0; i + 1 < out.right_line_pts.size(); ++i) {
                            cv::Point p1(out.right_line_pts[i].first * scale_x, out.right_line_pts[i].second * scale_y);
                            cv::Point p2(out.right_line_pts[i+1].first * scale_x, out.right_line_pts[i+1].second * scale_y);
                            cv::line(overlay, p1, p2, cv::Scalar(0, 165, 255), 2);
                        }
                    }

                    // 3. TIM ĐƯỜNG (Màu Đỏ)
                    if (!out.centerline_pts.empty()) {
                        for (size_t i = 0; i + 1 < out.centerline_pts.size(); ++i) {
                            cv::Point p1(out.centerline_pts[i].first * scale_x, out.centerline_pts[i].second * scale_y);
                            cv::Point p2(out.centerline_pts[i+1].first * scale_x, out.centerline_pts[i+1].second * scale_y);
                            cv::line(overlay, p1, p2, cv::Scalar(0, 0, 255), 2);
                        }
                    }
                }
            }

            char text1[128], text2[128];
            std::snprintf(text1, sizeof(text1),
                "FPS: %.1f | Spd: %.1f | Steer: %d",
                fps, speed, final_angle);

            std::snprintf(text2, sizeof(text2),
                "Near_E: %.1f | Far_E: %.1f | Curve Ahead: %s",
                out.near_e, out.far_e, out.is_curve_ahead ? "YES (EARLY TURN)" : "NO");

            cv::putText(overlay, text1, {15, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 0), 2);
            cv::putText(overlay, text2, {15, 60}, cv::FONT_HERSHEY_SIMPLEX, 0.65, 
                        out.is_curve_ahead ? cv::Scalar(0, 165, 255) : cv::Scalar(0, 255, 255), 2);

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
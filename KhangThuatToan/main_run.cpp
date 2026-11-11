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

static constexpr float MAX_SPEED_CMS         = 30.0f;
static constexpr int   MAX_STEER_DEG        = 50;
static constexpr bool  INVERT_STEER         = true;
static constexpr bool  SHOW_UI              = true;
static constexpr bool  DA_CHANNEL           = 1; 
static constexpr float MIN_SPEED_CMS        = 22.0f;
static constexpr int   SIGN_INTERVAL_FRAMES = 5;
static constexpr int   SIGN_BLOCK_FRAMES    = 40; 

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

static int MapWheelAngleToServoCmd(double desiredWheelDeg, bool invertSteer) {
    double servoCmd = invertSteer ? -desiredWheelDeg : desiredWheelDeg;

    return std::clamp(
        static_cast<int>(std::lround(servoCmd)),
        -MAX_STEER_DEG,
        MAX_STEER_DEG
    );
}

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

        file_ << "# AUTO CONTROL & TELEMETRY LOG (UCR2026 Enhanced Debug)\n";
        file_ << "# Columns: Frame,Timestamp_ms,FPS,Speed_cms,Servo_Angle,PID_Angle,"
                 "Near_Error,Mid_Error,Far_Error,Active_Sign,Turn_State,Turn_Direction,"
                 "Scanline_Dev,Junction_Opened,Straight_Frames,Turn_Elapsed,"
                 "Turn_Bias,Steer_Floor_Deg,P0,P1,P2,Left_Pts,Right_Pts,IsValid\n";
        file_.flush();
        return true;
    }

    void LogStep(
        long frameIdx,
        double elapsedMs,
        double fps,
        float speed,
        int finalAngle,
        const v20::LaneFollower::Output& out,
        const std::string& activeSign,

        int turnState,
        const std::string& turnDirection,
        double scanlineDev,
        bool junctionOpened,
        int straightFrames,
        float turnElapsed,

        float turnBias,
        float steerFloorDeg,

        const cv::Point& p0,
        const cv::Point& p1,
        const cv::Point& p2,

        const std::vector<cv::Point>& leftPts,
        const std::vector<cv::Point>& rightPts
    ) {
        if (!file_.is_open()) return;

        auto writePoints = [&](const std::vector<cv::Point>& pts) {
            std::ostringstream ss;
            for (size_t i = 0; i < pts.size(); ++i) {
                if (i > 0) ss << ";";
                ss << pts[i].x << ":" << pts[i].y;
            }
            return ss.str();
        };

        auto formatPoint = [](const cv::Point& p) {
            return std::to_string(p.x) + ":" + std::to_string(p.y);
        };

        file_
            << frameIdx << ","
            << std::fixed << std::setprecision(1) << elapsedMs << ","
            << fps << ","
            << std::setprecision(2) << speed << ","
            << finalAngle << ","
            << out.angle << ","
            << out.near_e << ","
            << out.mid_e << ","
            << out.far_e << ","
            << activeSign << ","
            << turnState << ","
            << turnDirection << ","
            << std::setprecision(2) << scanlineDev << ","
            << (junctionOpened ? 1 : 0) << ","
            << straightFrames << ","
            << turnElapsed << ","
            << turnBias << ","
            << steerFloorDeg << ","
            << formatPoint(p0) << ","
            << formatPoint(p1) << ","
            << formatPoint(p2) << ","
            << "\"" << writePoints(leftPts) << "\","
            << "\"" << writePoints(rightPts) << "\","
            << (out.valid ? 1 : 0)
            << "\n";

        if (frameIdx % 30 == 0)
            file_.flush();
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

    v20::Config cfg;
    cfg.max_speed = MAX_SPEED_CMS;
    cfg.min_speed = MIN_SPEED_CMS;
    cfg.curve_scale = 140.0;
    cfg.min_rows = 5;
    cfg.coverage_threshold = 0.05;

    cfg.kp = 0.40;               
    cfg.kd = 0.42;               
    cfg.derivative_alpha = 0.25; 
    cfg.output_limit = 50.0;     
    cfg.emergency_output_limit = 50.0;
    cfg.rate_limit = 10.0;        
    cfg.emergency_rate_limit = 18.0; 
    cfg.deadband = 0.5;

    v20::LaneFollower follower(cfg);
    v20::LaneTurnController turn_controller;
    v20::CornerAssist corner_assist;
    std::string handled_sign = "NONE";
    v20::LaneFollower::Output previous_out;

    RunFileLogger runLogger;
    if (runLogger.Init()) {
        std::cout << "[Logger] Writing extended telemetry to: " << runLogger.GetFilename() << "\n";
    }

    cv::Mat frame, mask;
    long frameCount = 0;
    long totalFrames = 0;
    double currentFps = 0.0;
    auto tPrev = std::chrono::high_resolution_clock::now();
    auto tPrevLoop = std::chrono::high_resolution_clock::now();
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

        auto tNowLoop = std::chrono::high_resolution_clock::now();
        double dt_loop = std::chrono::duration<double>(tNowLoop - tPrevLoop).count();
        tPrevLoop = tNowLoop;
        if (dt_loop <= 0.0) dt_loop = 1.0 / 30.0;

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
        v20::LaneTurnController::TurnResult turn_res{v20::LaneTurnController::State::IDLE, 0.0, "NONE"};
        double applied_floor = 0.0;
        // --- UI DATA EXPORT VARIABLES ---
        std::vector<cv::Point> ui_left_pts, ui_right_pts;
        int ui_y1 = 0, ui_y2 = 0;

        if (roadOK) {
            dec = navBrain.Process(mask, follower.speed());

            int h = mask.rows, w = mask.cols;
            int y1 = static_cast<int>(h * 0.45);
            int y2 = static_cast<int>(h * 0.65);
            int y_bottom = h - 1;

            auto get_mask_boundaries = [&](int y_row, int& lx, int& rx) {
                lx = -1; rx = -1;
                if (y_row < 0 || y_row >= h) return;
                const uchar* row = mask.ptr<uchar>(y_row);
                int start = -1; int best_len = -1;
                for (int x = 0; x <= w; x++) {
                    bool on = (x < w) && (row[x] > 0);
                    if (on && start < 0) start = x;
                    if (!on && start >= 0) {
                        int len = x - start;
                        if (len > best_len) { best_len = len; lx = start; rx = x - 1; }
                        start = -1;
                    }
                }
            };

            auto find_edge_corner_points = [&](int start_y, int end_y, bool is_left) {
                std::vector<cv::Point> pts;
                bool touching = false;
                for (int y = start_y; y <= end_y; y += 2) {
                    int lx, rx;
                    get_mask_boundaries(y, lx, rx);
                    bool is_touching = is_left ? (lx != -1 && lx <= 2) : (rx != -1 && rx >= w - 3);
                    if (is_touching && !touching) {
                        touching = true;
                        pts.push_back(cv::Point(is_left ? 0 : w - 1, y));
                    } else if (!is_touching && touching) {
                        touching = false;
                    }
                }
                return pts;
            };

            int lx1, rx1, lx2, rx2, lx_bot, rx_bot;
            get_mask_boundaries(y1, lx1, rx1);
            get_mask_boundaries(y2, lx2, rx2);
            get_mask_boundaries(y_bottom, lx_bot, rx_bot);

            std::vector<cv::Point> left_pts, right_pts;
            if (lx1 != -1) left_pts.push_back(cv::Point(lx1, y1));
            for (auto p : find_edge_corner_points(y1, y_bottom, true)) left_pts.push_back(p);
            if (lx2 != -1) left_pts.push_back(cv::Point(lx2, y2));
            if (lx_bot != -1) left_pts.push_back(cv::Point(lx_bot, y_bottom));

            if (rx1 != -1) right_pts.push_back(cv::Point(rx1, y1));
            for (auto p : find_edge_corner_points(y1, y_bottom, false)) right_pts.push_back(p);
            if (rx2 != -1) right_pts.push_back(cv::Point(rx2, y2));
            if (rx_bot != -1) right_pts.push_back(cv::Point(rx_bot, y_bottom));

            std::sort(left_pts.begin(), left_pts.end(), [](const cv::Point& a, const cv::Point& b){ return a.y < b.y; });
            std::sort(right_pts.begin(), right_pts.end(), [](const cv::Point& a, const cv::Point& b){ return a.y < b.y; });
            left_pts.erase(std::unique(left_pts.begin(), left_pts.end()), left_pts.end());
            right_pts.erase(std::unique(right_pts.begin(), right_pts.end()), right_pts.end());

            // --- EXPORT DATA LÊN UI ĐỂ VẼ ---
            ui_left_pts = left_pts;
            ui_right_pts = right_pts;
            ui_y1 = y1; 
            ui_y2 = y2;

            // Không nhận lệnh rẽ/roundabout khi đang ôm cua 90° (biển vẫn được latch, xử lý ngay sau khi ôm cua xong)
            if (turn_controller.state == v20::LaneTurnController::State::IDLE && !corner_assist.active()) {
                if (dec.active_sign == "roundabout") {
                    turn_controller.start_turn("RIGHT", true);
                    handled_sign = dec.active_sign;
                    navBrain.ClearLatchedSign();
                }
                else if (dec.action == NavAction::FORCE_LEFT) {
                    turn_controller.start_turn("LEFT", false);
                    handled_sign = dec.active_sign;
                    navBrain.ClearLatchedSign();
                }
                else if (dec.action == NavAction::FORCE_RIGHT) {
                    turn_controller.start_turn("RIGHT", false);
                    handled_sign = dec.active_sign;
                    navBrain.ClearLatchedSign();
                }
            }
            turn_res = turn_controller.update(
                left_pts, right_pts, dt_loop, previous_out.valid,
                lx1, rx1, lx2, rx2, w,
                (dec.active_sign != "NONE")
            );
            // Góc sàn: ưu tiên TurnController; khi IDLE thì CornerAssist lo cua 90° không biển
            applied_floor = turn_res.steer_floor_deg;
            if (turn_controller.state == v20::LaneTurnController::State::IDLE) {
                const double cf = corner_assist.update(lx1, rx1, lx2, rx2, lx_bot, rx_bot, w, dt_loop);
                if (std::abs(cf) > std::abs(applied_floor)) applied_floor = cf;
            } else {
                corner_assist.reset();
            }
            out = follower.step(
                mask,
                turn_res.steer_bias_px,
                turn_res.lock_opposite_err,
                applied_floor,
                turn_res.steer_cap_deg
            );
            previous_out = out;
            if (turn_res.state == v20::LaneTurnController::State::DONE) {
                navBrain.ClearLatchedSign();
                if (handled_sign != "NONE") navBrain.BlockSign(handled_sign, SIGN_BLOCK_FRAMES);
                handled_sign = "NONE";
            }

            if (dec.emergency_stop) {
                speed = 0.0f;
                final_angle = MapWheelAngleToServoCmd(0.0, INVERT_STEER);
            } else {
                final_angle = MapWheelAngleToServoCmd(
                    out.angle,
                    INVERT_STEER
                );

                speed = std::clamp(
                    dec.target_speed,
                    MIN_SPEED_CMS,
                    MAX_SPEED_CMS
                );
            }

            motor.SetSpeedCms(speed);
            servo.SetAngle(final_angle);

        } else {
            previous_out = v20::LaneFollower::Output{};
            motor.SetSpeedCms(0.0f);
            servo.SetAngle(MapWheelAngleToServoCmd(0.0, INVERT_STEER));
            follower.resetTracking();
        }

        frameCount++;
        auto tNow = std::chrono::high_resolution_clock::now();
        double dt_fps = std::chrono::duration<double>(tNow - tPrev).count();
        if (dt_fps > 0.5) {
            currentFps = frameCount / dt_fps;
            frameCount = 0;
            tPrev = tNow;
        }

        const double tCurrentTick = static_cast<double>(cv::getTickCount());
        const double elapsedMs = ((tCurrentTick - appStartTick) / cv::getTickFrequency()) * 1000.0;

        if (roadOK) {
            runLogger.LogStep(
                totalFrames,
                elapsedMs,
                currentFps,
                speed,
                final_angle,
                out,
                dec.active_sign,

                static_cast<int>(turn_res.state),
                turn_res.direction,
                turn_res.dev,
                turn_res.junction_opened,
                turn_res.straight_frames,
                static_cast<float>(turn_controller.get_turn_elapsed()),

                static_cast<float>(turn_res.steer_bias_px),
                static_cast<float>(applied_floor),   

                turn_res.p0,
                turn_res.p1,
                turn_res.p2,

                ui_left_pts,
                ui_right_pts
            );
            
            // In terminal debug ngắn gọn
            std::cout << "[F " << totalFrames << "] "
                    << "Dev: " << std::fixed << std::setprecision(1) << turn_res.dev << "px | "
                    << "Junc: " << (turn_res.junction_opened ? "OPEN" : "WAIT") << " | "
                    << "StrFrames: " << turn_res.straight_frames << "/3 | "
                    << "P0(" << turn_res.p0.x << "," << turn_res.p0.y << ") "
                    << "P1(" << turn_res.p1.x << "," << turn_res.p1.y << ") "
                    << "P2(" << turn_res.p2.x << "," << turn_res.p2.y << ")\n";
        }

        if (SHOW_UI) {
            cv::Mat overlay = frame.clone();
            if (roadOK) {
                cv::Mat green(frame.size(), frame.type(), cv::Scalar(0, 255, 0));
                cv::Mat blended;
                cv::addWeighted(frame, 0.65, green, 0.35, 0, blended);
                blended.copyTo(overlay, mask);

                // ============================================================
                // A. VẼ VÙNG QUÉT ĐO NGÃ RẼ (JUNCTION MEASUREMENT ZONE)
                // ============================================================
                // Vẽ hộp trong suốt bao phủ toàn bộ dải quét ngã rẽ từ y = 160 đến 210
                cv::Mat junc_box_layer = overlay.clone();
                cv::Rect junc_zone(0, 160, frame.cols, 50); // Bề rộng 50px từ y=160 đến 210
                cv::rectangle(junc_box_layer, junc_zone, cv::Scalar(255, 255, 0), -1);
                cv::addWeighted(junc_box_layer, 0.20, overlay, 0.80, 0, overlay); // Độ mờ 20%
                
                // Vẽ 2 đường viền vàng neon giới hạn vùng quét
                cv::line(overlay, cv::Point(0, 160), cv::Point(frame.cols - 1, 160), cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
                cv::line(overlay, cv::Point(0, 210), cv::Point(frame.cols - 1, 210), cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
                cv::putText(overlay, "JUNCTION SCAN ZONE (Y: 160-210)", cv::Point(15, 180),
                            cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
                
                // Khi miệng ngã rẽ thực sự mở ra
                if (turn_res.junction_opened) {
                    std::string junc_text = (turn_res.direction == "LEFT") ? ">> LEFT JUNCTION OPENED <<" : ">> RIGHT JUNCTION OPENED <<";
                    cv::putText(overlay, junc_text, cv::Point(static_cast<int>(frame.cols * 0.22), 190),
                                cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
                }

                // 1. DRAW HORIZONTAL SCANLINES (Red)
                if (ui_y1 > 0 && ui_y2 > 0) {
                    cv::line(overlay, cv::Point(0, ui_y1), cv::Point(frame.cols - 1, ui_y1), cv::Scalar(0, 0, 255), 1);
                    cv::line(overlay, cv::Point(0, ui_y2), cv::Point(frame.cols - 1, ui_y2), cv::Scalar(0, 0, 255), 1);
                }

                // 2. DRAW LEFT AND RIGHT BOUNDARIES (Blue)
                if (ui_left_pts.size() >= 2) {
                    for (size_t i = 0; i < ui_left_pts.size() - 1; i++) {
                        cv::line(overlay, ui_left_pts[i], ui_left_pts[i + 1], cv::Scalar(255, 0, 0), 2);
                    }
                }
                if (ui_right_pts.size() >= 2) {
                    for (size_t i = 0; i < ui_right_pts.size() - 1; i++) {
                        cv::line(overlay, ui_right_pts[i], ui_right_pts[i + 1], cv::Scalar(255, 0, 0), 2);
                    }
                }

                // 3. DRAW CORNER POINTS (Green with Black border)
                for (const auto& p : ui_left_pts) {
                    cv::circle(overlay, p, 4, cv::Scalar(0, 255, 0), -1);
                    cv::circle(overlay, p, 4, cv::Scalar(0, 0, 0), 1);
                }
                for (const auto& p : ui_right_pts) {
                    cv::circle(overlay, p, 4, cv::Scalar(0, 255, 0), -1);
                    cv::circle(overlay, p, 4, cv::Scalar(0, 0, 0), 1);
                }
                // ============================================================
                // BỔ SUNG: VẼ 3 ĐIỂM P0, P1, P2 ĐO ĐỘ CONG SCANLINE
                // ============================================================
                if (turn_res.p0.x >= 0) {
                    cv::circle(overlay, turn_res.p0, 6, cv::Scalar(0, 165, 255), -1); // Cam
                    cv::putText(overlay, "P0", turn_res.p0 + cv::Point(6, -4), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 165, 255), 2);
                }
                if (turn_res.p1.x >= 0) {
                    cv::circle(overlay, turn_res.p1, 6, cv::Scalar(255, 0, 255), -1); // Tím
                    cv::putText(overlay, "P1", turn_res.p1 + cv::Point(6, -4), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 0, 255), 2);
                }
                if (turn_res.p2.x >= 0) {
                    cv::circle(overlay, turn_res.p2, 6, cv::Scalar(255, 255, 0), -1); // Cyan
                    cv::putText(overlay, "P2", turn_res.p2 + cv::Point(6, -4), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 0), 2);
                }
                if (turn_res.p0.x >= 0 && turn_res.p1.x >= 0 && turn_res.p2.x >= 0) {
                    cv::line(overlay, turn_res.p0, turn_res.p2, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
                }

                // 4. DRAW LANE FOLLOWER CENTER POINTS (Yellow)
                if (out.proc_width > 0 && out.proc_height > 0) {
                    float scale_x = (float)frame.cols / out.proc_width;
                    float scale_y = (float)frame.rows / out.proc_height;
                    for (size_t i = 0; i < out.centerline_pts.size(); i += 8) {
                        int cx = static_cast<int>(out.centerline_pts[i].first * scale_x);
                        int cy = static_cast<int>(out.centerline_pts[i].second * scale_y);
                        cv::circle(overlay, cv::Point(cx, cy), 2, cv::Scalar(0, 255, 255), -1);
                    }
                }

                // 5. DESIRED PATH VISUALIZATION (Magenta Bezier Curve)
                int start_x = frame.cols / 2;
                int start_y = frame.rows - 1;
                int end_y = static_cast<int>(frame.rows * 0.45);
                
                float max_angle = 25.0f; 
                float steer_norm = std::clamp(static_cast<float>(out.angle) / max_angle, -1.0f, 1.0f);
                float lateral_offset = steer_norm * frame.cols * 0.35f;

                int end_x = std::clamp(static_cast<int>(start_x + lateral_offset), 0, frame.cols - 1);
                int control_x = static_cast<int>(start_x + lateral_offset * 0.45f);
                int control_y = static_cast<int>(frame.rows * 0.78f);

                std::vector<cv::Point> path_points;
                for (int i = 0; i <= 50; i++) {
                    float t = i / 50.0f;
                    float x = std::pow(1 - t, 2) * start_x + 2 * (1 - t) * t * control_x + std::pow(t, 2) * end_x;
                    float y = std::pow(1 - t, 2) * start_y + 2 * (1 - t) * t * control_y + std::pow(t, 2) * end_y;
                    path_points.push_back(cv::Point(static_cast<int>(x), static_cast<int>(y)));
                }

                std::vector<std::vector<cv::Point>> paths = {path_points};
                cv::polylines(overlay, paths, false, cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
                cv::circle(overlay, cv::Point(end_x, end_y), 4, cv::Scalar(0, 0, 255), -1);
                
                auto action_to_str = [](NavAction a) -> const char* {
                    switch (a) {
                        case NavAction::FOLLOW_LANE: return "FOLLOW_LANE";
                        case NavAction::FORCE_LEFT:  return "FORCE_LEFT";
                        case NavAction::FORCE_RIGHT: return "FORCE_RIGHT";
                        case NavAction::STOP_PARKING:        return "STOP";
                        default:                     return "UNKNOWN";
                    }
                };

                auto state_to_str = [](v20::LaneTurnController::State s) -> const char* {
                    using S = v20::LaneTurnController::State;
                    switch (s) {
                        case S::IDLE:          return "IDLE";
                        case S::WAIT_CURVE:    return "WAIT_CURVE";
                        case S::WAIT_STRAIGHT: return "WAIT_STRAIGHT";
                        case S::EXECUTE:       return "EXECUTE";
                        case S::SETTLE:        return "SETTLE";
                        case S::DONE:          return "DONE";
                        default:               return "UNKNOWN";
                    }
                };
                
                // --- THÊM TEXT MỚI BẰNG LAMBDA ---
                auto draw_text = [](cv::Mat& img, const std::string& text, cv::Point pos, cv::Scalar color) {
                    cv::putText(img, text, pos, cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv::LINE_AA);
                };

                draw_text(overlay, cv::format("Frame: %ld", totalFrames), cv::Point(15, 25), cv::Scalar(255, 255, 255));
                draw_text(overlay, cv::format("Action: %s", action_to_str(dec.action)), cv::Point(15, 50), cv::Scalar(0, 255, 255));
                draw_text(overlay, cv::format("Sign: %s", dec.active_sign.c_str()), cv::Point(15, 75), cv::Scalar(0, 255, 0));
                draw_text(overlay, cv::format("Speed: %.1f", speed), cv::Point(15, 100), cv::Scalar(255, 255, 255));
                
                draw_text(overlay, cv::format("Turn State: %s", state_to_str(turn_res.state)), cv::Point(15, 125), cv::Scalar(0, 165, 255));
                draw_text(overlay, cv::format("Turn Dir: %s", turn_res.direction.c_str()), cv::Point(15, 150), cv::Scalar(0, 165, 255)); 
                
                draw_text(overlay, cv::format("Turn Bias: %.1fpx", turn_res.steer_bias_px), cv::Point(15, 175), cv::Scalar(255, 255, 0));
                draw_text(overlay, cv::format("Turn Floor: %.1fdeg", applied_floor), cv::Point(15, 200), cv::Scalar(255, 255, 0)); // <--- ĐỔI DÒNG 749 NÀY                
                draw_text(overlay, cv::format("Near E: %.1f | Far E: %.1f", out.near_e, out.far_e), cv::Point(15, 225), cv::Scalar(255, 255, 255));
                draw_text(overlay, cv::format("Lane Angle: %.2f", out.angle), cv::Point(15, 250), cv::Scalar(255, 0, 255));
            }

            signDetector.Draw(overlay, cachedSigns);

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

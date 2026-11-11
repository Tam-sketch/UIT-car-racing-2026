// ============================================================
// main.cpp — UCR 2026: v21 Lane + turn controller + SignVoter
// Đã fix: active_block_ticks đúng đơn vị, xoá dead code
// ============================================================
#include "hardware/Camera.h"
#include "hardware/Motor.h"
#include "hardware/Servo.h"
#include "common/Logger.h"

#include "control/LaneFollowerV21.h"
#include "control/TurnController.h"
#include "control/SignVoter.h"
#include "SignDetector.h"
#include "NavTypes.h"
#include "config/RobotConfig.h"

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

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
// Road segmentation — TensorRT inline
// ============================================================
class TrtModel {
public:
    ~TrtModel() { Cleanup(); }

    bool Init(const std::string& enginePath) {
        try {
            std::ifstream f(enginePath, std::ios::binary);
            if (!f) throw std::runtime_error("Cannot open engine: " + enginePath);
            f.seekg(0, std::ios::end);
            size_t size = (size_t)f.tellg();
            f.seekg(0, std::ios::beg);
            std::vector<char> data(size);
            f.read(data.data(), size);

            runtime_ = nvinfer1::createInferRuntime(gTrtLogger);
            if (!runtime_) throw std::runtime_error("createInferRuntime failed");
            engine_  = runtime_->deserializeCudaEngine(data.data(), data.size());
            if (!engine_)  throw std::runtime_error("deserializeCudaEngine failed");
            context_ = engine_->createExecutionContext();
            if (!context_) throw std::runtime_error("createExecutionContext failed");

            const int nb = engine_->getNbBindings();
            buffers_.assign(nb, nullptr);
            inputIndex_ = -1; outputIndex_ = -1;
            for (int i = 0; i < nb; ++i) {
                bool isIn = engine_->bindingIsInput(i);
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
            std::cout << "[TRT Road] Ready. Input " << inputW_ << "x" << inputH_ << "\n";
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[Init Road Error] " << e.what() << "\n";
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

            cv::Mat rgb, fimg;
            cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
            rgb.convertTo(fimg, CV_32FC3, 1.0 / 255.0);

            const size_t plane = (size_t)inputW_ * inputH_;
            const float* ptr = fimg.ptr<float>();
            for (size_t i = 0; i < plane; ++i) {
                inputData_[i]             = ptr[i*3 + 0];
                inputData_[i + plane]     = ptr[i*3 + 1];
                inputData_[i + 2*plane]   = ptr[i*3 + 2];
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
            const float* da  = (ucr::config::ROAD_DA_CHANNEL == 1) ? ch1 : ch0;
            const float* bg  = (ucr::config::ROAD_DA_CHANNEL == 1) ? ch0 : ch1;

            cv::Mat small(outputH_, outputW_, CV_8UC1);
            for (int y = 0; y < outputH_; ++y) {
                uchar* row = small.ptr<uchar>(y);
                for (int x = 0; x < outputW_; ++x)
                    row[x] = (da[y*outputW_ + x] - bg[y*outputW_ + x]
                              > ucr::config::ROAD_CONF_MARGIN) ? 255 : 0;
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
        if (stream_)  { cudaStreamDestroy(stream_); stream_ = nullptr; }
        if (context_) { delete context_; context_ = nullptr; }
        if (engine_)  { delete engine_;  engine_  = nullptr; }
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
// Traffic light HSV
// ============================================================
enum class LightColor { NONE, RED, YELLOW, GREEN };

static LightColor ClassifyTrafficLight(const cv::Mat& bgr, const cv::Rect& bbox) {
    using namespace ucr::config::tl;
    const float cx = bbox.x + bbox.width * 0.5f;
    const float cy = bbox.y + bbox.height * 0.5f;
    const float w  = bbox.width  * ROI_EXPAND;
    const float h  = bbox.height * ROI_EXPAND;
    const int x1 = std::max(0, (int)(cx - w * 0.5f));
    const int y1 = std::max(0, (int)(cy - h * 0.5f));
    const int x2 = std::min(bgr.cols, (int)(cx + w * 0.5f));
    const int y2 = std::min(bgr.rows, (int)(cy + h * 0.5f));
    if (x2 - x1 < 4 || y2 - y1 < 4) return LightColor::NONE;

    cv::Mat hsv;
    cv::cvtColor(bgr(cv::Rect(x1, y1, x2 - x1, y2 - y1)), hsv, cv::COLOR_BGR2HSV);

    cv::Mat m1, m2, mRed;
    cv::inRange(hsv, {RED_H1_LO, S_MIN, V_MIN}, {RED_H1_HI, 255, 255}, m1);
    cv::inRange(hsv, {RED_H2_LO, S_MIN, V_MIN}, {RED_H2_HI, 255, 255}, m2);
    cv::bitwise_or(m1, m2, mRed);
    const int nRed = cv::countNonZero(mRed);

    cv::Mat mYel;
    cv::inRange(hsv, {YEL_H_LO, S_MIN, V_MIN}, {YEL_H_HI, 255, 255}, mYel);
    const int nYel = cv::countNonZero(mYel);

    cv::Mat mGrn;
    cv::inRange(hsv, {GRN_H_LO, S_MIN, V_MIN}, {GRN_H_HI, 255, 255}, mGrn);
    const int nGrn = cv::countNonZero(mGrn);

    if (nRed >= MIN_LIT_PIXELS && nRed > nYel && nRed > nGrn) return LightColor::RED;
    if (nYel >= MIN_LIT_PIXELS && nYel > nRed && nYel > nGrn) return LightColor::YELLOW;
    if (nGrn >= MIN_LIT_PIXELS && nGrn > nRed && nGrn > nYel) return LightColor::GREEN;
    return LightColor::NONE;
}

static int MapWheelAngleToServoCmd(double wheel_deg) {
    float real_deg = -static_cast<float>(wheel_deg);
    real_deg = std::clamp(real_deg,
                          -ucr::config::MAX_REAL_ANGLE_DEG,
                           ucr::config::MAX_REAL_ANGLE_DEG);
    float servo = ucr::config::RealAngleToServo(real_deg);
    servo = std::clamp(servo,
                       -static_cast<float>(ucr::config::MAX_SERVO_CMD),
                        static_cast<float>(ucr::config::MAX_SERVO_CMD));
    return (int)std::lround(servo);
}

// ============================================================
int main() {
    std::signal(SIGINT,  SignalHandler);
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
    if (!roadModel.Init(ucr::config::ROAD_ENGINE_PATH)) {
        std::cerr << "Road Model init failed\n";
        return -1;
    }

    SignDetector signDetector;
    bool signReady = signDetector.Init(ucr::config::SIGN_ENGINE_PATH);
    if (!signReady)
        std::cerr << "[Main] SignDetector init failed — turns DISABLED\n";

    // ───────────────────────────────────────────────────────
    // SignVoter thay cho NavigationBrain
    // ───────────────────────────────────────────────────────
    ucr::control::SignVoter voter(
        ucr::config::SIGN_VOTER_ACTIVATE_FRAMES,
        ucr::config::SIGN_VOTER_TIMEOUT_FRAMES);

    // ✅ FIX: quy đổi timeout (đơn vị Update của voter) sang frame tick
    // Voter.Update() gọi mỗi SIGN_DETECT_EVERY tick → 30 Update = 60 tick
    const int SIGN_BLOCK_TICKS =
        ucr::config::SIGN_VOTER_TIMEOUT_FRAMES * ucr::config::SIGN_DETECT_EVERY;

    std::string active_turn_sign    = "NONE";
    int         active_block_ticks  = 0;

    // ───────────────────────────────────────────────────────
    v21::Config cfg;
    cfg.max_speed          = ucr::config::MAX_SPEED_CMS;
    cfg.min_speed          = ucr::config::MIN_SPEED_CMS;
    cfg.curve_scale        = ucr::config::CURVE_SCALE;
    cfg.post_pid_gain      = ucr::config::POST_PID_GAIN;
    cfg.min_rows           = ucr::config::MIN_ROWS;
    cfg.coverage_threshold = ucr::config::COVERAGE_THRESHOLD;
    cfg.rb_bias_ema        = ucr::config::RB_BIAS_EMA;

    v21::LaneFollower follower(cfg);
    turn::LaneTurnController turn_controller;
    turn::CornerAssist        corner_assist;
    v21::LaneFollower::Output previous_out;

    cv::Mat frame, mask;
    std::vector<SignObject> cachedSigns;

    long totalFrames = 0, fpsFrames = 0;
    double currentFps = 0.0;
    auto tPrev     = std::chrono::high_resolution_clock::now();
    auto tPrevLoop = std::chrono::high_resolution_clock::now();

    float applied_speed = 0.0f;
    int   lastRedSeenTick = -1;
    uint32_t tickIdx = 0;
    auto next_control = std::chrono::steady_clock::now();

    std::cout << "\n=== UCR2026 (v21 lane + turn + SignVoter) ===\n"
              << "Road: " << ucr::config::ROAD_ENGINE_PATH << "\n"
              << "Sign: " << ucr::config::SIGN_ENGINE_PATH
              << " (" << (signReady ? "OK" : "FAILED") << ")\n"
              << "Speed: [" << ucr::config::MIN_SPEED_CMS
              << ", " << ucr::config::MAX_SPEED_CMS << "] cm/s\n"
              << "Voter: activate=" << ucr::config::SIGN_VOTER_ACTIVATE_FRAMES
              << " timeout=" << ucr::config::SIGN_VOTER_TIMEOUT_FRAMES
              << " (block " << SIGN_BLOCK_TICKS << " ticks)\n"
              << "Press 'q' / ESC to quit\n\n";

    while (!g_stopSignal) {
        camera.GetFrame(frame);
        if (frame.empty()) {
            motor.SetSpeedCms(0.0f);
            servo.SetAngle(0);
            applied_speed = 0.0f;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        const bool roadOK = roadModel.Predict(frame, mask);

        auto now = std::chrono::steady_clock::now();
        double dt_loop = std::chrono::duration<double>(now - tPrevLoop).count();
        tPrevLoop = now;
        if (dt_loop <= 0.0 || dt_loop > 0.5) dt_loop = 1.0 / 30.0;
        totalFrames++;

        // ═══════════════════════════════════════════════════════
        // A. SIGN DETECTION + VOTER
        // ═══════════════════════════════════════════════════════
        if (signReady && (tickIdx % ucr::config::SIGN_DETECT_EVERY) == 0) {
            cachedSigns = signDetector.Detect(
                frame,
                ucr::config::SIGN_CONF_THRESH,
                ucr::config::SIGN_IOU_THRESH);

            for (const auto& s : cachedSigns) {
                if (s.name != "traffic_light") continue;
                if (s.confidence < ucr::config::SIGN_CONF_THRESH) continue;
                LightColor c = ClassifyTrafficLight(frame, s.box);
                if (c == LightColor::RED)        lastRedSeenTick = (int)tickIdx;
                else if (c == LightColor::GREEN) lastRedSeenTick = -1;
                break;
            }

            std::vector<int> classIds;
            for (const auto& s : cachedSigns) {
                if (s.name == "traffic_light") continue;
                if (s.confidence < ucr::config::SIGN_CONF_THRESH) continue;
                if (active_block_ticks > 0 && s.name == active_turn_sign) continue;
                classIds.push_back(s.class_id);
            }
            voter.Update(classIds, tickIdx);
        }

        // ═══════════════════════════════════════════════════════
        // B. ADAPTER: SignVoter → NavDecision
        // ═══════════════════════════════════════════════════════
        NavDecision dec;
        dec.action         = NavAction::FOLLOW_LANE;
        dec.active_sign    = "NONE";
        dec.emergency_stop = false;
        // ✅ FIX: đã xoá dec.target_speed (dead code, không dùng)

        {
            const int committed_id = voter.GetCommitted();
            if (committed_id >= 0 &&
                committed_id < (int)SignDetector::CLASS_NAMES.size()) {

                const std::string& sign_name = SignDetector::CLASS_NAMES[committed_id];
                const bool is_blocked =
                    (active_block_ticks > 0 && sign_name == active_turn_sign);

                if (!is_blocked) {
                    dec.active_sign = sign_name;

                    if (sign_name == "parking") {
                        dec.action = NavAction::STOP_PARKING;
                        dec.emergency_stop = true;
                    }
                    else if (sign_name == "turn_left" ||
                             sign_name == "construction_ahead" ||
                             sign_name == "no_right") {
                        dec.action = NavAction::FORCE_LEFT;
                    }
                    else if (sign_name == "turn_right" ||
                             sign_name == "no_left") {
                        dec.action = NavAction::FORCE_RIGHT;
                    }
                }
            }
        }

        // ═══════════════════════════════════════════════════════
        // C. CONTROL
        // ═══════════════════════════════════════════════════════
        v21::LaneFollower::Output out;
        turn::LaneTurnController::TurnResult turn_res{
            turn::LaneTurnController::State::IDLE, 0.0, "NONE"};
        double applied_floor = 0.0;
        float  speed_cmd = 0.0f;
        int    final_servo = 0;

        if (roadOK) {
            const int h = mask.rows, w = mask.cols;
            const int y1 = (int)(h * 0.45);
            const int y2 = (int)(h * 0.65);
            const int y_bottom = h - 1;

            auto get_mask_boundaries = [&](int y_row, int& lx, int& rx) {
                lx = -1; rx = -1;
                if (y_row < 0 || y_row >= h) return;
                const uchar* row = mask.ptr<uchar>(y_row);
                int start = -1, best_len = -1;
                for (int x = 0; x <= w; x++) {
                    const bool on = (x < w) && (row[x] > 0);
                    if (on && start < 0) start = x;
                    if (!on && start >= 0) {
                        int len = x - start;
                        if (len > best_len) { best_len = len; lx = start; rx = x - 1; }
                        start = -1;
                    }
                }
            };

            auto find_edge_corner_points = [&](int sy, int ey, bool isLeft) {
                std::vector<cv::Point> pts;
                bool touching = false;
                for (int y = sy; y <= ey; y += 2) {
                    int lx, rx;
                    get_mask_boundaries(y, lx, rx);
                    const bool t = isLeft ? (lx != -1 && lx <= 2)
                                          : (rx != -1 && rx >= w - 3);
                    if (t && !touching) { touching = true; pts.push_back({isLeft ? 0 : w - 1, y}); }
                    else if (!t && touching) { touching = false; }
                }
                return pts;
            };

            int lx1, rx1, lx2, rx2, lx_bot, rx_bot;
            get_mask_boundaries(y1,       lx1, rx1);
            get_mask_boundaries(y2,       lx2, rx2);
            get_mask_boundaries(y_bottom, lx_bot, rx_bot);

            std::vector<cv::Point> left_pts, right_pts;
            if (lx1    != -1) left_pts.push_back({lx1, y1});
            for (auto& p : find_edge_corner_points(y1, y_bottom, true)) left_pts.push_back(p);
            if (lx2    != -1) left_pts.push_back({lx2, y2});
            if (lx_bot != -1) left_pts.push_back({lx_bot, y_bottom});

            if (rx1    != -1) right_pts.push_back({rx1, y1});
            for (auto& p : find_edge_corner_points(y1, y_bottom, false)) right_pts.push_back(p);
            if (rx2    != -1) right_pts.push_back({rx2, y2});
            if (rx_bot != -1) right_pts.push_back({rx_bot, y_bottom});

            auto ptLess = [](const cv::Point& a, const cv::Point& b){ return a.y < b.y; };
            std::sort(left_pts.begin(),  left_pts.end(),  ptLess);
            std::sort(right_pts.begin(), right_pts.end(), ptLess);
            left_pts.erase(std::unique(left_pts.begin(),  left_pts.end()),  left_pts.end());
            right_pts.erase(std::unique(right_pts.begin(),right_pts.end()), right_pts.end());

            if (turn_controller.state == turn::LaneTurnController::State::IDLE
                && !corner_assist.active()) {
                if (dec.active_sign == "roundabout") {
                    turn_controller.start_turn("RIGHT", true);
                    active_turn_sign   = "roundabout";
                    active_block_ticks = SIGN_BLOCK_TICKS;   // ✅ FIX
                    voter.Reset();
                } else if (dec.action == NavAction::FORCE_LEFT) {
                    turn_controller.start_turn("LEFT", false);
                    active_turn_sign   = dec.active_sign;
                    active_block_ticks = SIGN_BLOCK_TICKS;   // ✅ FIX
                    voter.Reset();
                } else if (dec.action == NavAction::FORCE_RIGHT) {
                    turn_controller.start_turn("RIGHT", false);
                    active_turn_sign   = dec.active_sign;
                    active_block_ticks = SIGN_BLOCK_TICKS;   // ✅ FIX
                    voter.Reset();
                }
            }

            turn_res = turn_controller.update(
                left_pts, right_pts, dt_loop, previous_out.valid,
                lx1, rx1, lx2, rx2, w,
                (dec.active_sign != "NONE"));

            applied_floor = turn_res.steer_floor_deg;
            if (turn_controller.state == turn::LaneTurnController::State::IDLE) {
                double cf = corner_assist.update(lx1, rx1, lx2, rx2, lx_bot, rx_bot, w, dt_loop);
                if (std::abs(cf) > std::abs(applied_floor)) applied_floor = cf;
            } else {
                corner_assist.reset();
            }

            out = follower.step(
                mask,
                v21::SignState{},
                NAN,
                (float)turn_res.steer_bias_px,
                (float)applied_floor,
                (float)turn_res.steer_cap_deg,
                turn_res.lock_opposite_err);
            previous_out = out;

            if (turn_res.state != turn::LaneTurnController::State::IDLE) {
                const char* sn = "IDLE";
                using S = turn::LaneTurnController::State;
                switch (turn_res.state) {
                    case S::WAIT_CURVE:    sn = "WAIT_CURVE";    break;
                    case S::WAIT_STRAIGHT: sn = "WAIT_STRAIGHT"; break;
                    case S::EXECUTE:       sn = "EXECUTE";       break;
                    case S::SETTLE:        sn = "SETTLE";        break;
                    case S::DONE:          sn = "DONE";          break;
                    default: break;
                }
                out.state_str = std::string(sn) + "_" + turn_res.direction;
            }

            if (turn_res.state == turn::LaneTurnController::State::DONE) {
                voter.Reset();
                if (active_turn_sign != "NONE") {
                    active_block_ticks = SIGN_BLOCK_TICKS;   // ✅ FIX
                }
            }

            if (dec.emergency_stop) {
                speed_cmd   = 0.0f;
                final_servo = MapWheelAngleToServoCmd(0.0);
            } else if (!out.valid && out.coverage < 0.02f) {
                speed_cmd   = 0.0f;
                final_servo = MapWheelAngleToServoCmd(0.0);
            } else {
                final_servo = MapWheelAngleToServoCmd(out.angle);
                float target = out.valid ? static_cast<float>(out.speed)
                                        : static_cast<float>(ucr::config::MIN_SPEED_CMS);
                speed_cmd = std::clamp(target,
                                    static_cast<float>(ucr::config::MIN_SPEED_CMS),
                                    static_cast<float>(ucr::config::MAX_SPEED_CMS));
            }

            if (lastRedSeenTick >= 0 &&
                ((int)tickIdx - lastRedSeenTick <= ucr::config::RED_LIGHT_HANGOVER_TICKS)) {
                speed_cmd = 0.0f;
            }
        } else {
            motor.SetSpeedCms(0.0f);
            servo.SetAngle(0);
            follower.resetTracking();
            applied_speed = 0.0f;
            previous_out = v21::LaneFollower::Output{};
            turn_controller = turn::LaneTurnController{};
            corner_assist.reset();
            voter.Reset();
            active_turn_sign   = "NONE";
            active_block_ticks = 0;
        }

        if (active_block_ticks > 0 &&
            turn_controller.state == turn::LaneTurnController::State::IDLE) {
            --active_block_ticks;
            if (active_block_ticks == 0) {
                active_turn_sign = "NONE";
            }
        }

        if (roadOK) {
            const float max_delta =
                ucr::config::MAX_ACCEL_CMS2 *
                (ucr::config::CONTROL_PERIOD_MS / 1000.0f);
            if (speed_cmd > applied_speed)
                applied_speed = std::min(speed_cmd, applied_speed + max_delta);
            else
                applied_speed = speed_cmd;

            motor.SetSpeedCms(applied_speed);
            servo.SetAngle(final_servo);
        }

        fpsFrames++;
        auto tNow = std::chrono::high_resolution_clock::now();
        double dtFps = std::chrono::duration<double>(tNow - tPrev).count();
        if (dtFps > 0.5) { currentFps = fpsFrames / dtFps; fpsFrames = 0; tPrev = tNow; }

        if (ucr::config::SHOW_UI) {
            cv::Mat overlay = frame.clone();
            if (roadOK) {
                cv::Mat green(frame.size(), frame.type(), cv::Scalar(0, 255, 0));
                cv::Mat blended;
                cv::addWeighted(frame, 0.65, green, 0.35, 0, blended);
                blended.copyTo(overlay, mask);

                if (!out.final_path.empty() && out.proc_width > 0) {
                    float sx = (float)frame.cols / out.proc_width;
                    float sy = (float)frame.rows / out.proc_height;
                    std::vector<cv::Point> pts;
                    for (const auto& p : out.final_path)
                        pts.emplace_back((int)(p.x * sx), (int)(p.y * sy));
                    cv::Scalar col(0, 0, 255);
                    if (turn_res.state != turn::LaneTurnController::State::IDLE)
                        col = cv::Scalar(255, 0, 255);
                    for (size_t i = 0; i + 1 < pts.size(); ++i)
                        cv::line(overlay, pts[i], pts[i + 1], col, 2);
                }
            }

            signDetector.Draw(overlay, cachedSigns);

            auto action_to_str = [](NavAction a) -> const char* {
                switch (a) {
                    case NavAction::FOLLOW_LANE:  return "FOLLOW";
                    case NavAction::FORCE_LEFT:   return "LEFT";
                    case NavAction::FORCE_RIGHT:  return "RIGHT";
                    case NavAction::STOP_PARKING: return "STOP";
                }
                return "?";
            };

            char l1[256], l2[256], l3[256], l4[256];
            std::snprintf(l1, sizeof(l1),
                "FPS %.1f | spd %.1f/%.1f | PID %.1f | out %d",
                currentFps, applied_speed, out.speed, out.angle, final_servo);
            std::snprintf(l2, sizeof(l2),
                "sign %s | action %s | turn %d/%s",
                dec.active_sign.c_str(), action_to_str(dec.action),
                (int)turn_res.state, turn_res.direction.c_str());
            std::snprintf(l3, sizeof(l3),
                "bias %.1f | floor %.1f | near %.1f far %.1f | cov %.3f%s",
                turn_res.steer_bias_px, applied_floor,
                out.near_e, out.far_e, out.coverage,
                out.valid ? "" : " INV");
            std::snprintf(l4, sizeof(l4),
                "voter committed=%d | block %s(%d)",
                voter.GetCommitted(),
                active_turn_sign.c_str(), active_block_ticks);
            cv::putText(overlay, l1, {10, 25},  cv::FONT_HERSHEY_SIMPLEX, 0.55, {0,255,0}, 2);
            cv::putText(overlay, l2, {10, 50},  cv::FONT_HERSHEY_SIMPLEX, 0.55, {0,255,255}, 2);
            cv::putText(overlay, l3, {10, 75},  cv::FONT_HERSHEY_SIMPLEX, 0.55, {0,255,0}, 2);
            cv::putText(overlay, l4, {10, 100}, cv::FONT_HERSHEY_SIMPLEX, 0.50, {255,128,0}, 1);

            cv::imshow("UCR2026", overlay);
            int key = cv::waitKey(1) & 0xFF;
            if (key == 27 || key == 'q') break;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        ++tickIdx;
        if (now >= next_control) {
            next_control += std::chrono::milliseconds(ucr::config::CONTROL_PERIOD_MS);
            if (now - next_control > std::chrono::seconds(1))
                next_control = now +
                    std::chrono::milliseconds(ucr::config::CONTROL_PERIOD_MS);

            static auto lastLog = std::chrono::steady_clock::now();
            if (now - lastLog >= std::chrono::seconds(1)) {
                std::printf(
                    "[t=%u] voter=%d sign=%s turn=%d/%s bias=%.1f floor=%.1f "
                    "near=%.1f far=%.1f spd=%.1f servo=%d block=%s(%d)\n",
                    tickIdx, voter.GetCommitted(), dec.active_sign.c_str(),
                    (int)turn_res.state, turn_res.direction.c_str(),
                    turn_res.steer_bias_px, applied_floor,
                    out.near_e, out.far_e, applied_speed, final_servo,
                    active_turn_sign.c_str(), active_block_ticks);
                lastLog = now;
            }
        }
    }

    motor.SetSpeedCms(0.0f);
    servo.SetAngle(0);
    if (ucr::config::SHOW_UI) cv::destroyAllWindows();
    std::cout << "\n[System] Stopped safely\n";
    return 0;
}
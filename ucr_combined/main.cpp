// ============================================================
// main_run.cpp — UCR2026 Full Integrated
//
// Merge of three working pieces:
//   • v20::LaneFollower          — lane follow (LaneFollowerV20.h)
//   • v20::LaneTurnController    — turn sequencing (regular turns)
//   • v20::CornerAssist          — 90° unmarked corners
//   • RbState (V6.6)             — roundabout state machine (bias-only)
//   • SignVoter (V6.6)           — sign commit + timeout
//   • SignDetector (v2)          — 8-class YOLO detector
//
// Pipeline per frame:
//   camera → TrtModel.Predict → mask
//   SignDetector → SignVoter → SignCommit
//   RbState: voter==roundabout → APPROACH → IN_CIRCLE → EXIT_PREP →
//            EXIT_SEARCH (commits a RIGHT turn via LaneTurnController)
//            → EXITING → NONE
//   If rb_state==NONE and SignCommit is a turn class and LaneTurnController
//   is IDLE, trigger LaneTurnController.start_turn(dir)
//   LaneTurnController.update → TurnResult (steer_bias_px, floor, cap)
//   CornerAssist.update (only if rb_state==NONE && turn_ctrl IDLE)
//   v20::LaneFollower.step(mask, bias, execTurn, floor, cap)
//   v20 angle → servo
// ============================================================
#include "hardware/Camera.h"
#include "hardware/Motor.h"
#include "hardware/Servo.h"
#include "common/Logger.h"

#include "control/LaneFollowerV20.h"
#include "control/SignVoter.h"
#include "perception/SignDetector.h"

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

// ============================================================
// CONSTANTS
// ============================================================
static const std::string ENGINE_PATH      = "/home/car4/Downloads/road/kga2.engine";
static const std::string SIGN_ENGINE_PATH = "/home/car4/Downloads/sign/huy_sign.engine";

// Speed / control
static constexpr float MAX_SPEED_CMS         = 30.0f;
static constexpr float MIN_SPEED_CMS         = 22.0f;
static constexpr float MAX_ACCEL_CMS2        = 30.0f;
static constexpr bool  SHOW_UI               = true;
static constexpr int   SIGN_INTERVAL_FRAMES  = 1;
static constexpr float SIGN_CONF_THRESH      = 0.70f;
static constexpr float SIGN_NMS_THRESH       = 0.45f;
static constexpr bool  DA_CHANNEL            = 1;
static constexpr int   MAX_STEER_DEG         = 50;      // legacy fallback only
static constexpr bool  INVERT_STEER          = true;    // legacy fallback only

// Servo calibration (V6.6 / Log 12 measured)
static constexpr float SERVO_LEFT_SLOPE    = 0.96f;
static constexpr float SERVO_LEFT_OFFSET   = -6.2f;
static constexpr float SERVO_RIGHT_SLOPE   = 0.71f;
static constexpr float SERVO_RIGHT_OFFSET  = 2.5f;
static constexpr float MAX_REAL_ANGLE_DEG  = 30.0f;
static constexpr int   MAX_SERVO_CMD       = 55;

// Sign class indices (match SignDetector::CLASS_NAMES)
namespace sign_class {
    inline constexpr int CONSTRUCTION_AHEAD = 0;
    inline constexpr int NO_LEFT            = 1;
    inline constexpr int NO_RIGHT           = 2;
    inline constexpr int PARKING            = 3;
    inline constexpr int ROUNDABOUT         = 4;
    inline constexpr int TRAFFIC_LIGHT      = 5;
    inline constexpr int TURN_LEFT          = 6;
    inline constexpr int TURN_RIGHT         = 7;
}

// Traffic light HSV classifier (V6.6)
namespace tl {
    inline constexpr int RED_H1_LO = 0,   RED_H1_HI = 10;
    inline constexpr int RED_H2_LO = 170, RED_H2_HI = 180;
    inline constexpr int YEL_H_LO  = 15,  YEL_H_HI  = 35;
    inline constexpr int GRN_H_LO  = 40,  GRN_H_HI  = 90;
    inline constexpr int S_MIN     = 100;
    inline constexpr int V_MIN     = 100;
    inline constexpr int MIN_LIT_PIXELS = 20;
    inline constexpr float ROI_EXPAND   = 1.20f;
}

// Stop hangovers (seconds)
static constexpr double PERSON_STOP_HANGOVER_S = 2.0;
static constexpr double RED_LIGHT_HANGOVER_S   = 1.0;

// Voter
static constexpr int SIGN_VOTER_ACTIVATE_FRAMES = 2;
static constexpr int SIGN_VOTER_TIMEOUT_DS      = 30;    // 30 deciseconds = 3.0 s

// Sign re-trigger block after a completed turn (V6.6 handled_sign)
static constexpr double SIGN_BLOCK_SECONDS      = 4.0;

// Roundabout timings (V6.6, converted to seconds at 10 Hz)
static constexpr float  RB_BIAS_PX_RIGHT        = 55.0f;
static constexpr float  RB_BIAS_PX_LEFT         = 30.0f;
static constexpr double RB_MISSING_SIGN_S       = 1.0;   // was 10 ticks @ 10 Hz
static constexpr double RB_CIRCLE_TIMEOUT_S     = 10.0;  // was 100 ticks
static constexpr double RB_EXIT_SEARCH_TIMEOUT_S = 4.0;  // was 40 ticks
static constexpr double RB_COOLDOWN_S           = 2.0;   // was 20 ticks
static constexpr float  RB_SPEED_MIN_CMS        = 20.0f;
static constexpr float  RB_SPEED_MAX_CMS        = 30.0f;

volatile std::sig_atomic_t g_stopSignal = 0;
static void SignalHandler(int) { g_stopSignal = 1; }

// ============================================================
// TRT MODEL (from main_run.cpp)
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
// ROUNDABOUT STATE (V6.6)
// ============================================================
enum class RbState {
    NONE, APPROACH, IN_CIRCLE, EXIT_PREP, EXIT_SEARCH, EXITING
};

static const char* RbStateName(RbState s) {
    switch (s) {
        case RbState::APPROACH:    return "APPROACH";
        case RbState::IN_CIRCLE:   return "IN_CIRCLE";
        case RbState::EXIT_PREP:   return "EXIT_PREP";
        case RbState::EXIT_SEARCH: return "EXIT_SEARCH";
        case RbState::EXITING:     return "EXITING";
        default:                   return "NONE";
    }
}

// ============================================================
// SIGN COMMIT (replaces v21::SignState)
// ============================================================
struct SignCommit {
    bool active   = false;
    int  side     = 0;     // -1 left, +1 right, 0 none
    int  class_id = -1;
};

static SignCommit MakeSignCommit(int committed) {
    SignCommit s;
    if (committed < 0) return s;
    using namespace sign_class;
    s.class_id = committed;
    switch (committed) {
        case TURN_LEFT:  s.active = true; s.side = -1; break;
        case TURN_RIGHT: s.active = true; s.side = +1; break;
        case NO_LEFT:    s.active = true; s.side = +1; break;
        case NO_RIGHT:   s.active = true; s.side = -1; break;
        // CONSTRUCTION_AHEAD was already remapped to TURN_LEFT in main loop
        // ROUNDABOUT, PARKING, TRAFFIC_LIGHT → inactive
        default:         s.active = false; s.side = 0; break;
    }
    return s;
}

// ============================================================
// TRAFFIC LIGHT
// ============================================================
enum class LightColor { NONE, RED, YELLOW, GREEN };

static LightColor ClassifyTrafficLight(const cv::Mat& bgr, const cv::Rect& bbox) {
    using namespace tl;
    const int cx = bbox.x + bbox.width / 2;
    const int cy = bbox.y + bbox.height / 2;
    const int w  = static_cast<int>(bbox.width  * ROI_EXPAND);
    const int h  = static_cast<int>(bbox.height * ROI_EXPAND);
    const int x1 = std::max(0, cx - w / 2);
    const int y1 = std::max(0, cy - h / 2);
    const int x2 = std::min(bgr.cols, cx + w / 2);
    const int y2 = std::min(bgr.rows, cy + h / 2);
    if (x2 - x1 < 4 || y2 - y1 < 4) return LightColor::NONE;

    cv::Mat hsv;
    cv::cvtColor(bgr(cv::Rect(x1, y1, x2 - x1, y2 - y1)), hsv, cv::COLOR_BGR2HSV);

    cv::Mat mRed1, mRed2, mRed;
    cv::inRange(hsv, cv::Scalar(RED_H1_LO, S_MIN, V_MIN),
                     cv::Scalar(RED_H1_HI, 255, 255), mRed1);
    cv::inRange(hsv, cv::Scalar(RED_H2_LO, S_MIN, V_MIN),
                     cv::Scalar(RED_H2_HI, 255, 255), mRed2);
    cv::bitwise_or(mRed1, mRed2, mRed);
    int nRed = cv::countNonZero(mRed);

    cv::Mat mYel;
    cv::inRange(hsv, cv::Scalar(YEL_H_LO, S_MIN, V_MIN),
                     cv::Scalar(YEL_H_HI, 255, 255), mYel);
    int nYel = cv::countNonZero(mYel);

    cv::Mat mGrn;
    cv::inRange(hsv, cv::Scalar(GRN_H_LO, S_MIN, V_MIN),
                     cv::Scalar(GRN_H_HI, 255, 255), mGrn);
    int nGrn = cv::countNonZero(mGrn);

    if (nRed >= MIN_LIT_PIXELS && nRed > nYel && nRed > nGrn) return LightColor::RED;
    if (nYel >= MIN_LIT_PIXELS && nYel > nRed && nYel > nGrn) return LightColor::YELLOW;
    if (nGrn >= MIN_LIT_PIXELS && nGrn > nRed && nGrn > nYel) return LightColor::GREEN;
    return LightColor::NONE;
}

// ============================================================
// SERVO CALIBRATION (V6.6)
// ============================================================
static float RealAngleToServo(float real_deg) {
    if (real_deg > 0.0f) return (real_deg - SERVO_LEFT_OFFSET) / SERVO_LEFT_SLOPE;
    if (real_deg < 0.0f) return (real_deg - SERVO_RIGHT_OFFSET) / SERVO_RIGHT_SLOPE;
    return 0.0f;
}

static const char* SignName(int id) {
    using namespace sign_class;
    switch (id) {
        case CONSTRUCTION_AHEAD: return "construct";
        case NO_LEFT:            return "no_left";
        case NO_RIGHT:           return "no_right";
        case PARKING:            return "parking";
        case ROUNDABOUT:         return "roundabout";
        case TRAFFIC_LIGHT:      return "traffic_light";
        case TURN_LEFT:          return "turn_left";
        case TURN_RIGHT:         return "turn_right";
        default:                 return "-";
    }
}

// ============================================================
// MAIN
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
        std::cerr << "Servo init failed\n"; motor.SetSpeedCms(0.0f); return -1;
    }
    motor.SetSpeedCms(0.0f);
    servo.SetAngle(0);

    TrtModel roadModel;
    if (!roadModel.Init(ENGINE_PATH)) {
        std::cerr << "Road Model init failed\n"; return -1;
    }

    SignDetector signDetector;
    bool signReady = signDetector.Init(SIGN_ENGINE_PATH);
    if (!signReady) {
        std::cerr << "[Main] Sign Detector init failed — turns/roundabout DISABLED\n";
    }

    // Sign voter (V6.6) — tickIdx passed in deciseconds
    ucr::control::SignVoter voter(
        SIGN_VOTER_ACTIVATE_FRAMES,
        SIGN_VOTER_TIMEOUT_DS);

    // v20 core
    v20::Config cfg;
    cfg.max_speed          = MAX_SPEED_CMS;
    cfg.min_speed          = MIN_SPEED_CMS;
    cfg.curve_scale        = 140.0;
    cfg.min_rows           = 5;
    cfg.coverage_threshold = 0.05;
    cfg.kp = 0.40;
    cfg.kd = 0.42;
    cfg.derivative_alpha = 0.25;
    cfg.output_limit = 50.0;
    cfg.emergency_output_limit = 50.0;
    cfg.rate_limit = 10.0;
    cfg.emergency_rate_limit = 18.0;
    cfg.deadband = 0.5;

    v20::LaneFollower       follower(cfg);
    v20::LaneTurnController turn_controller;
    v20::CornerAssist       corner_assist;
    v20::LaneFollower::Output previous_out;

    // Run-state
    RbState  rb_state = RbState::NONE;
    double   rb_enter_time         = 0.0;
    double   rb_phase_start_time   = 0.0;
    double   rb_exit_search_start  = 0.0;
    double   rb_exiting_start      = 0.0;
    double   rb_cooldown_until     = 0.0;
    int      rb_missing_sign_ticks = 0;
    double   last_rb_visible_time  = -1e9;

    double   last_person_seen_time = -1e9;
    double   last_red_seen_time    = -1e9;
    double   sign_block_until_time = -1.0;

    long     tickIdx          = 0;
    long     totalFrames      = 0;
    double   currentFps       = 0.0;
    double   elapsed_total    = 0.0;
    float    applied_speed    = 0.0f;

    auto tPrev     = std::chrono::high_resolution_clock::now();
    auto tPrevLoop = std::chrono::high_resolution_clock::now();
    long frameCount = 0;

    cv::Mat frame, mask;
    std::vector<SignObject> cachedSigns;

    std::cout << "\n=== UCR2026: Full Integrated (v20 + RbState + SignVoter) ===\n"
              << "Road: " << ENGINE_PATH << "\n"
              << "Sign: " << SIGN_ENGINE_PATH << (signReady ? "" : " (DISABLED)") << "\n"
              << "Speed: [" << MIN_SPEED_CMS << ", " << MAX_SPEED_CMS << "] cm/s\n"
              << "Roundabout: bias R=" << RB_BIAS_PX_RIGHT
              << " L=" << RB_BIAS_PX_LEFT
              << " | missing_sign=" << RB_MISSING_SIGN_S << "s"
              << " circle_timeout=" << RB_CIRCLE_TIMEOUT_S << "s\n"
              << "Press 'q' or ESC to quit\n\n";

    while (!g_stopSignal) {
        camera.GetFrame(frame);
        if (frame.empty()) {
            motor.SetSpeedCms(0.0f);
            servo.SetAngle(0);
            applied_speed = 0.0f;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        auto tNowLoop = std::chrono::high_resolution_clock::now();
        double dt_loop = std::chrono::duration<double>(tNowLoop - tPrevLoop).count();
        tPrevLoop = tNowLoop;
        if (dt_loop <= 0.0) dt_loop = 1.0 / 30.0;
        if (dt_loop > 1.0) dt_loop = 1.0;   // clamp long stalls
        elapsed_total += dt_loop;
        totalFrames++;

        // -----------------------------------------------------
        // 1. Sign detection + voter update
        // -----------------------------------------------------
        if (signReady && (totalFrames % SIGN_INTERVAL_FRAMES) == 0) {
            cachedSigns = signDetector.Detect(frame, SIGN_CONF_THRESH, SIGN_NMS_THRESH);

            std::vector<int> classIds;
            classIds.reserve(cachedSigns.size());
            for (auto& d : cachedSigns) {
                if (d.confidence < SIGN_CONF_THRESH) continue;

                // walking_person (defensive — not in current 8-class detector)
                if (d.name == "walking_person" || d.class_id == 8) {
                    last_person_seen_time = elapsed_total;
                    continue;
                }
                // traffic_light
                if (d.name == "traffic_light" || d.class_id == TRAFFIC_LIGHT) {
                    if (ClassifyTrafficLight(frame, d.box) == LightColor::RED)
                        last_red_seen_time = elapsed_total;
                    else
                        last_red_seen_time = -1e9;
                    continue;
                }
                // Remap construction_ahead → turn_left (V6.6)
                int cls = d.class_id;
                if (cls == sign_class::CONSTRUCTION_AHEAD) cls = sign_class::TURN_LEFT;

                // Feed only turn-class + roundabout to the voter
                if (cls == sign_class::TURN_LEFT  ||
                    cls == sign_class::TURN_RIGHT ||
                    cls == sign_class::NO_LEFT    ||
                    cls == sign_class::NO_RIGHT   ||
                    cls == sign_class::ROUNDABOUT) {
                    classIds.push_back(cls);
                }
            }
            // tickIdx in deciseconds
            uint32_t ds = static_cast<uint32_t>(elapsed_total * 10.0);
            voter.Update(classIds, ds);
        }

        // -----------------------------------------------------
        // 2. RbState progression
        // -----------------------------------------------------
        bool rb_visible_now = false;
        for (const auto& d : cachedSigns) {
            if ((d.name == "roundabout" || d.class_id == sign_class::ROUNDABOUT) &&
                d.confidence >= SIGN_CONF_THRESH) {
                rb_visible_now = true;
                break;
            }
        }
        if (rb_visible_now) last_rb_visible_time = elapsed_total;

        const bool rb_committed =
            (voter.GetCommitted() == sign_class::ROUNDABOUT);

        // Missing-sign accumulator (only used in APPROACH / EXIT_PREP)
        if (rb_state == RbState::APPROACH || rb_state == RbState::EXIT_PREP) {
            if (rb_visible_now) rb_missing_sign_ticks = 0;
            else                ++rb_missing_sign_ticks;
        }

        switch (rb_state) {
            case RbState::NONE:
                if (rb_committed &&
                    elapsed_total >= rb_cooldown_until &&
                    turn_controller.state == v20::LaneTurnController::State::IDLE &&
                    !corner_assist.active()) {
                    rb_state = RbState::APPROACH;
                    rb_missing_sign_ticks = 0;
                    rb_phase_start_time   = elapsed_total;
                }
                break;

            case RbState::APPROACH:
                if (rb_missing_sign_ticks * dt_loop >= RB_MISSING_SIGN_S) {
                    rb_state = RbState::IN_CIRCLE;
                    rb_enter_time      = elapsed_total;
                    rb_phase_start_time = elapsed_total;
                }
                break;

            case RbState::IN_CIRCLE: {
                const bool timeout = (elapsed_total - rb_enter_time) >= RB_CIRCLE_TIMEOUT_S;
                const bool rb_seen_again = rb_committed && rb_visible_now;
                if (timeout || rb_seen_again) {
                    rb_state = RbState::EXIT_PREP;
                    rb_missing_sign_ticks = 0;
                    rb_phase_start_time   = elapsed_total;
                }
                break;
            }

            case RbState::EXIT_PREP:
                if (rb_missing_sign_ticks * dt_loop >= RB_MISSING_SIGN_S) {
                    rb_state = RbState::EXIT_SEARCH;
                    rb_exit_search_start = elapsed_total;
                    rb_phase_start_time  = elapsed_total;
                }
                break;

            case RbState::EXIT_SEARCH: {
                // Trigger a RIGHT turn via turn_controller if not already running
                if (turn_controller.state == v20::LaneTurnController::State::IDLE) {
                    turn_controller.start_turn("RIGHT", false);
                }
                const bool moved_on =
                    turn_controller.state == v20::LaneTurnController::State::WAIT_STRAIGHT ||
                    turn_controller.state == v20::LaneTurnController::State::EXECUTE    ||
                    turn_controller.state == v20::LaneTurnController::State::SETTLE;
                const bool timed_out =
                    (elapsed_total - rb_exit_search_start) >= RB_EXIT_SEARCH_TIMEOUT_S;
                if (moved_on || timed_out) {
                    rb_state = RbState::EXITING;
                    rb_exiting_start = elapsed_total;
                    rb_phase_start_time = elapsed_total;
                }
                break;
            }

            case RbState::EXITING: {
                const bool turn_done =
                    turn_controller.state == v20::LaneTurnController::State::IDLE;
                const bool safety_timeout =
                    (elapsed_total - rb_exiting_start) >= RB_CIRCLE_TIMEOUT_S;
                if (turn_done || safety_timeout) {
                    if (safety_timeout && !turn_done) {
                        // Force-reset a stuck turn
                        turn_controller = v20::LaneTurnController{};
                        corner_assist.reset();
                    }
                    rb_state = RbState::NONE;
                    rb_cooldown_until = elapsed_total + RB_COOLDOWN_S;
                    voter.Reset();
                    // Arm T-junction auto-detection after roundabout (V20 feature)
                    turn_controller.waiting_for_tjunction = true;
                    turn_controller.tjunction_cooldown =
                        v20::LaneTurnController::TJUNCTION_COOLDOWN_S;
                }
                break;
            }
        }

        // -----------------------------------------------------
        // 3. Build SignCommit + Rb bias
        // -----------------------------------------------------
        SignCommit signCommit = MakeSignCommit(voter.GetCommitted());

        float rb_bias_px = 0.0f;
        switch (rb_state) {
            case RbState::APPROACH:    rb_bias_px = +RB_BIAS_PX_RIGHT; break;
            case RbState::IN_CIRCLE:   rb_bias_px = -RB_BIAS_PX_LEFT;  break;
            case RbState::EXIT_PREP:   rb_bias_px = +RB_BIAS_PX_RIGHT; break;
            case RbState::EXIT_SEARCH: rb_bias_px = -RB_BIAS_PX_LEFT;  break;
            case RbState::EXITING:     rb_bias_px =  0.0f;             break;
            default:                   rb_bias_px =  0.0f;             break;
        }

        // -----------------------------------------------------
        // 4. Road inference
        // -----------------------------------------------------
        bool roadOK = roadModel.Predict(frame, mask);

        v20::LaneFollower::Output out;
        float speed = MIN_SPEED_CMS;
        int   final_angle = 0;
        v20::LaneTurnController::TurnResult turn_res{
            v20::LaneTurnController::State::IDLE, 0.0, "NONE"};
        double applied_floor = 0.0;
        double applied_cap   = 0.0;
        float  final_bias    = rb_bias_px;
        bool   exec_turn     = (rb_state != RbState::NONE);

        std::vector<cv::Point> ui_left_pts, ui_right_pts;
        int ui_y1 = 0, ui_y2 = 0;
        int ui_lx1 = -1, ui_rx1 = -1, ui_lx2 = -1, ui_rx2 = -1;
        int ui_lx_bot = -1, ui_rx_bot = -1;

        if (roadOK) {
            const int h = mask.rows, w = mask.cols;
            const int y1 = static_cast<int>(h * 0.45);
            const int y2 = static_cast<int>(h * 0.65);
            const int y_bottom = h - 1;

            auto get_mask_boundaries = [&](int y_row, int& lx, int& rx) {
                lx = -1; rx = -1;
                if (y_row < 0 || y_row >= h) return;
                const uchar* row = mask.ptr<uchar>(y_row);
                int start = -1, best_len = -1;
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
            auto find_edge_corner_points = [&](int sy, int ey, bool is_left) {
                std::vector<cv::Point> pts;
                bool touching = false;
                for (int y = sy; y <= ey; y += 2) {
                    int lx, rx;
                    get_mask_boundaries(y, lx, rx);
                    bool t = is_left ? (lx != -1 && lx <= 2) : (rx != -1 && rx >= w - 3);
                    if (t && !touching) {
                        touching = true;
                        pts.push_back(cv::Point(is_left ? 0 : w - 1, y));
                    } else if (!t && touching) touching = false;
                }
                return pts;
            };

            int lx1, rx1, lx2, rx2, lx_bot, rx_bot;
            get_mask_boundaries(y1, lx1, rx1);
            get_mask_boundaries(y2, lx2, rx2);
            get_mask_boundaries(y_bottom, lx_bot, rx_bot);
            ui_lx1 = lx1; ui_rx1 = rx1; ui_lx2 = lx2; ui_rx2 = rx2;
            ui_lx_bot = lx_bot; ui_rx_bot = rx_bot;

            std::vector<cv::Point> left_pts, right_pts;
            if (lx1 != -1) left_pts.push_back(cv::Point(lx1, y1));
            for (auto p : find_edge_corner_points(y1, y_bottom, true)) left_pts.push_back(p);
            if (lx2 != -1) left_pts.push_back(cv::Point(lx2, y2));
            if (lx_bot != -1) left_pts.push_back(cv::Point(lx_bot, y_bottom));

            if (rx1 != -1) right_pts.push_back(cv::Point(rx1, y1));
            for (auto p : find_edge_corner_points(y1, y_bottom, false)) right_pts.push_back(p);
            if (rx2 != -1) right_pts.push_back(cv::Point(rx2, y2));
            if (rx_bot != -1) right_pts.push_back(cv::Point(rx_bot, y_bottom));

            std::sort(left_pts.begin(), left_pts.end(),
                      [](const cv::Point& a, const cv::Point& b){ return a.y < b.y; });
            std::sort(right_pts.begin(), right_pts.end(),
                      [](const cv::Point& a, const cv::Point& b){ return a.y < b.y; });

            ui_left_pts  = left_pts;
            ui_right_pts = right_pts;
            ui_y1 = y1; ui_y2 = y2;

            // --- Accept new sign trigger ONLY when not in roundabout ---
            if (rb_state == RbState::NONE &&
                signCommit.active &&
                turn_controller.state == v20::LaneTurnController::State::IDLE &&
                !corner_assist.active() &&
                elapsed_total >= sign_block_until_time) {
                std::string dir = (signCommit.side == -1) ? "LEFT" : "RIGHT";
                turn_controller.start_turn(dir, false);
                voter.Reset();
            }

            // --- Turn controller update (always, so EXIT_SEARCH's turn progresses) ---
            turn_res = turn_controller.update(
                left_pts, right_pts, dt_loop, previous_out.valid,
                lx1, rx1, lx2, rx2, w,
                signCommit.active
            );

            // --- Floor / cap: turn_controller drives when running ---
            applied_floor = turn_res.steer_floor_deg;
            applied_cap   = turn_res.steer_cap_deg;

            // --- CornerAssist only when both are idle and not in roundabout ---
            if (rb_state == RbState::NONE &&
                turn_controller.state == v20::LaneTurnController::State::IDLE) {
                double cf = corner_assist.update(
                    lx1, rx1, lx2, rx2, lx_bot, rx_bot, w, dt_loop,
                    signCommit.active,
                    previous_out.near_e, previous_out.mid_e, previous_out.far_e
                );
                if (std::abs(cf) > std::abs(applied_floor)) applied_floor = cf;
            } else if (rb_state != RbState::NONE) {
                corner_assist.reset();
            }

            // --- Merge biases ---
            // Outside EXITING, rb_bias drives when rb is active.
            // turn_res.steer_bias_px drives the regular/exit turn.
            if (turn_controller.state != v20::LaneTurnController::State::IDLE) {
                final_bias += turn_res.steer_bias_px;
            }
            exec_turn = (rb_state != RbState::NONE) || turn_res.lock_opposite_err;

            // --- Sign-block after a completed turn ---
            if (turn_res.state == v20::LaneTurnController::State::DONE) {
                sign_block_until_time = elapsed_total + SIGN_BLOCK_SECONDS;
                voter.Reset();
            }

            // --- Follower step ---
            out = follower.step(mask, final_bias, exec_turn, applied_floor, applied_cap);
            previous_out = out;

            // --- Speed ---
            speed = static_cast<float>(out.speed);
            if (rb_state != RbState::NONE && out.valid) {
                speed = std::clamp(speed, RB_SPEED_MIN_CMS, RB_SPEED_MAX_CMS);
            }

            // --- Stop overrides ---
            const bool stopForPerson =
                (elapsed_total - last_person_seen_time) <= PERSON_STOP_HANGOVER_S;
            const bool stopForRedLight =
                (elapsed_total - last_red_seen_time) <= RED_LIGHT_HANGOVER_S;
            if (stopForPerson || stopForRedLight) speed = 0.0f;

            // --- Soft-start ramp ---
            const float max_delta = MAX_ACCEL_CMS2 * static_cast<float>(dt_loop);
            if (speed > applied_speed)
                applied_speed = std::min(speed, applied_speed + max_delta);
            else
                applied_speed = speed;

            // --- Angle → servo (V6.6 calibration) ---
            float desired_real_left = -static_cast<float>(out.angle);
            desired_real_left = std::clamp(desired_real_left,
                                           -MAX_REAL_ANGLE_DEG, MAX_REAL_ANGLE_DEG);
            float servo_cmd = RealAngleToServo(desired_real_left);
            servo_cmd = std::clamp(servo_cmd,
                                   -static_cast<float>(MAX_SERVO_CMD),
                                    static_cast<float>(MAX_SERVO_CMD));
            final_angle = static_cast<int>(std::lround(servo_cmd));

            motor.SetSpeedCms(applied_speed);
            servo.SetAngle(final_angle);

            ++tickIdx;

            // 1 Hz debug
            static auto lastLog = std::chrono::steady_clock::now();
            auto tNow = std::chrono::steady_clock::now();
            if (tNow - lastLog >= std::chrono::seconds(1)) {
                const char* stopStr = stopForPerson   ? "STOP:person"
                                    : stopForRedLight ? "STOP:red"
                                    :                   "-";
                std::printf(
                    "[t=%ld] sign=%-11s(%d) rb=%-11s bias=%+5.1f "
                    "turn=%-13s near=%6.1f mid=%6.1f far=%6.1f "
                    "%-11s spd=%.1f pid=%.1f srv=%d\n",
                    tickIdx,
                    SignName(voter.GetCommitted()),
                    voter.GetCommitted(),
                    RbStateName(rb_state),
                    final_bias,
                    v20::LaneTurnController::State::EXECUTE == turn_res.state ? "EXEC" :
                    v20::LaneTurnController::State::WAIT_CURVE == turn_res.state ? "WAIT_CURVE" :
                    v20::LaneTurnController::State::WAIT_STRAIGHT == turn_res.state ? "WAIT_STRAIGHT" :
                    v20::LaneTurnController::State::SETTLE == turn_res.state ? "SETTLE" :
                    v20::LaneTurnController::State::DONE == turn_res.state ? "DONE" : "IDLE",
                    out.near_e, out.mid_e, out.far_e,
                    stopStr,
                    applied_speed, out.angle, final_angle);
                lastLog = tNow;
            }
        } else {
            // Road inference failed
            previous_out = v20::LaneFollower::Output{};
            motor.SetSpeedCms(0.0f);
            servo.SetAngle(0);
            follower.resetTracking();
            applied_speed = 0.0f;
        }

        // -----------------------------------------------------
        // FPS
        // -----------------------------------------------------
        frameCount++;
        auto tNow = std::chrono::high_resolution_clock::now();
        double dt_fps = std::chrono::duration<double>(tNow - tPrev).count();
        if (dt_fps > 0.5) {
            currentFps = frameCount / dt_fps;
            frameCount = 0;
            tPrev = tNow;
        }

        // -----------------------------------------------------
        // UI
        // -----------------------------------------------------
        if (SHOW_UI) {
            cv::Mat overlay = frame.clone();

            if (roadOK) {
                cv::Mat green(frame.size(), frame.type(), cv::Scalar(0, 255, 0));
                cv::Mat blended;
                cv::addWeighted(frame, 0.65, green, 0.35, 0, blended);
                blended.copyTo(overlay, mask);

                // Scanlines
                if (ui_y1 > 0 && ui_y2 > 0) {
                    cv::line(overlay, cv::Point(0, ui_y1), cv::Point(frame.cols - 1, ui_y1),
                             cv::Scalar(0, 0, 255), 1);
                    cv::line(overlay, cv::Point(0, ui_y2), cv::Point(frame.cols - 1, ui_y2),
                             cv::Scalar(0, 0, 255), 1);
                }

                // Left / right boundary polylines
                for (size_t i = 0; i + 1 < ui_left_pts.size(); ++i)
                    cv::line(overlay, ui_left_pts[i], ui_left_pts[i + 1],
                             cv::Scalar(255, 0, 0), 2);
                for (size_t i = 0; i + 1 < ui_right_pts.size(); ++i)
                    cv::line(overlay, ui_right_pts[i], ui_right_pts[i + 1],
                             cv::Scalar(255, 0, 0), 2);
                for (const auto& p : ui_left_pts)
                    cv::circle(overlay, p, 3, cv::Scalar(0, 255, 0), -1);
                for (const auto& p : ui_right_pts)
                    cv::circle(overlay, p, 3, cv::Scalar(0, 255, 0), -1);

                // Centerline dots
                if (out.proc_width > 0 && out.proc_height > 0) {
                    float sx = float(frame.cols) / out.proc_width;
                    float sy = float(frame.rows) / out.proc_height;
                    for (size_t i = 0; i < out.centerline_pts.size(); i += 8) {
                        int cx = int(out.centerline_pts[i].first * sx);
                        int cy = int(out.centerline_pts[i].second * sy);
                        cv::circle(overlay, cv::Point(cx, cy), 2,
                                   cv::Scalar(0, 255, 255), -1);
                    }
                }

                // Desired path (magenta bezier)
                int start_x = frame.cols / 2;
                int start_y = frame.rows - 1;
                int end_y   = static_cast<int>(frame.rows * 0.45);
                float steer_norm = std::clamp(float(out.angle) / 25.0f, -1.0f, 1.0f);
                float lateral_offset = steer_norm * frame.cols * 0.35f;
                int end_x = std::clamp(int(start_x + lateral_offset), 0, frame.cols - 1);
                int control_x = int(start_x + lateral_offset * 0.45f);
                int control_y = int(frame.rows * 0.78f);
                std::vector<cv::Point> path_points;
                for (int i = 0; i <= 50; ++i) {
                    float t = i / 50.0f;
                    float x = std::pow(1 - t, 2) * start_x
                            + 2 * (1 - t) * t * control_x
                            + std::pow(t, 2) * end_x;
                    float y = std::pow(1 - t, 2) * start_y
                            + 2 * (1 - t) * t * control_y
                            + std::pow(t, 2) * end_y;
                    path_points.push_back(cv::Point(int(x), int(y)));
                }
                std::vector<std::vector<cv::Point>> pv = {path_points};
                cv::polylines(overlay, pv, false, cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
            }

            // Sign detections
            for (const auto& s : cachedSigns) {
                cv::rectangle(overlay, s.box, cv::Scalar(0, 255, 255), 2);
                char label[64];
                std::snprintf(label, sizeof(label), "%s %.2f",
                              s.name.c_str(), s.confidence);
                cv::putText(overlay, label, {s.box.x, std::max(12, s.box.y - 5)},
                            cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            cv::Scalar(0, 255, 255), 1);
            }

            // HUD
            char t1[200], t2[200], t3[200];
            std::snprintf(t1, sizeof(t1),
                "FPS %.1f | spd %.1f/%.1f | pid %.1f | srv %d | bias %+.1f",
                currentFps, applied_speed, speed, out.angle, final_angle, final_bias);
            std::snprintf(t2, sizeof(t2),
                "sign %s (%d) | rb %s | turn %s %s",
                SignName(voter.GetCommitted()),
                voter.GetCommitted(),
                RbStateName(rb_state),
                turn_res.direction.c_str(),
                (turn_res.state == v20::LaneTurnController::State::IDLE ? "IDLE" :
                 turn_res.state == v20::LaneTurnController::State::WAIT_CURVE ? "WAIT_CURVE" :
                 turn_res.state == v20::LaneTurnController::State::WAIT_STRAIGHT ? "WAIT_STRAIGHT" :
                 turn_res.state == v20::LaneTurnController::State::EXECUTE ? "EXECUTE" :
                 turn_res.state == v20::LaneTurnController::State::SETTLE ? "SETTLE" : "DONE"));
            const bool stopP = (elapsed_total - last_person_seen_time) <= PERSON_STOP_HANGOVER_S;
            const bool stopR = (elapsed_total - last_red_seen_time)    <= RED_LIGHT_HANGOVER_S;
            std::snprintf(t3, sizeof(t3),
                "near %.1f mid %.1f far %.1f | %s",
                out.near_e, out.mid_e, out.far_e,
                stopP ? "STOP:person" : stopR ? "STOP:red" : "-");

            cv::putText(overlay, t1, {10, 30},  cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 2);
            cv::putText(overlay, t2, {10, 55},  cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 2);
            cv::putText(overlay, t3, {10, 80},  cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 2);

            cv::imshow("Lane Follower", overlay);
            const int key = cv::waitKey(1) & 0xFF;
            if (key == 27 || key == 'q') break;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    motor.SetSpeedCms(0.0f);
    servo.SetAngle(0);
    if (SHOW_UI) cv::destroyAllWindows();
    std::cout << "\n[System] Stopped safely\n";
    return 0;
}
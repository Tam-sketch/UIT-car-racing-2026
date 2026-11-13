// ============================================================
// main_run.cpp — UCR2026 Full Integrated (Parking & Obstacle from chinh.cpp)
// + Roundabout logic ported from roundabout.cpp (unconditional T-junction arm)
// ============================================================
#include "hardware/Camera.h"
#include "hardware/Motor.h"
#include "hardware/Servo.h"
#include "common/Logger.h"

#include "control/LaneFollowerV20.h"
#include "control/SignVoter.h"
#include "perception/SignDetector.h"
#include "sensor/BNO055.h"
#include "sensor/TOF.h"
#include "control/TelemetryHelper.h"

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
#include <deque>

// ============================================================
// CONSTANTS
// ============================================================
static const std::string ENGINE_PATH      = "/home/car4/Downloads/road/kga3.engine";
static const std::string SIGN_ENGINE_PATH = "/home/car4/Downloads/sign/signv4.engine";

// Speed / control
static constexpr float MAX_SPEED_CMS         = 30.0f;
static constexpr float MIN_SPEED_CMS         = 22.0f;
static constexpr float MAX_ACCEL_CMS2        = 30.0f;
static constexpr bool  SHOW_UI               = true;
static constexpr int   SIGN_INTERVAL_FRAMES  = 1;
static constexpr float SIGN_CONF_THRESH      = 0.70f;
static constexpr float SIGN_NMS_THRESH       = 0.45f;
static constexpr bool  DA_CHANNEL            = 1;
static constexpr int   MAX_STEER_DEG         = 50;
static constexpr bool  INVERT_STEER          = true;

// Servo calibration
static constexpr float SERVO_LEFT_SLOPE    = 0.96f;
static constexpr float SERVO_LEFT_OFFSET   = -6.2f;
static constexpr float SERVO_RIGHT_SLOPE   = 0.71f;
static constexpr float SERVO_RIGHT_OFFSET  = 2.5f;
static constexpr float MAX_REAL_ANGLE_DEG  = 45.0f;
static constexpr int   MAX_SERVO_CMD       = 55;

// Rear-facing ToF sensor slots
static constexpr int TOF_IDX_REAR_LEFT  = 1;
static constexpr int TOF_IDX_REAR_RIGHT = 3;
static constexpr int TOF_IDX_LEFT       = 1;
static constexpr int TOF_IDX_RIGHT      = 3;

// Heading source
static constexpr float HEADING_SIGN = 1.0f;

// Sign class indices
namespace sign_class {
    inline constexpr int CONSTRUCTION_AHEAD = 0;
    inline constexpr int NO_LEFT            = 1;
    inline constexpr int NO_RIGHT           = 2;
    inline constexpr int PARKING            = 3;
    inline constexpr int ROUNDABOUT         = 4;
    inline constexpr int TRAFFIC_LIGHT      = 5;
    inline constexpr int TURN_LEFT          = 6;
    inline constexpr int TURN_RIGHT         = 7;
    inline constexpr int WALKING_PERSON     = 8;
}

// Traffic light HSV
namespace tl {
    inline constexpr int RED_H1_LO = 0,   RED_H1_HI = 10;
    inline constexpr int RED_H2_LO = 170, RED_H2_HI = 180;
    inline constexpr int YEL_H_LO  = 15,  YEL_H_HI  = 35;
    inline constexpr int GRN_H_LO  = 40,  GRN_H_HI  = 90;

    inline constexpr int S_MIN     = 150;
    inline constexpr int S_MAX     = 255;
    inline constexpr int V_MIN     = 220;

    inline constexpr int S_WHITE_MAX = 50;
    inline constexpr int V_WHITE_MIN = 220;
    inline constexpr int MIN_WHITE_PIXELS = 10;

    inline constexpr int MIN_LIT_PIXELS = 10;
    inline constexpr float ROI_EXPAND   = 1.05f;
    inline constexpr bool TL_DEBUG      = false;
}

static constexpr double PERSON_STOP_HANGOVER_S = 2.0;
static constexpr double RED_LIGHT_HANGOVER_S   = 1.0;

static constexpr int SIGN_VOTER_ACTIVATE_FRAMES = 2;
static constexpr int SIGN_VOTER_TIMEOUT_DS      = 30;

static constexpr double SIGN_BLOCK_SECONDS      = 4.0;

// Roundabout timings (updated from roundabout.cpp)
static constexpr float  RB_BIAS_PX_RIGHT         = 25.0f;
static constexpr float  RB_BIAS_PX_LEFT          = 22.0f;
static constexpr float  RB_BIAS_EMA              = 0.30f;
static constexpr double RB_MISSING_SIGN_S        = 2.0;
static constexpr double RB_CIRCLE_TIMEOUT_S      = 8.0;
static constexpr double RB_EXIT_SEARCH_TIMEOUT_S = 15.0;
static constexpr double RB_EXIT_FORCE_EXEC_S     = 1.5;
static constexpr double RB_COOLDOWN_S            = 2.0;
static constexpr float  RB_SPEED_MIN_CMS         = 20.0f;
static constexpr float  RB_SPEED_MAX_CMS         = 30.0f;

// Roundabout exit turn params
static constexpr float  RB_EXIT_TURN_FLOOR_DEG   = 26.0f;
static constexpr float  RB_EXIT_TURN_BIAS_PX     = 30.0f;

// ============================================================
// PARKING & OBSTACLE CONFIGURATION (From chinh.cpp)
// ============================================================
static constexpr float PARK_TOF_BLOCKED_MM          = 350.0f;
static constexpr float PARK_APPROACH_SPEED          = 13.0f;
static constexpr float PARK_REVERSE_SPEED           = -10.0f;
static constexpr float PARK_MAX_STEER_DEG           = 45.0f;
static constexpr float PARK_TARGET_DELTA_DEG        = 50.0f;
static constexpr float PARK_ALIGN_TOL_DEG           = 3.0f;
static constexpr double PARK_STOP_DWELL_S           = 0.5;

static constexpr int   PARK_CORRIDOR_EXTRA_MARGIN_PX = 18;
static constexpr float PARK_CORRIDOR_FAR_RATIO       = 0.68f;
static constexpr float PARK_STEER_TRIM_DEG           = +2.5f;
static constexpr float PARK_HEADING_KP               = 0.90f;
static constexpr float PARK_HEADING_MAX_CORR_DEG     = 8.0f;

enum class ParkingState {
    IDLE = 0,
    APPROACH,
    SLOT1_WAIT_BLOCKED,
    SLOT2_SEARCH_FREE,
    SLOT3_WAIT_BLOCKED,
    WAIT_FULL_STOP,
    REVERSE_1,
    REVERSE_2,
    DONE
};

static const char* ParkingStateStr(ParkingState s) {
    switch (s) {
        case ParkingState::IDLE:               return "IDLE";
        case ParkingState::APPROACH:           return "APPROACH";
        case ParkingState::SLOT1_WAIT_BLOCKED: return "SLOT1_BLOCKED";
        case ParkingState::SLOT2_SEARCH_FREE:  return "SLOT2_FREE_SEARCH";
        case ParkingState::SLOT3_WAIT_BLOCKED: return "SLOT3_BLOCKED_STOP";
        case ParkingState::WAIT_FULL_STOP:     return "WAIT_STOP";
        case ParkingState::REVERSE_1:          return "REVERSE_1_TURN45";
        case ParkingState::REVERSE_2:          return "REVERSE_2_ALIGN";
        case ParkingState::DONE:               return "DONE";
        default:                               return "UNKNOWN";
    }
}

struct ParkingCorridor {
    bool latched = false;
    int y_bot = 299;
    int y_far = 135;

    int x_left_bot = 40;
    int x_left_far = 85;
    int x_right_bot = 440;
    int x_right_far = 395;

    int get_left_x(int y) const {
        if (y_bot == y_far) return x_left_bot;
        double t = static_cast<double>(y_bot - y) / static_cast<double>(y_bot - y_far);
        return static_cast<int>(std::lround(x_left_bot + t * (x_left_far - x_left_bot)));
    }

    int get_right_x(int y) const {
        if (y_bot == y_far) return x_right_bot;
        double t = static_cast<double>(y_bot - y) / static_cast<double>(y_bot - y_far);
        return static_cast<int>(std::lround(x_right_bot + t * (x_right_far - x_right_bot)));
    }
};

// ============================================================
// TELEMETRY
// ============================================================
static constexpr const char* TELEMETRY_SERVER_IP = "192.168.1.100";
static constexpr int         TELEMETRY_PORT      = 9999;
static constexpr const char* TELEMETRY_CAR_ID    = "MAYCAY";

volatile std::sig_atomic_t g_stopSignal = 0;
static void SignalHandler(int) { g_stopSignal = 1; }

// ============================================================
// TRT MODEL
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
// ROUNDABOUT STATE
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

static float normalizeAngleP(float a) {
    while (a >  180.0f) a -= 360.0f;
    while (a < -180.0f) a += 360.0f;
    return a;
}

// ============================================================
// SIGN COMMIT & LIGHT COLOR
// ============================================================
struct SignCommit {
    bool active   = false;
    int  side     = 0;
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
        default:         s.active = false; s.side = 0; break;
    }
    return s;
}

enum class LightColor { NONE, RED, YELLOW, GREEN };

static LightColor ClassifyTrafficLight(const cv::Mat& bgr, const cv::Rect& bbox) {
    using namespace tl;
    if (bgr.empty()) return LightColor::NONE;

    cv::Rect safe_bbox = bbox & cv::Rect(0, 0, bgr.cols, bgr.rows);
    if (safe_bbox.width < 4 || safe_bbox.height < 4) return LightColor::NONE;

    const int cx = safe_bbox.x + safe_bbox.width / 2;
    const int cy = safe_bbox.y + safe_bbox.height / 2;
    const int w  = static_cast<int>(safe_bbox.width  * ROI_EXPAND);
    const int h  = static_cast<int>(safe_bbox.height * ROI_EXPAND);

    const int x1 = std::max(0, cx - w / 2);
    const int y1 = std::max(0, cy - h / 2);
    const int x2 = std::min(bgr.cols, cx + w / 2);
    const int y2 = std::min(bgr.rows, cy + h / 2);

    int roi_w = x2 - x1;
    int roi_h = y2 - y1;
    if (roi_w < 4 || roi_h < 4) return LightColor::NONE;

    cv::Mat hsv;
    cv::cvtColor(bgr(cv::Rect(x1, y1, roi_w, roi_h)), hsv, cv::COLOR_BGR2HSV);

    if (roi_h > roi_w * 1.5) {
        cv::Mat v_chan;
        cv::extractChannel(hsv, v_chan, 2);

        int h_third = roi_h / 3;
        if (h_third < 1) return LightColor::NONE;

        int top_lit = cv::countNonZero(v_chan(cv::Rect(0, 0, roi_w, h_third)) > V_MIN);
        int mid_lit = cv::countNonZero(v_chan(cv::Rect(0, h_third, roi_w, h_third)) > V_MIN);
        int bot_lit = cv::countNonZero(v_chan(cv::Rect(0, 2 * h_third, roi_w, h_third)) > V_MIN);

        if (TL_DEBUG) std::printf("[TL] SLICE r=%d y=%d g=%d\n", top_lit, mid_lit, bot_lit);

        if (top_lit >= MIN_LIT_PIXELS && top_lit > mid_lit * 1.5 && top_lit > bot_lit * 1.5) return LightColor::RED;
        if (mid_lit >= MIN_LIT_PIXELS && mid_lit > top_lit * 1.5 && mid_lit > bot_lit * 1.5) return LightColor::YELLOW;
        if (bot_lit >= MIN_LIT_PIXELS && bot_lit > top_lit * 1.5 && bot_lit > mid_lit * 1.5) return LightColor::GREEN;
        return LightColor::NONE;
    }

    cv::Mat mRed1, mRed2, mRed;
    cv::inRange(hsv, cv::Scalar(RED_H1_LO, S_MIN, V_MIN), cv::Scalar(RED_H1_HI, S_MAX, 255), mRed1);
    cv::inRange(hsv, cv::Scalar(RED_H2_LO, S_MIN, V_MIN), cv::Scalar(RED_H2_HI, S_MAX, 255), mRed2);
    cv::bitwise_or(mRed1, mRed2, mRed);
    int nRed = cv::countNonZero(mRed);

    cv::Mat mYel;
    cv::inRange(hsv, cv::Scalar(YEL_H_LO, S_MIN, V_MIN), cv::Scalar(YEL_H_HI, S_MAX, 255), mYel);
    int nYel = cv::countNonZero(mYel);

    cv::Mat mGrn;
    cv::inRange(hsv, cv::Scalar(GRN_H_LO, S_MIN, V_MIN), cv::Scalar(GRN_H_HI, S_MAX, 255), mGrn);
    int nGrn = cv::countNonZero(mGrn);

    cv::Mat mWhite;
    cv::inRange(hsv, cv::Scalar(0, 0, V_WHITE_MIN), cv::Scalar(180, S_WHITE_MAX, 255), mWhite);
    int nWhite = cv::countNonZero(mWhite);

    if (nRed > 0 && nRed >= nYel && nRed >= nGrn) nRed += nWhite;
    else if (nGrn > 0 && nGrn > nRed && nGrn >= nYel) nGrn += nWhite;
    else if (nYel > 0 && nYel > nRed && nYel > nGrn) nYel += nWhite;

    if (TL_DEBUG) std::printf("[TL] SINGLE r=%d y=%d g=%d (w=%d)\n", nRed, nYel, nGrn, nWhite);

    if (nRed >= MIN_LIT_PIXELS && nRed > nYel && nRed > nGrn) return LightColor::RED;
    if (nYel >= MIN_LIT_PIXELS && nYel > nRed && nYel > nGrn) return LightColor::YELLOW;
    if (nGrn >= MIN_LIT_PIXELS && nGrn > nRed && nGrn > nYel) return LightColor::GREEN;

    return LightColor::NONE;
}

// ============================================================
// SERVO CALIBRATION
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
        case WALKING_PERSON:     return "walking_person";
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

    BNO055 imu;
    bool imu_ok = imu.initialize(std::chrono::milliseconds(1000));
    if (!imu_ok) {
        std::cerr << "[IMU] init FAILED: " << imu.lastError() << "\n";
    }

    sensor::TOF tof;
    bool tof_ok = tof.Init("/dev/i2c-1");
    if (!tof_ok) {
        std::cerr << "[TOF] init FAILED: " << tof.LastError() << "\n";
    }

    TrtModel roadModel;
    if (!roadModel.Init(ENGINE_PATH)) {
        std::cerr << "Road Model init failed\n"; return -1;
    }

    SignDetector signDetector;
    bool signReady = signDetector.Init(SIGN_ENGINE_PATH);
    if (!signReady) {
        std::cerr << "[Main] Sign Detector init failed\n";
    }

    telemetry::Helper telem;
    const bool telem_ready = telem.Init(TELEMETRY_SERVER_IP, TELEMETRY_PORT, TELEMETRY_CAR_ID);

    ucr::control::SignVoter voter(SIGN_VOTER_ACTIVATE_FRAMES, SIGN_VOTER_TIMEOUT_DS);

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
    v20::ObstacleAvoidanceController obstacle_controller;
    v20::LaneFollower::Output previous_out;

    const double DEFAULT_TURN_STEER_DEG = turn_controller.TURN_STEER_DEG;
    const double DEFAULT_TURN_BIAS_PX   = turn_controller.TURN_BIAS_PX;

    // Run-state
    RbState  rb_state              = RbState::NONE;
    RbState  rb_state_prev         = RbState::NONE;
    float    rb_bias_smoothed      = 0.0f;
    double   rb_enter_time         = 0.0;
    double   rb_exit_search_start  = 0.0;
    double   rb_exiting_start      = 0.0;
    double   rb_cooldown_until     = 0.0;
    int      rb_missing_sign_ticks = 0;
    int      rb_pass_count         = 0;

    double   last_person_seen_time = -1e9;
    double   last_red_seen_time    = -1e9;
    double   sign_block_until_time = -1.0;
    int      active_turn_sign_class = -1;

    // Parking State (From chinh.cpp)
    ParkingState park_state = ParkingState::IDLE;
    ParkingCorridor park_corridor;
    int park_free_side = -1;
    float park_heading_0 = 0.0f;
    float park_straight_heading = 0.0f;
    double park_dwell_time = 0.0;
    double park_reverse_elapsed = 0.0;
    int slot1_confirm_cnt = 0;
    int slot2_confirm_cnt = 0;
    int slot3_confirm_cnt = 0;
    float slot1_tof_left = -1.0f, slot1_tof_right = -1.0f;
    float slot2_tof_left = -1.0f, slot2_tof_right = -1.0f;
    float slot3_tof_left = -1.0f, slot3_tof_right = -1.0f;
    float tof_left = -1.0f, tof_right = -1.0f;
    float current_heading = 0.0f;
    bool heading_ok = false;

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

    std::cout << "\n=== UCR2026: Auto Run (Full Integrated) ===\n";

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
        if (dt_loop > 1.0)  dt_loop = 1.0;
        elapsed_total += dt_loop;
        totalFrames++;

        bool person_seen_this_frame = false;
        bool red_seen_this_frame    = false;
        bool green_seen_this_frame  = false;

        // 1. Sign detection + voter update
        if (signReady && (totalFrames % SIGN_INTERVAL_FRAMES) == 0) {
            cachedSigns = signDetector.Detect(frame, SIGN_CONF_THRESH, SIGN_NMS_THRESH);

            std::vector<int> classIds;
            classIds.reserve(cachedSigns.size());
            for (auto& d : cachedSigns) {
                if (d.confidence < SIGN_CONF_THRESH) continue;

                if (d.name == "walking_person" || d.class_id == sign_class::WALKING_PERSON) {
                    last_person_seen_time  = elapsed_total;
                    person_seen_this_frame = true;
                    continue;
                }
                if (d.name == "traffic_light" || d.class_id == sign_class::TRAFFIC_LIGHT) {
                    const LightColor lc = ClassifyTrafficLight(frame, d.box);
                    if (lc == LightColor::RED) {
                        last_red_seen_time  = elapsed_total;
                        red_seen_this_frame = true;
                    } else {
                        last_red_seen_time = -1e9;
                    }
                    if (lc == LightColor::GREEN) {
                        green_seen_this_frame = true;
                    }
                    continue;
                }

                int cls = d.class_id;
                if (cls == sign_class::CONSTRUCTION_AHEAD) cls = sign_class::TURN_LEFT;

                if (cls == sign_class::TURN_LEFT   ||
                    cls == sign_class::TURN_RIGHT  ||
                    cls == sign_class::NO_LEFT     ||
                    cls == sign_class::NO_RIGHT    ||
                    cls == sign_class::ROUNDABOUT  ||
                    cls == sign_class::PARKING) {
                    classIds.push_back(cls);
                }
            }
            uint32_t ds = static_cast<uint32_t>(elapsed_total * 10.0);
            voter.Update(classIds, ds);
        }

        // 2. RbState progression
        bool rb_visible_now = false;
        for (const auto& d : cachedSigns) {
            if ((d.name == "roundabout" || d.class_id == sign_class::ROUNDABOUT) &&
                d.confidence >= SIGN_CONF_THRESH) {
                rb_visible_now = true;
                break;
            }
        }

        const bool rb_committed = (voter.GetCommitted() == sign_class::ROUNDABOUT);

        if (rb_state == RbState::APPROACH || rb_state == RbState::EXIT_PREP) {
            if (rb_visible_now) rb_missing_sign_ticks = 0;
            else                ++rb_missing_sign_ticks;
        }

        switch (rb_state) {
            case RbState::NONE:
                if (rb_committed && elapsed_total >= rb_cooldown_until &&
                    turn_controller.state == v20::LaneTurnController::State::IDLE &&
                    !corner_assist.active()) {
                    rb_pass_count++;
                    if (rb_pass_count > 2) rb_pass_count = 2;
                    rb_state = RbState::APPROACH;
                    rb_missing_sign_ticks = 0;
                    rb_bias_smoothed      = 0.0f;
                    turn_controller.waiting_for_tjunction   = false;
                    turn_controller.tjunction_cooldown      = 0.0;
                    turn_controller.tjunction_detect_frames = 0;
                }
                break;
            case RbState::APPROACH:
                if (rb_missing_sign_ticks * dt_loop >= RB_MISSING_SIGN_S) {
                    rb_state = RbState::IN_CIRCLE;
                    rb_enter_time       = elapsed_total;
                }
                break;
            case RbState::IN_CIRCLE: {
                const bool timeout = (elapsed_total - rb_enter_time) >= RB_CIRCLE_TIMEOUT_S;
                const bool rb_seen_again = rb_committed && rb_visible_now;
                if (timeout || rb_seen_again) {
                    rb_state = RbState::EXIT_PREP;
                    rb_missing_sign_ticks = 0;
                }
                break;
            }
            case RbState::EXIT_PREP:
                if (rb_missing_sign_ticks * dt_loop >= RB_MISSING_SIGN_S) {
                    rb_state = RbState::EXIT_SEARCH;
                    rb_exit_search_start = elapsed_total;
                }
                break;
            case RbState::EXIT_SEARCH: {
                if (turn_controller.state == v20::LaneTurnController::State::IDLE) {
                    const char* exit_dir = (rb_pass_count == 1) ? "LEFT" : "RIGHT";
                    turn_controller.TURN_STEER_DEG = RB_EXIT_TURN_FLOOR_DEG;
                    turn_controller.TURN_BIAS_PX   = RB_EXIT_TURN_BIAS_PX;
                    turn_controller.start_turn(exit_dir, false);
                }
                const double in_exit = elapsed_total - rb_exit_search_start;
                if (in_exit >= RB_EXIT_FORCE_EXEC_S &&
                    (turn_controller.state == v20::LaneTurnController::State::WAIT_CURVE ||
                     turn_controller.state == v20::LaneTurnController::State::WAIT_STRAIGHT)) {
                    turn_controller.force_execute_now();
                }
                const bool moved_on = (turn_controller.state != v20::LaneTurnController::State::IDLE &&
                                       turn_controller.state != v20::LaneTurnController::State::WAIT_CURVE);
                const bool timed_out = in_exit >= RB_EXIT_SEARCH_TIMEOUT_S;
                if (moved_on || timed_out) {
                    rb_state = RbState::EXITING;
                    rb_exiting_start = elapsed_total;
                }
                break;
            }
            case RbState::EXITING: {
                const bool turn_done = turn_controller.state == v20::LaneTurnController::State::IDLE;
                const bool safety_timeout = (elapsed_total - rb_exiting_start) >= RB_CIRCLE_TIMEOUT_S;
                if (turn_done || safety_timeout) {
                    if (safety_timeout && !turn_done) {
                        turn_controller = v20::LaneTurnController{};
                        corner_assist.reset();
                    }
                    turn_controller.TURN_STEER_DEG = DEFAULT_TURN_STEER_DEG;
                    turn_controller.TURN_BIAS_PX   = DEFAULT_TURN_BIAS_PX;
                    rb_state = RbState::NONE;
                    rb_bias_smoothed = 0.0f;

                    // [roundabout.cpp] Arm T-junction after BOTH rb1 and rb2 exits.
                    //   rb1 exit → T-junction 1 (LEFT)
                    //   rb2 exit → T-junction 2 (RIGHT)
                    // turn_controller will fire it and return to IDLE → PID resumes.
                    turn_controller.waiting_for_tjunction = true;
                    turn_controller.tjunction_cooldown = turn_controller.TJUNCTION_COOLDOWN_S;

                    rb_cooldown_until = elapsed_total + RB_COOLDOWN_S;
                    voter.Reset();
                    active_turn_sign_class = -1;
                }
                break;
            }
        }

        // 3. Sensor reads
        heading_ok = false;
        if (imu_ok) {
            BNO055::Data imu_data;
            if (imu.read(imu_data)) {
                current_heading = HEADING_SIGN * static_cast<float>(imu_data.eulerDeg.x);
                heading_ok = true;
            }
        }

        tof_left = -1.0f;
        tof_right = -1.0f;
        if (tof_ok) {
            sensor::TOFReadings readings;
            tof.Read(readings, std::chrono::milliseconds(50));
            if (readings[TOF_IDX_LEFT].valid && readings[TOF_IDX_LEFT].distanceMm > 0)
                tof_left = static_cast<float>(readings[TOF_IDX_LEFT].distanceMm);
            if (readings[TOF_IDX_RIGHT].valid && readings[TOF_IDX_RIGHT].distanceMm > 0)
                tof_right = static_cast<float>(readings[TOF_IDX_RIGHT].distanceMm);
        }

        // 4. Road inference & Core logic
        bool roadOK = roadModel.Predict(frame, mask);
        v20::LaneFollower::Output out;
        float speed = MIN_SPEED_CMS;
        int   final_angle = 0;
        v20::LaneTurnController::TurnResult turn_res{v20::LaneTurnController::State::IDLE, 0.0, "NONE"};
        v20::ObstacleAvoidanceController::ObstacleResult obs_res;
        double applied_floor = 0.0;
        bool parking_active = (park_state != ParkingState::IDLE);

        std::vector<cv::Point> ui_left_pts, ui_right_pts;
        int ui_y1 = 0, ui_y2 = 0;

        if (roadOK) {
            int h = mask.rows, w = mask.cols;
            int y1 = static_cast<int>(h * 0.45);
            int y2 = static_cast<int>(h * 0.65);
            int y_bottom = h - 1;

            // --- PARKING: INIT & CORRIDOR LATCH ---
            if (voter.GetCommitted() == sign_class::PARKING && park_state == ParkingState::IDLE) {
                park_state = ParkingState::APPROACH;
                slot1_confirm_cnt = 0; slot2_confirm_cnt = 0; slot3_confirm_cnt = 0;
                if (heading_ok) park_straight_heading = current_heading;
            }

            if (parking_active && !park_corridor.latched) {
                auto get_raw_bounds = [&](const cv::Mat& m, int y_row, int& lx, int& rx) {
                    lx = -1; rx = -1;
                    if (m.empty() || y_row < 0 || y_row >= m.rows) return;
                    const uchar* row = m.ptr<uchar>(y_row);
                    int start = -1; int best_len = -1;
                    for (int x = 0; x <= m.cols; x++) {
                        bool on = (x < m.cols) && (row[x] > 0);
                        if (on && start < 0) start = x;
                        if (!on && start >= 0) {
                            int end_x = x - 1; int len = x - start;
                            if (len > 15 && len > best_len) { best_len = len; lx = start; rx = end_x; }
                            start = -1;
                        }
                    }
                };

                int raw_lx1 = -1, raw_rx1 = -1, raw_lx_bot = -1, raw_rx_bot = -1;
                get_raw_bounds(mask, y1, raw_lx1, raw_rx1);
                get_raw_bounds(mask, y_bottom, raw_lx_bot, raw_rx_bot);

                int clx_bot = (raw_lx_bot > 5 && raw_lx_bot < w / 2 - 10) ? raw_lx_bot : 50;
                int crx_bot = (raw_rx_bot > w / 2 + 10 && raw_rx_bot < w - 6) ? raw_rx_bot : (w - 51);

                int center_bot = (clx_bot + crx_bot) / 2;
                int half_w_bot = std::max((crx_bot - clx_bot) / 2 + PARK_CORRIDOR_EXTRA_MARGIN_PX, 185);
                int half_w_far = static_cast<int>(half_w_bot * PARK_CORRIDOR_FAR_RATIO);

                park_corridor.x_left_bot  = std::clamp(center_bot - half_w_bot, 5, w / 2 - 25);
                park_corridor.x_right_bot = std::clamp(center_bot + half_w_bot, w / 2 + 25, w - 6);
                park_corridor.x_left_far  = std::clamp(center_bot - half_w_far, 10, w / 2 - 15);
                park_corridor.x_right_far = std::clamp(center_bot + half_w_far, w / 2 + 15, w - 11);
                park_corridor.y_bot       = y_bottom;
                park_corridor.y_far       = y1;
                park_corridor.latched     = true;
                if (heading_ok) park_straight_heading = current_heading;
            }

            // Mask out everything outside the corridor
            cv::Mat active_mask = mask;
            cv::Mat park_mask;
            if (parking_active && park_corridor.latched) {
                park_mask = mask.clone();
                for (int y = 0; y < park_mask.rows; ++y) {
                    int xl = std::clamp(park_corridor.get_left_x(y), 0, park_mask.cols);
                    int xr = std::clamp(park_corridor.get_right_x(y), 0, park_mask.cols);
                    uchar* ptr = park_mask.ptr<uchar>(y);
                    if (xl > 0) std::memset(ptr, 0, xl);
                    if (xr < park_mask.cols) std::memset(ptr + xr, 0, park_mask.cols - xr);
                }
                active_mask = park_mask;
            }

            const int pref_side = turn_controller.get_preferred_side();
            auto get_mask_boundaries = [&](int y_row, int& lx, int& rx) {
                lx = -1; rx = -1;
                if (y_row < 0 || y_row >= h) return;
                const uchar* row = active_mask.ptr<uchar>(y_row);
                int start = -1; int best_len = -1;
                bool has_pref = false;
                const int half_w = w / 2;
                for (int x = 0; x <= w; x++) {
                    bool on = (x < w) && (row[x] > 0);
                    if (on && start < 0) start = x;
                    if (!on && start >= 0) {
                        int end_x = x - 1; int len = x - start;
                        if (len > 15) {
                            const int seg_center = (start + end_x) / 2;
                            bool is_pref = (pref_side < 0 && seg_center <= half_w) || (pref_side > 0 && seg_center >= half_w);
                            bool take = false;
                            if (pref_side != 0) {
                                if (is_pref && !has_pref) take = true;
                                else if (is_pref && has_pref) {
                                    if (pref_side < 0 && start < lx) take = true;
                                    if (pref_side > 0 && end_x > rx) take = true;
                                } else if (!has_pref && len > best_len) take = true;
                            } else if (len > best_len) take = true;

                            if (take) {
                                best_len = len; lx = start; rx = end_x;
                                if (is_pref) has_pref = true;
                            }
                        }
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

            ui_left_pts = left_pts; ui_right_pts = right_pts;
            ui_y1 = y1; ui_y2 = y2;

            const bool hazard_stop = ((elapsed_total - last_person_seen_time) <= PERSON_STOP_HANGOVER_S) ||
                                     ((elapsed_total - last_red_seen_time) <= RED_LIGHT_HANGOVER_S);

            // --- TURN / SIGN ACCEPTANCE ---
            SignCommit signCommit = MakeSignCommit(voter.GetCommitted());
            const bool can_accept_sign = !parking_active && (rb_state == RbState::NONE) && !corner_assist.active() &&
                (turn_controller.state == v20::LaneTurnController::State::IDLE ||
                 ((turn_controller.state == v20::LaneTurnController::State::WAIT_CURVE ||
                   turn_controller.state == v20::LaneTurnController::State::WAIT_STRAIGHT) &&
                  signCommit.active && active_turn_sign_class != signCommit.class_id));

            if (can_accept_sign && elapsed_total >= sign_block_until_time) {
                if (signCommit.active) {
                    std::string dir = (signCommit.side == -1) ? "LEFT" : "RIGHT";
                    turn_controller.start_turn(dir, false);
                    active_turn_sign_class = signCommit.class_id;
                    voter.Reset();
                }
            }

            if (!parking_active && !hazard_stop) {
                turn_res = turn_controller.update(left_pts, right_pts, dt_loop, previous_out.valid,
                                                  lx1, rx1, lx2, rx2, w, signCommit.active);
            } else {
                if (turn_controller.state == v20::LaneTurnController::State::WAIT_CURVE ||
                    turn_controller.state == v20::LaneTurnController::State::WAIT_STRAIGHT) {
                    turn_controller.straight_frames = 0;
                }
                turn_res.state = turn_controller.state;
                turn_res.direction = turn_controller.direction;
                turn_res.straight_frames = turn_controller.straight_frames;
                turn_res.t_junction_count = turn_controller.t_junction_count;
            }

            // --- OBSTACLE AVOIDANCE ---
            const bool turn_active_or_armed = (turn_controller.state != v20::LaneTurnController::State::IDLE) ||
                                              (rb_state != RbState::NONE) || (signCommit.active) || (active_turn_sign_class != -1);
            if (!parking_active && !hazard_stop && !turn_active_or_armed) {
                obs_res = obstacle_controller.update(active_mask, left_pts, right_pts,
                    lx1, rx1, lx2, rx2, lx_bot, rx_bot, w, dt_loop,
                    previous_out.near_e, previous_out.mid_e, previous_out.far_e);
            } else {
                obstacle_controller.reset();
            }

            // --- ROUNDABOUT BIAS ---
            float rb_bias_px = 0.0f;
            if (rb_state != RbState::NONE) {
                corner_assist.reset();
                switch (rb_state) {
                    case RbState::APPROACH:    rb_bias_px = +RB_BIAS_PX_RIGHT; break;
                    case RbState::IN_CIRCLE:   rb_bias_px = -RB_BIAS_PX_LEFT;  break;
                    case RbState::EXIT_PREP:   rb_bias_px = +RB_BIAS_PX_RIGHT; break;
                    case RbState::EXIT_SEARCH: rb_bias_px = 0.0f;              break;
                    default:                   rb_bias_px = 0.0f;              break;
                }
                rb_bias_smoothed = (1.0f - RB_BIAS_EMA) * rb_bias_smoothed + RB_BIAS_EMA * rb_bias_px;
            } else {
                rb_bias_smoothed = 0.0f;
            }

            // --- AGGREGATE BIAS & FLOOR ---
            double effective_bias      = (rb_state != RbState::NONE) ? rb_bias_smoothed : turn_res.steer_bias_px;
            double effective_floor     = turn_res.steer_floor_deg;
            int    effective_pref_side = turn_controller.get_preferred_side();
            bool   effective_is_turn   = (rb_state != RbState::NONE) || turn_res.lock_opposite_err;

            if (rb_state != RbState::NONE) {
                if (turn_controller.state != v20::LaneTurnController::State::IDLE) {
                    effective_bias += turn_res.steer_bias_px;
                }
            } else if (!parking_active && obstacle_controller.active()) {
                effective_bias      = obs_res.steer_bias_px;
                effective_floor     = obs_res.steer_floor_deg;
                effective_pref_side = obs_res.preferred_side;
                effective_is_turn   = true;
            } else {
                const bool allow_corner_assist = !parking_active && !hazard_stop && (
                    (turn_controller.state == v20::LaneTurnController::State::IDLE) ||
                    (turn_controller.is_unmarked_tjunction && turn_controller.state == v20::LaneTurnController::State::WAIT_CURVE &&
                     !turn_res.junction_opened && !(lx2 >= 0 && lx2 <= 15 && rx2 >= w - 15)));
                if (allow_corner_assist) {
                    const double cf = corner_assist.update(lx1, rx1, lx2, rx2, lx_bot, rx_bot, w, dt_loop,
                        (signCommit.active), previous_out.near_e, previous_out.mid_e, previous_out.far_e);
                    if (std::abs(cf) > std::abs(effective_floor)) effective_floor = cf;
                } else if (!hazard_stop) {
                    corner_assist.reset();
                }
            }
            applied_floor = effective_floor;

            // --- PARKING STATE MACHINE UPDATE ---
            const bool is_left_blocked  = (tof_left > 0.0f && tof_left < PARK_TOF_BLOCKED_MM);
            const bool is_right_blocked = (tof_right > 0.0f && tof_right < PARK_TOF_BLOCKED_MM);
            const bool is_left_free     = (tof_left < 0.0f || tof_left >= PARK_TOF_BLOCKED_MM);
            const bool is_right_free    = (tof_right < 0.0f || tof_right >= PARK_TOF_BLOCKED_MM);

            if (parking_active) {
                switch (park_state) {
                    case ParkingState::APPROACH:
                    case ParkingState::SLOT1_WAIT_BLOCKED: {
                        if (is_left_blocked && is_right_blocked) {
                            slot1_confirm_cnt++;
                            if (slot1_confirm_cnt >= 2) {
                                slot1_tof_left  = tof_left;
                                slot1_tof_right = tof_right;
                                park_state = ParkingState::SLOT2_SEARCH_FREE;
                            }
                        } else slot1_confirm_cnt = 0;
                        break;
                    }
                    case ParkingState::SLOT2_SEARCH_FREE: {
                        if (is_left_free && is_right_blocked) {
                            slot2_confirm_cnt++;
                            if (slot2_confirm_cnt >= 2) {
                                park_free_side = -1;
                                slot2_tof_left = tof_left; slot2_tof_right = tof_right;
                                park_state = ParkingState::SLOT3_WAIT_BLOCKED;
                            }
                        } else if (is_right_free && is_left_blocked) {
                            slot2_confirm_cnt++;
                            if (slot2_confirm_cnt >= 2) {
                                park_free_side = 1;
                                slot2_tof_left = tof_left; slot2_tof_right = tof_right;
                                park_state = ParkingState::SLOT3_WAIT_BLOCKED;
                            }
                        } else slot2_confirm_cnt = 0;
                        break;
                    }
                    case ParkingState::SLOT3_WAIT_BLOCKED: {
                        if (is_left_blocked && is_right_blocked) {
                            slot3_confirm_cnt++;
                            if (slot3_confirm_cnt >= 2) {
                                slot3_tof_left  = tof_left;
                                slot3_tof_right = tof_right;
                                if (heading_ok) park_heading_0 = current_heading;
                                park_dwell_time = 0.0;
                                park_state = ParkingState::WAIT_FULL_STOP;
                            }
                        } else slot3_confirm_cnt = 0;
                        break;
                    }
                    case ParkingState::WAIT_FULL_STOP: {
                        park_dwell_time += dt_loop;
                        if (heading_ok) park_heading_0 = current_heading;
                        if (park_dwell_time >= PARK_STOP_DWELL_S) {
                            park_state = ParkingState::REVERSE_1;
                            park_reverse_elapsed = 0.0;
                        }
                        break;
                    }
                    case ParkingState::REVERSE_1: {
                        park_reverse_elapsed += dt_loop;
                        float delta_h = std::abs(normalizeAngleP(current_heading - park_heading_0));
                        if (delta_h >= PARK_TARGET_DELTA_DEG || park_reverse_elapsed >= 7.0) {
                            park_state = ParkingState::REVERSE_2;
                            park_reverse_elapsed = 0.0;
                        }
                        break;
                    }
                    case ParkingState::REVERSE_2: {
                        park_reverse_elapsed += dt_loop;
                        float delta_h = std::abs(normalizeAngleP(current_heading - park_heading_0));
                        if (delta_h <= PARK_ALIGN_TOL_DEG || park_reverse_elapsed >= 7.0) {
                            park_state = ParkingState::DONE;
                        }
                        break;
                    }
                    default: break;
                }
            }

            // --- EXECUTE LANE FOLLOWER ---
            out = follower.step(
                active_mask,
                effective_bias,
                effective_is_turn,
                applied_floor,
                turn_res.steer_cap_deg,
                effective_pref_side,
                obs_res.lock_obstacle_dir,
                obs_res.obstacle_side
            );
            previous_out = out;

            if (turn_res.state == v20::LaneTurnController::State::DONE) {
                sign_block_until_time = elapsed_total + SIGN_BLOCK_SECONDS;
                voter.Reset();
                active_turn_sign_class = -1;
            }

            // --- FINAL SPEED & ANGLE ---
            if (parking_active) {
                if (park_state == ParkingState::WAIT_FULL_STOP || park_state == ParkingState::DONE) {
                    speed = 0.0f;
                    out.angle = 0.0f;
                } else if (park_state == ParkingState::REVERSE_1) {
                    speed = PARK_REVERSE_SPEED;
                    out.angle = (park_free_side < 0) ? -PARK_MAX_STEER_DEG : PARK_MAX_STEER_DEG;
                } else if (park_state == ParkingState::REVERSE_2) {
                    speed = PARK_REVERSE_SPEED;
                    out.angle = (park_free_side < 0) ? PARK_MAX_STEER_DEG : -PARK_MAX_STEER_DEG;
                } else {
                    speed = PARK_APPROACH_SPEED;
                    float heading_corr = 0.0f;
                    if (heading_ok) {
                        float h_err = normalizeAngleP(current_heading - park_straight_heading);
                        heading_corr = std::clamp(-PARK_HEADING_KP * h_err, -PARK_HEADING_MAX_CORR_DEG, PARK_HEADING_MAX_CORR_DEG);
                    }
                    out.angle = std::clamp(out.angle + PARK_STEER_TRIM_DEG + heading_corr, -16.0, 16.0);
                }
            } else if (hazard_stop) {
                speed = 0.0f;
                out.angle = 0.0f;
            } else {
                speed = static_cast<float>(out.speed);
                if (rb_state != RbState::NONE && out.valid) {
                    speed = std::clamp(speed, RB_SPEED_MIN_CMS, RB_SPEED_MAX_CMS);
                }
            }

        } else {
            previous_out = v20::LaneFollower::Output{};
            motor.SetSpeedCms(0.0f);
            servo.SetAngle(0);
            follower.resetTracking();
            applied_speed = 0.0f;
            frameCount++;
            auto tNow = std::chrono::high_resolution_clock::now();
            double dt_fps = std::chrono::duration<double>(tNow - tPrev).count();
            if (dt_fps > 0.5) { currentFps = frameCount / dt_fps; frameCount = 0; tPrev = tNow; }
            if (SHOW_UI) {
                cv::imshow("Lane Follower", frame);
                int key = cv::waitKey(1) & 0xFF;
                if (key == 27 || key == 'q') break;
            }
            continue;
        }

        // 6. Output Soft-start, Servo, Motor
        const float max_delta = MAX_ACCEL_CMS2 * static_cast<float>(dt_loop);
        if (speed > applied_speed)
            applied_speed = std::min(speed, applied_speed + max_delta);
        else
            applied_speed = speed;

        float desired_real_left = -static_cast<float>(out.angle);
        desired_real_left = std::clamp(desired_real_left, -MAX_REAL_ANGLE_DEG, MAX_REAL_ANGLE_DEG);
        float servo_cmd = RealAngleToServo(desired_real_left);
        servo_cmd = std::clamp(servo_cmd, -static_cast<float>(MAX_SERVO_CMD), static_cast<float>(MAX_SERVO_CMD));
        final_angle = static_cast<int>(std::lround(servo_cmd));

        motor.SetSpeedCms(applied_speed);
        servo.SetAngle(final_angle);

        // --- TELEMETRY ---
        if (telem_ready) {
            telemetry::StateInputs ti;
            ti.pedestrian_detected  = person_seen_this_frame;
            ti.red_light_detected   = red_seen_this_frame;
            ti.green_light_detected = green_seen_this_frame;
            ti.in_roundabout        = (rb_state != RbState::NONE);
            ti.parking_active       = (park_state != ParkingState::IDLE);
            ti.bridge_active        = false;
            ti.active_sign_class    = active_turn_sign_class;
            ti.last_seen_sign_class = voter.GetCommitted();

            const double now_telem_s = std::chrono::duration<double>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            telem.TrackInputs(ti, now_telem_s);
            telem.Tick(frame, applied_speed, final_angle, now_telem_s);
        }

        ++tickIdx;

        // 1 Hz debug print
        static auto lastLog = std::chrono::steady_clock::now();
        auto tNow = std::chrono::steady_clock::now();
        if (tNow - lastLog >= std::chrono::seconds(1)) {
            std::printf(
                "[t=%ld] sign=%-11s rb=%-11s park=%-11s near=%6.1f hdg=%7.2f tofL=%5.1f tofR=%5.1f "
                "spd=%.1f pid=%.1f srv=%d telem=%d(%s)\n",
                tickIdx, SignName(voter.GetCommitted()), RbStateName(rb_state),
                ParkingStateStr(park_state), out.near_e, current_heading, tof_left, tof_right,
                applied_speed, out.angle, final_angle, telem_ready ? telem.CurrentState() : -1,
                telem_ready ? telemetry::VehicleStateName(telem.CurrentState()) : "OFF");
            lastLog = tNow;
        }

        // FPS
        frameCount++;
        {
            auto tF = std::chrono::high_resolution_clock::now();
            double dt_fps = std::chrono::duration<double>(tF - tPrev).count();
            if (dt_fps > 0.5) { currentFps = frameCount / dt_fps; frameCount = 0; tPrev = tF; }
        }

        // --- UI ---
        if (SHOW_UI) {
            cv::Mat overlay = frame.clone();

            if (roadOK) {
                cv::Mat green(frame.size(), frame.type(), cv::Scalar(0, 255, 0));
                cv::Mat blended;
                cv::addWeighted(frame, 0.65, green, 0.35, 0, blended);
                blended.copyTo(overlay, mask);

                if (parking_active && park_corridor.latched) {
                    cv::line(overlay, cv::Point(park_corridor.x_left_bot, park_corridor.y_bot),
                                      cv::Point(park_corridor.x_left_far, park_corridor.y_far),
                                      cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
                    cv::line(overlay, cv::Point(park_corridor.x_right_bot, park_corridor.y_bot),
                                      cv::Point(park_corridor.x_right_far, park_corridor.y_far),
                                      cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
                }

                if (ui_y1 > 0 && ui_y2 > 0) {
                    cv::line(overlay, cv::Point(0, ui_y1), cv::Point(frame.cols - 1, ui_y1), cv::Scalar(0, 0, 255), 1);
                    cv::line(overlay, cv::Point(0, ui_y2), cv::Point(frame.cols - 1, ui_y2), cv::Scalar(0, 0, 255), 1);
                }
                for (size_t i = 0; i + 1 < ui_left_pts.size(); ++i)
                    cv::line(overlay, ui_left_pts[i], ui_left_pts[i + 1], cv::Scalar(255, 0, 0), 2);
                for (size_t i = 0; i + 1 < ui_right_pts.size(); ++i)
                    cv::line(overlay, ui_right_pts[i], ui_right_pts[i + 1], cv::Scalar(255, 0, 0), 2);

                if (obstacle_controller.active()) {
                    if (obs_res.y_cut > 0) {
                        cv::line(overlay, cv::Point(0, obs_res.y_cut), cv::Point(frame.cols - 1, obs_res.y_cut),
                                 cv::Scalar(0, 140, 255), 2, cv::LINE_AA);
                    }
                    if (obs_res.state == v20::ObstacleAvoidanceController::State::DODGE) {
                        std::string dodge_text = (obs_res.dodge_dir < 0) ? ">> DODGE LEFT <<" : ">> DODGE RIGHT <<";
                        cv::putText(overlay, dodge_text, cv::Point(static_cast<int>(frame.cols * 0.12), 110),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 165, 255), 2, cv::LINE_AA);
                    }
                }
            }

            char t1[200], t2[200], t3[200], t4[200];
            std::snprintf(t1, sizeof(t1), "FPS %.1f | spd %.1f/%.1f | pid %.1f | srv %d", currentFps, applied_speed, speed, out.angle, final_angle);
            std::snprintf(t2, sizeof(t2), "sign %s | rb %s | park %s", SignName(voter.GetCommitted()), RbStateName(rb_state), ParkingStateStr(park_state));
            std::snprintf(t3, sizeof(t3), "hdg %.2f | tof L=%.0f R=%.0f", current_heading, tof_left, tof_right);
            std::snprintf(t4, sizeof(t4), "near %.1f | obs %s", out.near_e, obstacle_controller.active() ? obs_res.state_str.c_str() : "NONE");

            cv::putText(overlay, t1, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 2);
            cv::putText(overlay, t2, {10, 55}, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 2);
            cv::putText(overlay, t3, {10, 80}, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 2);
            cv::putText(overlay, t4, {10,105}, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 2);

            cv::imshow("Lane Follower", overlay);
            const int key = cv::waitKey(1) & 0xFF;
            if (key == 27 || key == 'q') break;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    motor.SetSpeedCms(0.0f);
    servo.SetAngle(0);
    if (tof_ok) tof.Stop();
    if (imu_ok) imu.close();
    if (SHOW_UI) cv::destroyAllWindows();
    std::cout << "\n[System] Stopped safely\n";
    return 0;
}

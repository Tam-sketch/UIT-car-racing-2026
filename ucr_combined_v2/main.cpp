// ============================================================
// main_run.cpp — UCR2026 Full Integrated + Heading-only Parking
// ============================================================
// 2026-11-12 changes:
//  - Wired BNO055 IMU and TOF sensors
//  - EXIT_SEARCH bias flipped from LEFT to RIGHT (R1)
//  - RB_BIAS_EMA applied (R2)
//  - EXIT_SEARCH force-execute after RB_EXIT_FORCE_EXEC_S (R3)
//  - waiting_for_tjunction cleared on roundabout entry (R4)
//  - PARK_HEADING_SIGN applied to both heading-hold P loops
// 2026-11-12 (2-pass):
//  - rb_pass_count: Pass 1 = LEFT exit, Pass 2 = RIGHT exit
//  - EXIT_SEARCH turn params override (floor=16°, bias=18px)
//  - EXITING resets rb_bias_smoothed (hard reset)
// 2026-11-12 (P0/P1 fixes):
//  - [P0] Snapshot & restore TURN_STEER_DEG/TURN_BIAS_PX across roundabout
//  - [P0] Parking reverse: 28→38° steer, 4→6s timeout, servo settle
//  - [P1] HEADING_SIGN applied to raw axis; PARK_HEADING_SIGN only to P-ctrl
//  - [P1] MAX_REAL_ANGLE_DEG 30→45, PARK_HEADING_KP 1.2→2.2, WIN 10→4
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
static constexpr float MAX_REAL_ANGLE_DEG  = 45.0f;   // [P1] was 30 — allow 38° reverse arcs
static constexpr int   MAX_SERVO_CMD       = 55;

// Rear-facing ToF sensor slots
static constexpr int TOF_IDX_REAR_LEFT  = 1;
static constexpr int TOF_IDX_REAR_RIGHT = 3;

// Heading source (BNO055 eulerDeg.x is yaw on this mount)
// HEADING_SIGN flips the RAW axis only. PARK_HEADING_SIGN (below) flips
// the parking P-controller error sign. Do NOT mix the two roles.
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
    inline constexpr int S_MIN     = 50;
    inline constexpr int V_MIN     = 230;
    inline constexpr int MIN_LIT_PIXELS = 10;
    inline constexpr float ROI_EXPAND   = 1.05f;
}

static constexpr double PERSON_STOP_HANGOVER_S = 2.0;
static constexpr double RED_LIGHT_HANGOVER_S   = 1.0;

static constexpr int SIGN_VOTER_ACTIVATE_FRAMES = 2;
static constexpr int SIGN_VOTER_TIMEOUT_DS      = 30;

static constexpr double SIGN_BLOCK_SECONDS      = 4.0;

// Roundabout timings
static constexpr float  RB_BIAS_PX_RIGHT         = 28.0f;
static constexpr float  RB_BIAS_PX_LEFT          = 20.0f;
static constexpr float  RB_BIAS_EMA              = 0.30f;
static constexpr double RB_MISSING_SIGN_S        = 2.0;
static constexpr double RB_CIRCLE_TIMEOUT_S      = 12.0;
static constexpr double RB_EXIT_SEARCH_TIMEOUT_S = 15.0;
static constexpr double RB_EXIT_FORCE_EXEC_S     = 1.5;
static constexpr double RB_COOLDOWN_S            = 2.0;
static constexpr float  RB_SPEED_MIN_CMS         = 20.0f;
static constexpr float  RB_SPEED_MAX_CMS         = 30.0f;

// Roundabout exit turn params (gentler than regular turn)
static constexpr float  RB_EXIT_TURN_FLOOR_DEG   = 16.0f;
static constexpr float  RB_EXIT_TURN_BIAS_PX     = 18.0f;

// ============================================================
// PARKING CONSTANTS
// ============================================================
static constexpr int   PARK_HEADING_WINDOW         = 4;       // [P1] was 10 — less tilt
static constexpr float PARK_HEADING_KP             = 2.2f;    // [P1] was 1.2 — faster
static constexpr float PARK_HEADING_MAX_DEG        = 25.0f;   // [P1] was 20
static constexpr double PARK_CAPTURE_TIMEOUT_S     = 3.0;

static constexpr float PARK_TOF_BLOCKED_MM         = 300.0f;
static constexpr float PARK_TOF_HYST_MM            = 50.0f;

static constexpr float PARK_CAPTURE_SPEED_CMS      = 15.0f;
static constexpr float PARK_APPROACH_SPEED_CMS     = 10.0f;
static constexpr float PARK_REVERSE_SPEED_CMS      = -8.0f;

static constexpr float PARK_REVERSE_STEER_DEG      = 38.0f;   // [P0] was 28 — ≈12.5°/s
static constexpr float PARK_TARGET_HEADING_DEG     = 45.0f;
static constexpr float PARK_HEADING_TOL_DEG        = 3.0f;

static constexpr double PARK_STOP_DWELL_S          = 0.5;
static constexpr double PARK_REVERSE_TIMEOUT_S     = 6.0;     // [P0] was 4.0 — headroom
static constexpr double PARK_REVERSE_SETTLE_S      = 0.4;     // [P0] NEW — servo settle

// Parking P-controller sign flip (independent of HEADING_SIGN above).
// Leave +1 unless the car steers AWAY from straight during SLOT_SEARCH
// even though hdg / ref display correctly.
static constexpr float PARK_HEADING_SIGN           = +1.0f;

// ============================================================
// TELEMETRY (UCR 2026 competition requirement)
// ============================================================
static constexpr const char* TELEMETRY_SERVER_IP = "192.168.1.100";
static constexpr int         TELEMETRY_PORT      = 9999;
static constexpr const char* TELEMETRY_CAR_ID    = "UCR2026";

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

// ============================================================
// PARKING STATE
// ============================================================
enum class ParkState {
    NONE, CAPTURE, SLOT_SEARCH, SLOT_FOUND, WAIT_FULL_STOP,
    REVERSE_1, REVERSE_2, DONE
};

static const char* ParkStateName(ParkState s) {
    switch (s) {
        case ParkState::CAPTURE:        return "CAPTURE";
        case ParkState::SLOT_SEARCH:    return "SLOT_SEARCH";
        case ParkState::SLOT_FOUND:     return "SLOT_FOUND";
        case ParkState::WAIT_FULL_STOP: return "STOP";
        case ParkState::REVERSE_1:      return "REV_1";
        case ParkState::REVERSE_2:      return "REV_2";
        case ParkState::DONE:           return "DONE";
        default:                        return "NONE";
    }
}

static float normalizeAngleP(float a) {
    while (a >  180.0f) a -= 360.0f;
    while (a < -180.0f) a += 360.0f;
    return a;
}

// Shift `angle` by a multiple of 360 so that `angle - ref` ∈ (-180, 180].
static float unwrapNear(float ref, float angle) {
    float d = angle - ref;
    while (d >  180.0f) d -= 360.0f;
    while (d < -180.0f) d += 360.0f;
    return ref + d;
}

// Wrap-safe median of heading samples (deg). Result in [0, 360).
static float medianHeadingDeg(const std::vector<float>& samples) {
    if (samples.empty()) return 0.0f;
    const float ref = samples.front();
    std::vector<float> unwrapped;
    unwrapped.reserve(samples.size());
    for (float s : samples) unwrapped.push_back(unwrapNear(ref, s));
    std::sort(unwrapped.begin(), unwrapped.end());
    float median = unwrapped[unwrapped.size() / 2];
    while (median <    0.0f) median += 360.0f;
    while (median >= 360.0f) median -= 360.0f;
    return median;
}

// ============================================================
// SIGN COMMIT
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

    // =========================================================
    // BNO055 IMU  (non-fatal: parking disabled if it fails)
    // =========================================================
    BNO055 imu;
    bool imu_ok = imu.initialize(std::chrono::milliseconds(1000));
    if (!imu_ok) {
        std::cerr << "[IMU] init FAILED: " << imu.lastError()
                  << " (parking DISABLED)\n";
    }

    // =========================================================
    // TOF cluster  (non-fatal: parking disabled if it fails)
    // =========================================================
    sensor::TOF tof;
    bool tof_ok = tof.Init("/dev/i2c-1");
    if (!tof_ok) {
        std::cerr << "[TOF] init FAILED: " << tof.LastError()
                  << " (parking DISABLED)\n";
    }

    TrtModel roadModel;
    if (!roadModel.Init(ENGINE_PATH)) {
        std::cerr << "Road Model init failed\n"; return -1;
    }

    SignDetector signDetector;
    bool signReady = signDetector.Init(SIGN_ENGINE_PATH);
    if (!signReady) {
        std::cerr << "[Main] Sign Detector init failed — turns/roundabout/parking DISABLED\n";
    }

    // Telemetry
    telemetry::Helper telem;
    const bool telem_ready = telem.Init(TELEMETRY_SERVER_IP, TELEMETRY_PORT, TELEMETRY_CAR_ID);
    if (!telem_ready) {
        std::cerr << "[TELEM] Init FAILED (bad IP or socket) — telemetry DISABLED\n";
    } else {
        std::cout << "[TELEM] Init OK → " << TELEMETRY_SERVER_IP
                  << ":" << TELEMETRY_PORT
                  << " car='" << TELEMETRY_CAR_ID << "'\n";
    }

    ucr::control::SignVoter voter(
        SIGN_VOTER_ACTIVATE_FRAMES,
        SIGN_VOTER_TIMEOUT_DS);

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

    // [P0] Snapshot default turn params — they get overridden for the
    // roundabout exit and must be restored after EXITING so subsequent
    // 90° turns (loop-back T-junction, sign-driven turns) use aggressive
    // defaults (32°, 35px) not the gentle exit params (16°, 18px).
    const double DEFAULT_TURN_STEER_DEG = turn_controller.TURN_STEER_DEG;
    const double DEFAULT_TURN_BIAS_PX   = turn_controller.TURN_BIAS_PX;

    // Run-state
    RbState  rb_state              = RbState::NONE;
    RbState  rb_state_prev         = RbState::NONE;
    float    rb_bias_smoothed      = 0.0f;
    double   rb_enter_time         = 0.0;
    double   rb_phase_start_time   = 0.0;
    double   rb_exit_search_start  = 0.0;
    double   rb_exiting_start      = 0.0;
    double   rb_cooldown_until     = 0.0;
    int      rb_missing_sign_ticks = 0;
    double   last_rb_visible_time  = -1e9;
    int      rb_pass_count         = 0;   // 0=idle, 1=pass 1, 2=pass 2

    double   last_person_seen_time = -1e9;
    double   last_red_seen_time    = -1e9;
    double   sign_block_until_time = -1.0;

    int      active_turn_sign_class = -1;

    // Parking state
    ParkState park_state              = ParkState::NONE;
    std::deque<float> park_heading_buf;
    float     park_ref_heading        = 0.0f;
    float     park_heading_0          = 0.0f;
    int       park_side               = 0;
    bool      left_blocked_h          = false;
    bool      right_blocked_h         = false;
    double    park_capture_elapsed    = 0.0;
    double    park_stop_dwell_elapsed = 0.0;
    double    park_reverse_elapsed    = 0.0;
    float     park_angle_override     = 0.0f;
    float     park_speed_override     = 0.0f;

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

    std::cout << "\n=== UCR2026: Full Integrated (v20 + RbState + SignVoter + Parking) ===\n"
              << "Road: " << ENGINE_PATH << "\n"
              << "Sign: " << SIGN_ENGINE_PATH << (signReady ? "" : " (DISABLED)") << "\n"
              << "IMU:  " << (imu_ok ? "OK" : "FAILED — parking disabled") << "\n"
              << "TOF:  " << (tof_ok ? "OK" : "FAILED — parking disabled") << "\n"
              << "TELEM:" << (telem_ready ? " OK" : " DISABLED")
              << " → " << TELEMETRY_SERVER_IP << ":" << TELEMETRY_PORT << "\n"
              << "Speed: [" << MIN_SPEED_CMS << ", " << MAX_SPEED_CMS << "] cm/s\n"
              << "Roundabout: 2-pass exit floor=" << RB_EXIT_TURN_FLOOR_DEG
              << "° bias=" << RB_EXIT_TURN_BIAS_PX << "px (restored after EXITING)\n"
              << "Parking: heading-hold, window=" << PARK_HEADING_WINDOW
              << " kp=" << PARK_HEADING_KP
              << " rev_steer=" << PARK_REVERSE_STEER_DEG
              << "° settle=" << PARK_REVERSE_SETTLE_S << "s"
              << " tof_blk=" << PARK_TOF_BLOCKED_MM << "mm"
              << " HEADING_SIGN=" << (HEADING_SIGN > 0 ? "+1" : "-1")
              << " PARK_HEADING_SIGN=" << (PARK_HEADING_SIGN > 0 ? "+1" : "-1") << "\n"
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
        if (dt_loop > 1.0)  dt_loop = 1.0;
        elapsed_total += dt_loop;
        totalFrames++;

        bool person_seen_this_frame = false;
        bool red_seen_this_frame    = false;
        bool green_seen_this_frame  = false;

        // -----------------------------------------------------
        // 1. Sign detection + voter update
        // -----------------------------------------------------
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
                    rb_pass_count++;
                    if (rb_pass_count > 2) rb_pass_count = 2;
                    std::printf("[RB] === Starting pass %d/2 ===\n", rb_pass_count);

                    rb_state = RbState::APPROACH;
                    rb_missing_sign_ticks = 0;
                    rb_phase_start_time   = elapsed_total;
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
                if (turn_controller.state == v20::LaneTurnController::State::IDLE) {
                    const char* exit_dir = (rb_pass_count == 1) ? "LEFT" : "RIGHT";
                    turn_controller.TURN_STEER_DEG = RB_EXIT_TURN_FLOOR_DEG;
                    turn_controller.TURN_BIAS_PX   = RB_EXIT_TURN_BIAS_PX;

                    std::printf("[RB] Pass %d/2 EXIT_SEARCH → start_turn(%s) floor=%.0f° bias=%.0fpx\n",
                                rb_pass_count, exit_dir,
                                turn_controller.TURN_STEER_DEG,
                                turn_controller.TURN_BIAS_PX);

                    turn_controller.start_turn(exit_dir, false);
                }
                const double in_exit = elapsed_total - rb_exit_search_start;
                if (in_exit >= RB_EXIT_FORCE_EXEC_S &&
                    (turn_controller.state == v20::LaneTurnController::State::WAIT_CURVE ||
                     turn_controller.state == v20::LaneTurnController::State::WAIT_STRAIGHT)) {
                    turn_controller.force_execute_now();
                    std::printf("[RB] EXIT_SEARCH force-execute at %.1fs\n", in_exit);
                }
                const bool moved_on =
                    turn_controller.state == v20::LaneTurnController::State::WAIT_STRAIGHT ||
                    turn_controller.state == v20::LaneTurnController::State::EXECUTE    ||
                    turn_controller.state == v20::LaneTurnController::State::SETTLE;
                const bool timed_out = in_exit >= RB_EXIT_SEARCH_TIMEOUT_S;
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
                        turn_controller = v20::LaneTurnController{};
                        corner_assist.reset();
                    }

                    // [P0] Restore default aggressive turn params so subsequent
                    // 90° turns (loop-back T-junction, sign turns) are not
                    // limited to the gentle roundabout-exit values.
                    turn_controller.TURN_STEER_DEG = DEFAULT_TURN_STEER_DEG;
                    turn_controller.TURN_BIAS_PX   = DEFAULT_TURN_BIAS_PX;

                    rb_state = RbState::NONE;
                    rb_bias_smoothed = 0.0f;

                    if (rb_pass_count == 1) {
                        turn_controller.waiting_for_tjunction = true;
                        turn_controller.tjunction_cooldown = turn_controller.TJUNCTION_COOLDOWN_S;
                        std::printf("[RB] Pass 1 done — armed T-junction for loop-back "
                                    "(turn defaults restored: %.0f° / %.0fpx)\n",
                                    turn_controller.TURN_STEER_DEG,
                                    turn_controller.TURN_BIAS_PX);
                    } else {
                        turn_controller.waiting_for_tjunction = false;
                        std::printf("[RB] Pass 2 done — roundabout sequence COMPLETE "
                                    "(turn defaults restored: %.0f° / %.0fpx)\n",
                                    turn_controller.TURN_STEER_DEG,
                                    turn_controller.TURN_BIAS_PX);
                        rb_pass_count = 0;
                    }

                    rb_cooldown_until = elapsed_total + RB_COOLDOWN_S;
                    voter.Reset();
                    active_turn_sign_class = -1;
                }
                break;
            }
        }

        if (rb_state != rb_state_prev) {
            std::printf("[RB] %-10s -> %-10s t=%.2f sign=%s bias=%.1f pass=%d\n",
                        RbStateName(rb_state_prev), RbStateName(rb_state),
                        elapsed_total, SignName(voter.GetCommitted()),
                        rb_bias_smoothed, rb_pass_count);
            rb_state_prev = rb_state;
        }

        // -----------------------------------------------------
        // 3. Sensor reads
        // -----------------------------------------------------
        float heading_now   = 0.0f;
        bool  heading_ok    = false;
        float tof_left_mm   = -1.0f;
        float tof_right_mm  = -1.0f;

        if (imu_ok) {
            BNO055::Data imu_data;
            if (imu.read(imu_data)) {
                // [P1] HEADING_SIGN flips the raw axis only. PARK_HEADING_SIGN
                // (further below) flips the P-controller error sign — the two
                // are independent so either can be adjusted without breaking
                // the other.
                heading_now = HEADING_SIGN * static_cast<float>(imu_data.eulerDeg.x);
                heading_ok  = true;
            }
        }

        if (tof_ok) {
            sensor::TOFReadings readings;
            tof.Read(readings, std::chrono::milliseconds(50));
            if (readings[TOF_IDX_REAR_LEFT].valid &&
                readings[TOF_IDX_REAR_LEFT].distanceMm > 0)
                tof_left_mm  = static_cast<float>(
                    readings[TOF_IDX_REAR_LEFT].distanceMm);
            if (readings[TOF_IDX_REAR_RIGHT].valid &&
                readings[TOF_IDX_REAR_RIGHT].distanceMm > 0)
                tof_right_mm = static_cast<float>(
                    readings[TOF_IDX_REAR_RIGHT].distanceMm);
        }

        // -----------------------------------------------------
        // 4. Parking state machine
        // -----------------------------------------------------
        if (park_state == ParkState::NONE &&
            rb_state == RbState::NONE &&
            turn_controller.state == v20::LaneTurnController::State::IDLE &&
            !corner_assist.active() &&
            voter.GetCommitted() == sign_class::PARKING) {
            park_state = ParkState::CAPTURE;
            park_heading_buf.clear();
            park_capture_elapsed = 0.0;
            std::printf("[PARK] Parking sign → CAPTURE\n");
        }

        park_angle_override = 0.0f;
        park_speed_override = 0.0f;

        switch (park_state) {
            case ParkState::CAPTURE: {
                park_capture_elapsed += dt_loop;
                if (heading_ok) {
                    park_heading_buf.push_back(heading_now);
                    if ((int)park_heading_buf.size() > PARK_HEADING_WINDOW)
                        park_heading_buf.pop_front();
                }
                if ((int)park_heading_buf.size() >= PARK_HEADING_WINDOW) {
                    std::vector<float> samples(park_heading_buf.begin(),
                                               park_heading_buf.end());
                    park_ref_heading = medianHeadingDeg(samples);
                    park_state = ParkState::SLOT_SEARCH;
                    std::printf("[PARK] Reference captured: %.2f° → SLOT_SEARCH\n",
                                park_ref_heading);
                } else if (park_capture_elapsed >= PARK_CAPTURE_TIMEOUT_S) {
                    if (park_heading_buf.empty()) {
                        std::printf("[PARK] CAPTURE timeout, no samples → ABORT\n");
                        park_state = ParkState::NONE;
                    } else {
                        std::vector<float> samples(park_heading_buf.begin(),
                                                   park_heading_buf.end());
                        park_ref_heading = medianHeadingDeg(samples);
                        park_state = ParkState::SLOT_SEARCH;
                        std::printf("[PARK] CAPTURE timeout, using %zu samples: %.2f° → SLOT_SEARCH\n",
                                    park_heading_buf.size(), park_ref_heading);
                    }
                }
                break;
            }
            case ParkState::SLOT_SEARCH: {
                if (tof_left_mm > 0 && tof_right_mm > 0) {
                    bool lb = left_blocked_h
                        ? (tof_left_mm  < PARK_TOF_BLOCKED_MM + PARK_TOF_HYST_MM)
                        : (tof_left_mm  < PARK_TOF_BLOCKED_MM - PARK_TOF_HYST_MM);
                    bool rb = right_blocked_h
                        ? (tof_right_mm < PARK_TOF_BLOCKED_MM + PARK_TOF_HYST_MM)
                        : (tof_right_mm < PARK_TOF_BLOCKED_MM - PARK_TOF_HYST_MM);
                    left_blocked_h  = lb;
                    right_blocked_h = rb;
                    if (lb != rb) {
                        park_side = lb ? -1 : +1;
                        park_state = ParkState::SLOT_FOUND;
                        std::printf("[PARK] Slot detected (%s) → SLOT_FOUND\n",
                                    lb ? "LEFT" : "RIGHT");
                    }
                }
                if (heading_ok) {
                    float herr = PARK_HEADING_SIGN * normalizeAngleP(park_ref_heading - heading_now);
                    double a = PARK_HEADING_KP * herr;
                    a = std::clamp(a, -(double)PARK_HEADING_MAX_DEG, (double)PARK_HEADING_MAX_DEG);
                    park_angle_override = (float)a;
                }
                park_speed_override = PARK_APPROACH_SPEED_CMS;
                break;
            }
            case ParkState::SLOT_FOUND: {
                if (tof_left_mm > 0 && tof_right_mm > 0) {
                    bool lb = left_blocked_h
                        ? (tof_left_mm  < PARK_TOF_BLOCKED_MM + PARK_TOF_HYST_MM)
                        : (tof_left_mm  < PARK_TOF_BLOCKED_MM - PARK_TOF_HYST_MM);
                    bool rb = right_blocked_h
                        ? (tof_right_mm < PARK_TOF_BLOCKED_MM + PARK_TOF_HYST_MM)
                        : (tof_right_mm < PARK_TOF_BLOCKED_MM - PARK_TOF_HYST_MM);
                    left_blocked_h  = lb;
                    right_blocked_h = rb;
                    if (lb && rb) {
                        park_heading_0 = heading_now;
                        park_stop_dwell_elapsed = 0.0;
                        park_state = ParkState::WAIT_FULL_STOP;
                        std::printf("[PARK] Both blocked, heading_0=%.2f° → STOP\n",
                                    park_heading_0);
                    }
                }
                if (heading_ok) {
                    float herr = PARK_HEADING_SIGN * normalizeAngleP(park_ref_heading - heading_now);
                    double a = PARK_HEADING_KP * herr;
                    a = std::clamp(a, -(double)PARK_HEADING_MAX_DEG, (double)PARK_HEADING_MAX_DEG);
                    park_angle_override = (float)a;
                }
                park_speed_override = PARK_APPROACH_SPEED_CMS;
                break;
            }
            case ParkState::WAIT_FULL_STOP: {
                park_angle_override = 0.0f;
                park_speed_override = 0.0f;
                park_stop_dwell_elapsed += dt_loop;
                if (park_stop_dwell_elapsed >= PARK_STOP_DWELL_S) {
                    park_reverse_elapsed = 0.0;
                    park_state = ParkState::REVERSE_1;
                    std::printf("[PARK] Stopped → REVERSE_1 (side=%s)\n",
                                park_side < 0 ? "LEFT" : "RIGHT");
                }
                break;
            }
            case ParkState::REVERSE_1: {
                park_reverse_elapsed += dt_loop;
                park_angle_override = (park_side < 0) ? -PARK_REVERSE_STEER_DEG
                                                      :  PARK_REVERSE_STEER_DEG;

                // [P0] Servo-settle: hold speed 0 until the servo has had
                // time to swing to the commanded angle, so the reverse arc
                // starts with wheels already turned (previously the first
                // ~0.3s of reversing happened mid-swing, shrinking the arc).
                if (park_reverse_elapsed < PARK_REVERSE_SETTLE_S) {
                    park_speed_override = 0.0f;
                } else {
                    park_speed_override = PARK_REVERSE_SPEED_CMS;
                    if (heading_ok) {
                        float d = std::abs(normalizeAngleP(heading_now - park_heading_0));
                        if (d >= PARK_TARGET_HEADING_DEG) {
                            park_state = ParkState::REVERSE_2;
                            park_reverse_elapsed = 0.0;
                            std::printf("[PARK] REVERSE_1 done (Δ=%.1f°) → REVERSE_2\n", d);
                        }
                    }
                }
                if (park_reverse_elapsed >= PARK_REVERSE_TIMEOUT_S) {
                    float d_actual = heading_ok
                        ? std::abs(normalizeAngleP(heading_now - park_heading_0))
                        : -1.0f;
                    std::printf("[PARK] REVERSE_1 TIMEOUT (Δ=%.1f°) → REVERSE_2\n", d_actual);
                    park_state = ParkState::REVERSE_2;
                    park_reverse_elapsed = 0.0;
                }
                break;
            }
            case ParkState::REVERSE_2: {
                park_reverse_elapsed += dt_loop;
                park_angle_override = (park_side < 0) ?  PARK_REVERSE_STEER_DEG
                                                      : -PARK_REVERSE_STEER_DEG;

                if (park_reverse_elapsed < PARK_REVERSE_SETTLE_S) {
                    park_speed_override = 0.0f;
                } else {
                    park_speed_override = PARK_REVERSE_SPEED_CMS;
                    if (heading_ok) {
                        float d = std::abs(normalizeAngleP(heading_now - park_heading_0));
                        if (d <= PARK_HEADING_TOL_DEG) {
                            park_state = ParkState::DONE;
                            std::printf("[PARK] REVERSE_2 done (Δ=%.1f°) → DONE\n", d);
                        }
                    }
                }
                if (park_reverse_elapsed >= PARK_REVERSE_TIMEOUT_S) {
                    float d_actual = heading_ok
                        ? std::abs(normalizeAngleP(heading_now - park_heading_0))
                        : -1.0f;
                    std::printf("[PARK] REVERSE_2 TIMEOUT (Δ=%.1f°) → DONE\n", d_actual);
                    park_state = ParkState::DONE;
                }
                break;
            }
            case ParkState::DONE: {
                park_angle_override = 0.0f;
                park_speed_override = 0.0f;
                break;
            }
            default: break;
        }

        // -----------------------------------------------------
        // 5. Road inference
        // -----------------------------------------------------
        bool roadOK = roadModel.Predict(frame, mask);

        v20::LaneFollower::Output out;
        float speed = MIN_SPEED_CMS;
        int   final_angle = 0;
        v20::LaneTurnController::TurnResult turn_res{
            v20::LaneTurnController::State::IDLE, 0.0, "NONE"};
        double applied_floor = 0.0;
        double applied_cap   = 0.0;
        float  final_bias    = 0.0f;
        bool   exec_turn     = false;

        std::vector<cv::Point> ui_left_pts, ui_right_pts;
        int ui_y1 = 0, ui_y2 = 0;
        int ui_lx1 = -1, ui_rx1 = -1, ui_lx2 = -1, ui_rx2 = -1;
        int ui_lx_bot = -1, ui_rx_bot = -1;

        const bool parking_active = (park_state != ParkState::NONE);

        if (parking_active && park_state == ParkState::CAPTURE) {
            if (roadOK) {
                out = follower.step(mask, 0.0, false, 0.0, 0.0, /*park_mode=*/true);
                speed = PARK_CAPTURE_SPEED_CMS;
                previous_out = out;
            } else {
                out.angle = 0.0;
                speed = 0.0f;
            }
        } else if (parking_active) {
            out.angle = park_angle_override;
            speed = park_speed_override;
            out.valid = true;
            previous_out.angle = out.angle;
            previous_out.valid = true;
        } else if (roadOK) {
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

            SignCommit signCommit = MakeSignCommit(voter.GetCommitted());

            if (rb_state == RbState::NONE &&
                signCommit.active &&
                turn_controller.state == v20::LaneTurnController::State::IDLE &&
                !corner_assist.active() &&
                elapsed_total >= sign_block_until_time) {
                std::string dir = (signCommit.side == -1) ? "LEFT" : "RIGHT";
                turn_controller.start_turn(dir, false);
                active_turn_sign_class = signCommit.class_id;
                voter.Reset();
            }

            turn_res = turn_controller.update(
                left_pts, right_pts, dt_loop, previous_out.valid,
                lx1, rx1, lx2, rx2, w,
                signCommit.active
            );

            applied_floor = turn_res.steer_floor_deg;
            applied_cap   = turn_res.steer_cap_deg;

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

            // ---- Roundabout bias table (EMA-smoothed) ----
            float rb_bias_px = 0.0f;
            switch (rb_state) {
                case RbState::APPROACH:    rb_bias_px = +RB_BIAS_PX_RIGHT; break;
                case RbState::IN_CIRCLE:   rb_bias_px = -RB_BIAS_PX_LEFT;  break;
                case RbState::EXIT_PREP:   rb_bias_px = +RB_BIAS_PX_RIGHT; break;
                case RbState::EXIT_SEARCH: rb_bias_px = 0.0f;              break;
                default:                   rb_bias_px =  0.0f;             break;
            }
            rb_bias_smoothed = (1.0f - RB_BIAS_EMA) * rb_bias_smoothed + RB_BIAS_EMA * rb_bias_px;
            final_bias = rb_bias_smoothed;
            if (turn_controller.state != v20::LaneTurnController::State::IDLE) {
                final_bias += turn_res.steer_bias_px;
            }
            exec_turn = (rb_state != RbState::NONE) || turn_res.lock_opposite_err;

            if (turn_res.state == v20::LaneTurnController::State::DONE) {
                sign_block_until_time = elapsed_total + SIGN_BLOCK_SECONDS;
                voter.Reset();
                active_turn_sign_class = -1;
            }

            out = follower.step(mask, final_bias, exec_turn, applied_floor, applied_cap);
            previous_out = out;

            speed = static_cast<float>(out.speed);
            if (rb_state != RbState::NONE && out.valid) {
                speed = std::clamp(speed, RB_SPEED_MIN_CMS, RB_SPEED_MAX_CMS);
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

        // -----------------------------------------------------
        // 6. Stop overrides, soft-start, servo, motor
        // -----------------------------------------------------
        if (!parking_active) {
            const bool stopForPerson =
                (elapsed_total - last_person_seen_time) <= PERSON_STOP_HANGOVER_S;
            const bool stopForRedLight =
                (elapsed_total - last_red_seen_time) <= RED_LIGHT_HANGOVER_S;
            if (stopForPerson || stopForRedLight) speed = 0.0f;
        }

        const float max_delta = MAX_ACCEL_CMS2 * static_cast<float>(dt_loop);
        if (speed > applied_speed)
            applied_speed = std::min(speed, applied_speed + max_delta);
        else
            applied_speed = speed;

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

        // -----------------------------------------------------
        // Telemetry (UCR 2026)
        // -----------------------------------------------------
        if (telem_ready) {
            telemetry::StateInputs ti;
            ti.pedestrian_detected  = person_seen_this_frame;
            ti.red_light_detected   = red_seen_this_frame;
            ti.green_light_detected = green_seen_this_frame;
            ti.in_roundabout        = (rb_state != RbState::NONE);
            ti.parking_active       = (park_state != ParkState::NONE);
            ti.bridge_active        = false;
            ti.active_sign_class    = active_turn_sign_class;
            ti.last_seen_sign_class = voter.GetCommitted();

            const double now_telem_s = std::chrono::duration<double>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            telem.TrackInputs(ti, now_telem_s);
            telem.Tick(frame, applied_speed, final_angle, now_telem_s);
        }

        ++tickIdx;

        // 1 Hz debug
        static auto lastLog = std::chrono::steady_clock::now();
        auto tNow = std::chrono::steady_clock::now();
        if (tNow - lastLog >= std::chrono::seconds(1)) {
            std::printf(
                "[t=%ld] sign=%-11s(%d) rb=%-11s(p%d) park=%-11s "
                "near=%6.1f hdg=%7.2f ref=%7.2f tofL=%5.1f tofR=%5.1f "
                "spd=%.1f pid=%.1f srv=%d telem=%d(%s)\n",
                tickIdx,
                SignName(voter.GetCommitted()),
                voter.GetCommitted(),
                RbStateName(rb_state),
                ParkStateName(park_state),
                out.near_e, heading_now, park_ref_heading,
                tof_left_mm, tof_right_mm,
                applied_speed, out.angle, final_angle,
                telem_ready ? telem.CurrentState() : -1,
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

        // UI
        if (SHOW_UI) {
            cv::Mat overlay = frame.clone();

            if (roadOK && parking_active && park_state != ParkState::CAPTURE) {
                // Skip mask overlay during pure heading-hold
            } else if (roadOK) {
                cv::Mat green(frame.size(), frame.type(), cv::Scalar(0, 255, 0));
                cv::Mat blended;
                cv::addWeighted(frame, 0.65, green, 0.35, 0, blended);
                blended.copyTo(overlay, mask);

                if (ui_y1 > 0 && ui_y2 > 0) {
                    cv::line(overlay, cv::Point(0, ui_y1), cv::Point(frame.cols - 1, ui_y1),
                             cv::Scalar(0, 0, 255), 1);
                    cv::line(overlay, cv::Point(0, ui_y2), cv::Point(frame.cols - 1, ui_y2),
                             cv::Scalar(0, 0, 255), 1);
                }
                for (size_t i = 0; i + 1 < ui_left_pts.size(); ++i)
                    cv::line(overlay, ui_left_pts[i], ui_left_pts[i + 1], cv::Scalar(255, 0, 0), 2);
                for (size_t i = 0; i + 1 < ui_right_pts.size(); ++i)
                    cv::line(overlay, ui_right_pts[i], ui_right_pts[i + 1], cv::Scalar(255, 0, 0), 2);
                for (const auto& p : ui_left_pts) cv::circle(overlay, p, 3, cv::Scalar(0, 255, 0), -1);
                for (const auto& p : ui_right_pts) cv::circle(overlay, p, 3, cv::Scalar(0, 255, 0), -1);

                if (out.proc_width > 0 && out.proc_height > 0) {
                    float sx = float(frame.cols) / out.proc_width;
                    float sy = float(frame.rows) / out.proc_height;
                    for (size_t i = 0; i < out.centerline_pts.size(); i += 8) {
                        int cx = int(out.centerline_pts[i].first * sx);
                        int cy = int(out.centerline_pts[i].second * sy);
                        cv::circle(overlay, cv::Point(cx, cy), 2, cv::Scalar(0, 255, 255), -1);
                    }
                }
            }

            for (const auto& s : cachedSigns) {
                cv::rectangle(overlay, s.box, cv::Scalar(0, 255, 255), 2);
                char label[64];
                std::snprintf(label, sizeof(label), "%s %.2f", s.name.c_str(), s.confidence);
                cv::putText(overlay, label, {s.box.x, std::max(12, s.box.y - 5)},
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
            }

            char t1[200], t2[200], t3[200], t4[200];
            std::snprintf(t1, sizeof(t1),
                "FPS %.1f | spd %.1f/%.1f | pid %.1f | srv %d",
                currentFps, applied_speed, speed, out.angle, final_angle);
            std::snprintf(t2, sizeof(t2),
                "sign %s (%d) | rb %s (pass %d/2) | park %s",
                SignName(voter.GetCommitted()), voter.GetCommitted(),
                RbStateName(rb_state), rb_pass_count, ParkStateName(park_state));
            std::snprintf(t3, sizeof(t3),
                "hdg %.2f° ref %.2f° Δ%.2f°",
                heading_now, park_ref_heading,
                normalizeAngleP(heading_now - park_ref_heading));
            std::snprintf(t4, sizeof(t4),
                "tof L=%.0f R=%.0f (blk<%.0f) | near %.1f",
                tof_left_mm, tof_right_mm, PARK_TOF_BLOCKED_MM, out.near_e);

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
// ============================================================
// main_run.cpp — UCR2026 Full Integrated (LaneFollower + SignDetector + NavigationBrain)
// ============================================================
//4
#include "hardware/Camera.h"
#include "hardware/Motor.h"
#include "hardware/Servo.h"
#include "common/Logger.h"

#include "control/LaneFollowerV20.h"
#include "SignDetector.h"
#include "NavigationBrain.h"
#include "SignVoter.h"

// Thư viện cảm biến TOF và BNO055 (IMU)
#include "sensor/TOF.h"
#include "sensor/BNO055.h"

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
static const std::string SIGN_ENGINE_PATH = "/home/car4/Downloads/sign/signv4.engine";

static constexpr float MAX_SPEED_CMS         = 30.0f;
static constexpr int   MAX_STEER_DEG        = 50;
static constexpr bool  INVERT_STEER         = true;
static constexpr bool  SHOW_UI              = true;
static constexpr bool  DA_CHANNEL           = 1; 
static constexpr float MIN_SPEED_CMS        = 22.0f;
static constexpr int   SIGN_INTERVAL_FRAMES = 1;
static constexpr int   SIGN_BLOCK_FRAMES    = 40; 
static constexpr double LANE_BIAS_RIGHT_PX = 7.0; 

// ============================================================
// SIGN CLASSES & VOTER CONSTANTS
// ============================================================
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

static const char* SignClassName(int c) {
    switch (c) {
        case sign_class::CONSTRUCTION_AHEAD: return "construction_ahead";
        case sign_class::NO_LEFT:            return "no_left";
        case sign_class::NO_RIGHT:           return "no_right";
        case sign_class::PARKING:            return "parking";
        case sign_class::ROUNDABOUT:         return "roundabout";
        case sign_class::TRAFFIC_LIGHT:      return "traffic_light";
        case sign_class::TURN_LEFT:          return "turn_left";
        case sign_class::TURN_RIGHT:         return "turn_right";
        case sign_class::WALKING_PERSON:     return "walking_person";
        default:                             return "NONE";
    }
}

static constexpr int    SIGN_VOTER_ACTIVATE_FRAMES = 2;
static constexpr int    SIGN_VOTER_TIMEOUT_DS      = 60; // 6.0s timeout (đủ thời gian tiến vào cua)
static constexpr double SIGN_BLOCK_SECONDS         = 4.0;

// ============================================================
// ROUNDABOUT STATE MACHINE (FROM MAIN.CPP)
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

// Thông số Roundabout độc lập (giữ nguyên PID chính)
static constexpr float  RB_BIAS_PX_RIGHT         = 28.0f;
static constexpr float  RB_BIAS_PX_LEFT          = 20.0f;
static constexpr float  RB_BIAS_EMA              = 0.30f;
static constexpr double RB_MISSING_SIGN_S        = 2.0;
static constexpr double RB_CIRCLE_TIMEOUT_S      = 12.0;
static constexpr double RB_EXIT_SEARCH_TIMEOUT_S = 15.0;
static constexpr double RB_EXIT_FORCE_EXEC_S     = 1.5;
static constexpr double RB_COOLDOWN_S            = 35.0; // Cooldown 35s chống nhận diện lại biển từ xa khi lên/xuống dốc cầu
static constexpr float  RB_SPEED_MIN_CMS         = 20.0f;
static constexpr float  RB_SPEED_MAX_CMS         = 30.0f;
static constexpr float  RB_EXIT_TURN_FLOOR_DEG   = 16.0f;
static constexpr float  RB_EXIT_TURN_BIAS_PX     = 18.0f;

// ============================================================
// PARKING CONFIGURATION & STATE MACHINE
// ============================================================
enum class ParkingState {
    IDLE = 0,
    APPROACH,            // Đi giữa 2 scanline cố định tiếp cận bãi đỗ
    SLOT1_WAIT_BLOCKED,  // Chờ cả 2 bên (TOF1 & TOF3) bị block ở ô 1
    SLOT2_SEARCH_FREE,   // Chờ tìm thấy ô trống (1 bên free, 1 bên block)
    SLOT3_WAIT_BLOCKED,  // Chạy tiếp đến khi cả 2 bên bị block ở ô 3 -> DỪNG
    WAIT_FULL_STOP,      // Dừng hẳn xe, lưu heading_0
    REVERSE_1,           // Bẻ lái tối đa về phía ô trống, lùi đến khi lệch 45 độ
    REVERSE_2,           // Bẻ lái hướng ngược lại, lùi đến khi heading về ban đầu +- 3 độ
    DONE                 // Đỗ xe hoàn thành, dừng xe
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

static inline float normalizeAngle(float angle) {
    while (angle >  180.0f) angle -= 360.0f;
    while (angle < -180.0f) angle += 360.0f;
    return angle;
}

// Cấu trúc hành lang 2 tia chéo thẳng tuyệt đối theo góc camera cho Parking (chống cong)
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

static constexpr float PARK_TOF_BLOCKED_MM          = 350.0f; // Ngưỡng mm (< 350mm coi là bị block)
static constexpr float PARK_APPROACH_SPEED          = 13.0f;  // Tốc độ tiến trong bãi đỗ (cm/s)
static constexpr float PARK_REVERSE_SPEED           = -10.0f;  // Tốc độ lùi chậm (cm/s)
static constexpr float PARK_MAX_STEER_DEG           = 45.0f;  // Góc bẻ lái khi lùi (độ bánh xe)
static constexpr float PARK_TARGET_DELTA_DEG        = 50.0f;  // Lệch 45 độ ở REVERSE_1
static constexpr float PARK_ALIGN_TOL_DEG           = 0.0f;   // Dung sai +-3 độ ở REVERSE_2
static constexpr double PARK_STOP_DWELL_S           = 0.5;    // Thời gian dừng hẳn trước khi lùi (s)
static constexpr int   TOF_IDX_LEFT                 = 1;      // TOF 1: cảm biến bên trái xe
static constexpr int   TOF_IDX_RIGHT                = 3;      // TOF 3: cảm biến bên phải xe

static constexpr int   PARK_CORRIDOR_EXTRA_MARGIN_PX = 18;    // Mở rộng lề mỗi bên thêm 18px để mask thông thoáng
static constexpr float PARK_CORRIDOR_FAR_RATIO       = 0.68f; // Tỉ lệ thu hẹp phối cảnh camera từ đáy lên far
static constexpr float PARK_STEER_TRIM_DEG           = +2.5f; // Bù sai số lệch trái (+2.5° lái sang phải khi chạy chậm)
static constexpr float PARK_HEADING_KP               = 0.90f; // Hệ số P giữ hướng thẳng tuyệt đối theo IMU
static constexpr float PARK_HEADING_MAX_CORR_DEG     = 8.0f;  // Giới hạn góc can thiệp của heading-hold

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
                 "Turn_Bias,Steer_Floor_Deg,P0,P1,P2,"
                 "Is_Roundabout,RB_Phase,RB_Exit,"
                 "Obs_State,Obs_Side,Obs_Y_Cut,Obs_Dist_Px,Obs_Near_Cut,Obs_Pass_Count,"
                 "TJ_Count,TJ_Detected,"
                 "TL_Color,TL_Stop,TL_Top_B,TL_Mid_B,TL_Bot_B,TL_Diff,"
                 "Park_State,Park_Side,TOF_L,TOF_R,Heading,Heading_0,Delta_H,"
                 "Left_Pts,Right_Pts,IsValid\n";
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

        // --- ROUNDABOUT TELEMETRY ---
        bool isRoundabout,
        const std::string& rbPhase,
        int rbExit,

        // --- OBSTACLE AVOIDANCE TELEMETRY ---
        const std::string& obsState,
        int obsSide,
        int obsYCut,
        int obsDistPx,
        bool obsNearCut,
        int obsPassCount,

        // --- T-JUNCTION TELEMETRY ---
        int tjCount,
        bool tjDetected,

        // --- TRAFFIC LIGHT TELEMETRY ---
        const std::string& tlColor,
        bool tlStop,
        int tlTopB,
        int tlMidB,
        int tlBotB,
        int tlDiff,

        // --- PARKING TELEMETRY ---
        const std::string& parkState,
        int parkSide,
        float tofLeft,
        float tofRight,
        float heading,
        float heading0,
        float deltaH,

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
            // --- THÔNG SỐ ROUNDABOUT ---
            << (isRoundabout ? 1 : 0) << ","
            << rbPhase << ","
            << rbExit << ","
            // --- THÔNG SỐ NÉ VẬT CẢN (OBSTACLE) ---
            << obsState << ","
            << obsSide << ","
            << obsYCut << ","
            << obsDistPx << ","
            << (obsNearCut ? 1 : 0) << ","
            << obsPassCount << ","
            // --- THÔNG SỐ T-JUNCTION KHÔNG BIỂN ---
            << tjCount << ","
            << (tjDetected ? 1 : 0) << ","
            // --- THÔNG SỐ ĐÈN GIAO THÔNG (TRAFFIC LIGHT) ---
            << tlColor << ","
            << (tlStop ? 1 : 0) << ","
            << tlTopB << ","
            << tlMidB << ","
            << tlBotB << ","
            << tlDiff << ","
            // --- THÔNG SỐ PARKING ---
            << parkState << ","
            << parkSide << ","
            << std::fixed << std::setprecision(1) << tofLeft << ","
            << tofRight << ","
            << heading << ","
            << heading0 << ","
            << deltaH << ","
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

    // --- Khởi tạo cảm biến TOF & BNO055 (IMU) ---
    sensor::TOF tof;
    bool tof_ok = tof.Init("/dev/i2c-1");
    if (!tof_ok) {
        std::cerr << "[Main] TOF init failed (/dev/i2c-1)\n";
    }

    BNO055 bno;
    bool imu_ok = bno.initialize(std::chrono::milliseconds(1000));
    if (!imu_ok) {
        std::cerr << "[Main] BNO055 init failed\n";
    }

    sensor::TOFReadings tof_readings;
    BNO055::Data bno_data;

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
    v20::ObstacleAvoidanceController obstacle_controller;
    std::string handled_sign = "NONE";
    v20::LaneFollower::Output previous_out;

    // --- Biến trạng thái Parking (Hành lang & Ô đỗ) ---
    ParkingState park_state = ParkingState::IDLE;
    ParkingCorridor park_corridor;
    int park_free_side = -1; // -1: LEFT trống, +1: RIGHT trống
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

    // --- Sign Voter & Timers ---
    ucr::control::SignVoter voter(SIGN_VOTER_ACTIVATE_FRAMES, SIGN_VOTER_TIMEOUT_DS);
    double sign_block_until_time = 0.0;
    double elapsed_total         = 0.0;

    // --- Roundabout State Machine (main.cpp 2-pass) ---
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
    int      rb_pass_count         = 0;

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
        elapsed_total += dt_loop;

        // --- ĐỌC CẢM BIẾN TOF & BNO055 (IMU) ---
        tof_left = -1.0f;
        tof_right = -1.0f;
        if (tof_ok && tof.Read(tof_readings, std::chrono::milliseconds(10))) {
            if (tof_readings[TOF_IDX_LEFT].valid && tof_readings[TOF_IDX_LEFT].distanceMm > 0) {
                tof_left = static_cast<float>(tof_readings[TOF_IDX_LEFT].distanceMm);
            }
            if (tof_readings[TOF_IDX_RIGHT].valid && tof_readings[TOF_IDX_RIGHT].distanceMm > 0) {
                tof_right = static_cast<float>(tof_readings[TOF_IDX_RIGHT].distanceMm);
            }
        }

        current_heading = 0.0f;
        heading_ok = false;
        if (imu_ok && bno.read(bno_data)) {
            current_heading = static_cast<float>(bno_data.eulerDeg.x);
            heading_ok = true;
        }

        if (totalFrames % SIGN_INTERVAL_FRAMES == 0) {
            // Hạ ngưỡng detect xuống 0.28f để model bắt biển từ rất xa
            cachedSigns = signDetector.Detect(frame, 0.28f);
            
            std::vector<ucr::control::SignObservation> signObs;
            signObs.reserve(cachedSigns.size());
            for (const auto& d : cachedSigns) {
                if (d.confidence < 0.30f) continue;
                // BỎ QUA traffic_light và walking_person: hai loại này là chướng ngại vật tức thời,
                // được xử lý độc lập trong navBrain.UpdateHazards(), TUYỆT ĐỐI KHÔNG đưa vào SignVoter
                // để tránh chiếm quyền hoặc cướp mất biển điều hướng (turn_left, turn_right, roundabout)!
                if (d.class_id == sign_class::TRAFFIC_LIGHT || d.name == "traffic_light" ||
                    d.class_id == sign_class::WALKING_PERSON || d.name == "walking_person") {
                    continue;
                }

                // Nới lỏng bộ lọc: chỉ lọc nhiễu hạt bụi cực nhỏ (< 100px)
                if (d.box.width < 10 || d.box.height < 10 || d.box.area() < 100) continue;

                // Lọc bóng mờ ở tít đỉnh trời (y+h < 45px và diện tích nhỏ)
                if (d.box.area() < 180 && (d.box.y + d.box.height) < 45) continue;

                // Riêng biển roundabout: chống nhận diện từ trên dốc cầu nhìn xa sang đảo
                // Yêu cầu diện tích >= 600px và đáy biển ở nửa dưới ảnh (y+h >= 105px)
                if (d.class_id == sign_class::ROUNDABOUT || d.name == "roundabout") {
                    if (d.box.area() < 600 || (d.box.y + d.box.height) < 105) continue;
                }

                int cls = d.class_id;
                if (cls < 0) {
                    for (size_t i = 0; i < ObjectDetector::CLASS_NAMES.size(); ++i) {
                        if (ObjectDetector::CLASS_NAMES[i] == d.name) { cls = static_cast<int>(i); break; }
                    }
                }
                if (cls >= 0) {
                    signObs.emplace_back(cls, d.confidence, d.box.area());
                }
            }

            uint32_t ds = static_cast<uint32_t>(elapsed_total * 10.0);
            voter.Update(signObs, ds);

            const bool turn_armed = (turn_controller.state != v20::LaneTurnController::State::IDLE) &&
                                    !turn_controller.is_unmarked_tjunction;
            navBrain.UpdateHazards(frame, cachedSigns, turn_armed);
        }

        // --- CHỐT BIỂN BÁO BẰNG SIGN VOTER (CHỐT CHÍNH XÁC, KHÔNG DROP KHI CONF BIẾN ĐỘNG) ---
        const int committed_class = voter.GetCommitted();
        std::string voter_sign = "NONE";
        NavAction   voter_action = NavAction::FOLLOW_LANE;
        if (elapsed_total >= sign_block_until_time && committed_class >= 0) {
            voter_sign = SignClassName(committed_class);
            navBrain.SetLatchedSign(voter_sign);
            if (committed_class == sign_class::TURN_LEFT || 
                committed_class == sign_class::NO_RIGHT || 
                committed_class == sign_class::CONSTRUCTION_AHEAD) {
                voter_action = NavAction::FORCE_LEFT;
            } else if (committed_class == sign_class::TURN_RIGHT || 
                       committed_class == sign_class::NO_LEFT) {
                voter_action = NavAction::FORCE_RIGHT;
            } else if (committed_class == sign_class::PARKING) {
                voter_action = NavAction::FOLLOW_LANE;
            } else if (committed_class == sign_class::ROUNDABOUT) {
                voter_action = NavAction::FOLLOW_LANE;
            }

            static int prev_logged_committed = -1;
            if (committed_class != prev_logged_committed) {
                std::printf("[SIGN_VOTER] >>> COMMITTED SIGN: %s (MaxConf=%.2f, Hits=%d, Score=%.1f) t=%.2f <<<\n",
                            voter_sign.c_str(), voter.GetCommittedMaxConf(),
                            voter.GetHitCount(committed_class), voter.GetScore(committed_class), elapsed_total);
                prev_logged_committed = committed_class;
            }
        }

        bool roadOK = roadModel.Predict(frame, mask);
        v20::LaneFollower::Output out;
        float speed = MIN_SPEED_CMS;
        int final_angle = 0;
        NavDecision dec;
        v20::LaneTurnController::TurnResult turn_res{v20::LaneTurnController::State::IDLE, 0.0, "NONE"};
        v20::ObstacleAvoidanceController::ObstacleResult obs_res;
        double applied_floor = 0.0;
        bool parking_active = (park_state != ParkingState::IDLE);
        // --- UI DATA EXPORT VARIABLES ---
        std::vector<cv::Point> ui_left_pts, ui_right_pts;
        int ui_y1 = 0, ui_y2 = 0;

        if (roadOK) {
            dec = navBrain.Process(mask, follower.speed());
            if (voter_sign != "NONE") {
                dec.active_sign = voter_sign;
                dec.action      = voter_action;
            }

            int h = mask.rows, w = mask.cols;
            int y1 = static_cast<int>(h * 0.45);
            int y2 = static_cast<int>(h * 0.65);
            int y_bottom = h - 1;

            auto get_raw_boundaries = [&](const cv::Mat& m, int y_row, int& lx, int& rx) {
                lx = -1; rx = -1;
                if (m.empty() || y_row < 0 || y_row >= m.rows) return;
                const uchar* row = m.ptr<uchar>(y_row);
                int start = -1; int best_len = -1;
                for (int x = 0; x <= m.cols; x++) {
                    bool on = (x < m.cols) && (row[x] > 0);
                    if (on && start < 0) start = x;
                    if (!on && start >= 0) {
                        int end_x = x - 1;
                        int len = x - start;
                        if (len > 15 && len > best_len) {
                            best_len = len; lx = start; rx = end_x;
                        }
                        start = -1;
                    }
                }
            };

            // -------------------------------------------------------------
            // QUẢN LÝ TRẠNG THÁI PARKING & KẺ 2 SCANLINE CHÉO 3 ĐIỂM CỐ ĐỊNH
            // -------------------------------------------------------------
            if (dec.active_sign == "parking") {
                parking_active = true;
            }
            if (dec.active_sign == "parking" && park_state == ParkingState::IDLE) {
                park_state = ParkingState::APPROACH;
                slot1_confirm_cnt = 0;
                slot2_confirm_cnt = 0;
                slot3_confirm_cnt = 0;
                if (heading_ok) park_straight_heading = current_heading;
                std::cout << "[PARK] >>> PARKING SIGN CONFIRMED -> ACTIVATING PARKING MODE (RefH=" << park_straight_heading << " deg) <<<\n";
            }

            // Chốt hành lang 2 tia chéo thẳng tuyệt đối (dù đường có bị cong/mất viền thì scanline vẫn thẳng và mở rộng)
            if (dec.active_sign == "parking" && !park_corridor.latched) {
                int raw_lx1 = -1, raw_rx1 = -1;
                int raw_lx_bot = -1, raw_rx_bot = -1;
                get_raw_boundaries(mask, y1, raw_lx1, raw_rx1);
                get_raw_boundaries(mask, y_bottom, raw_lx_bot, raw_rx_bot);

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

                std::cout << "[PARK] >>> LATCHED STRAIGHT PERSPECTIVE CORRIDOR: Left [Bot=("
                          << park_corridor.x_left_bot << "," << y_bottom << ") -> Far=("
                          << park_corridor.x_left_far << "," << y1 << ")], Right [Bot=("
                          << park_corridor.x_right_bot << "," << y_bottom << ") -> Far=("
                          << park_corridor.x_right_far << "," << y1 << ")], Center="
                          << center_bot << " HalfW(Bot=" << half_w_bot << ",Far=" << half_w_far << ") <<<\n";
            }

            // Bỏ qua tất cả các pixel mask nằm ngoài hành lang chéo 3 điểm khi vào bãi đỗ
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
                        int end_x = x - 1;
                        int len = x - start;
                        if (len > 15) {
                            const int seg_center = (start + end_x) / 2;
                            bool is_pref = (pref_side < 0 && seg_center <= half_w) ||
                                           (pref_side > 0 && seg_center >= half_w);
                            bool take = false;
                            if (pref_side != 0) {
                                if (is_pref && !has_pref) take = true;
                                else if (is_pref && has_pref) {
                                    if (pref_side < 0 && start < lx) take = true;
                                    if (pref_side > 0 && end_x > rx) take = true;
                                } else if (!has_pref && len > best_len) {
                                    take = true;
                                }
                            } else if (len > best_len) {
                                take = true;
                            }
                            if (take) {
                                best_len = len; lx = start; rx = end_x;
                                if (is_pref) has_pref = true;
                            }
                        }
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
                        pts.push_back(cv::Point(is_left ? 0 : w - 1, std::max(start_y, y - 2)));
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

            // Không nhận lệnh rẽ/roundabout khi đang trong chế độ parking hoặc đang ôm cua 90°
            // Cho phép ghi đè nếu đang ở WAIT_CURVE hoặc WAIT_STRAIGHT (chưa thực sự đánh lái EXECUTE) mà chốt được biển mới
            // -----------------------------------------------------
            // ROUNDABOUT LOGIC FROM MAIN.CPP (2-PASS STATE MACHINE)
            // -----------------------------------------------------
            bool rb_visible_now = false;
            for (const auto& d : cachedSigns) {
                if ((d.name == "roundabout" || d.class_id == sign_class::ROUNDABOUT) &&
                    d.confidence >= 0.55f && d.box.area() >= 700) {
                    rb_visible_now = true;
                    break;
                }
            }
            if (rb_visible_now) last_rb_visible_time = elapsed_total;

            const bool rb_committed = (voter.GetCommitted() == sign_class::ROUNDABOUT);

            if (rb_state == RbState::APPROACH || rb_state == RbState::EXIT_PREP) {
                if (rb_visible_now) rb_missing_sign_ticks = 0;
                else                ++rb_missing_sign_ticks;
            }

            switch (rb_state) {
                case RbState::NONE:
                    if (rb_committed &&
                        elapsed_total >= rb_cooldown_until &&
                        elapsed_total >= sign_block_until_time &&
                        !parking_active &&
                        turn_controller.state == v20::LaneTurnController::State::IDLE &&
                        !corner_assist.active() &&
                        std::abs(previous_out.angle) <= 14.0) { // Xe phải đang chạy thẳng hướng tới đảo, không phải đang ôm cua gắt chân cầu!

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

                        std::printf("[RB] Pass %d/2 EXIT_SEARCH -> start_turn(%s) floor=%.0f° bias=%.0fpx\n",
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
                        // Khôi phục ngay lập tức các thông số bẻ lái chuẩn của LaneTurnController
                        // để các biển báo rẽ sau (turn_right, no_right, no_left) đánh lái đủ lực cực đại
                        turn_controller.TURN_STEER_DEG = v20::LaneTurnController::DEFAULT_TURN_STEER_DEG;
                        turn_controller.TURN_BIAS_PX   = v20::LaneTurnController::DEFAULT_TURN_BIAS_PX;
                        turn_controller.TURN_INIT_DEG  = v20::LaneTurnController::DEFAULT_TURN_INIT_DEG;
                        turn_controller.TURN_RAMP_DEG_PER_S = 120.0;

                        rb_state = RbState::NONE;
                        rb_bias_smoothed = 0.0f;

                        if (rb_pass_count == 1) {
                            turn_controller.waiting_for_tjunction = true;
                            turn_controller.tjunction_cooldown = turn_controller.TJUNCTION_COOLDOWN_S;
                            std::printf("[RB] Pass 1 done — armed T-junction for loop-back\n");
                        } else {
                            turn_controller.waiting_for_tjunction = false;
                            std::printf("[RB] Pass 2 done — roundabout sequence COMPLETE\n");
                            rb_pass_count = 0;
                        }

                        rb_cooldown_until = elapsed_total + RB_COOLDOWN_S;
                        voter.Reset();
                    }
                    break;
                }
            }

            if (rb_state != rb_state_prev) {
                std::printf("[RB] %-10s -> %-10s t=%.2f sign=%s bias=%.1f pass=%d\n",
                            RbStateName(rb_state_prev), RbStateName(rb_state),
                            elapsed_total, SignClassName(voter.GetCommitted()),
                            rb_bias_smoothed, rb_pass_count);
                rb_state_prev = rb_state;
            }

            // Lệnh rẽ từ các biển báo khác (turn_left, turn_right, no_left, no_right, construction_ahead)
            const bool can_accept_sign = !parking_active && (rb_state == RbState::NONE) && !corner_assist.active() &&
                (turn_controller.state == v20::LaneTurnController::State::IDLE ||
                 ((turn_controller.state == v20::LaneTurnController::State::WAIT_CURVE ||
                   turn_controller.state == v20::LaneTurnController::State::WAIT_STRAIGHT) &&
                  dec.active_sign != handled_sign));
            if (can_accept_sign) {
                // Đảm bảo luôn nạp lực bẻ lái cực đại (floor 35°, bias 40px)
                turn_controller.TURN_STEER_DEG = v20::LaneTurnController::DEFAULT_TURN_STEER_DEG;
                turn_controller.TURN_BIAS_PX   = v20::LaneTurnController::DEFAULT_TURN_BIAS_PX;
                turn_controller.TURN_INIT_DEG  = v20::LaneTurnController::DEFAULT_TURN_INIT_DEG;
                turn_controller.TURN_RAMP_DEG_PER_S = 120.0;

                if (dec.action == NavAction::FORCE_LEFT) {
                    turn_controller.start_turn("LEFT", false);
                    handled_sign = dec.active_sign;
                }
                else if (dec.action == NavAction::FORCE_RIGHT) {
                    turn_controller.start_turn("RIGHT", false);
                    handled_sign = dec.active_sign;
                }
            }

            // Đang dừng vì người đi bộ / đèn đỏ-vàng: ĐÓNG BĂNG turn controller & corner assist
            // (xe đứng yên nên mask không đổi -> không được đếm frame thẳng ảo hay chạy hết timeout rẽ).
            // Biển đã chốt vẫn được giữ, khi được đi sẽ tiếp tục chờ frame thẳng rồi mới quành như cũ.
            const bool hazard_stop = navBrain.HazardStop();
            if (!parking_active && !hazard_stop) {
                turn_res = turn_controller.update(
                    left_pts, right_pts, dt_loop, previous_out.valid,
                    lx1, rx1, lx2, rx2, w,
                    (dec.active_sign != "NONE" || handled_sign != "NONE"),
                    previous_out.near_e, previous_out.mid_e, previous_out.far_e,
                    lx_bot, rx_bot
                );
            } else {
                // Khi đang dừng chờ đèn đỏ / người đi bộ: giữ sẵn trạng thái chờ rẽ của biển báo,
                // đồng thời giữ straight_frames = 0 để khi đèn xanh xe tiến thẳng đủ 3 frame mới bắt đầu quành
                if (turn_controller.state == v20::LaneTurnController::State::WAIT_CURVE ||
                    turn_controller.state == v20::LaneTurnController::State::WAIT_STRAIGHT) {
                    turn_controller.straight_frames = 0;
                }
                turn_res.state           = turn_controller.state;
                turn_res.direction       = turn_controller.direction;
                turn_res.straight_frames = turn_controller.straight_frames;
                turn_res.t_junction_count = turn_controller.t_junction_count;
            }
            // -------------------------------------------------------------
            // CẬP NHẬT MÁY TRẠNG THÁI NÉ VẬT CẢN (ObstacleAvoidanceController)
            // Tự động nhận diện vết cắt mask (step/shelf), đo khoảng cách px,
            // né mạnh sang làn đối diện, và KHÓA CỨNG hướng vật cản khi đang chạy ngang hông!
            // -------------------------------------------------------------
            const bool turn_active_or_armed = (turn_controller.state != v20::LaneTurnController::State::IDLE) ||
                                              (rb_state != RbState::NONE) ||
                                              (dec.active_sign != "NONE") ||
                                              (handled_sign != "NONE");
            if (!parking_active && !hazard_stop && !turn_active_or_armed) {
                obs_res = obstacle_controller.update(
                    active_mask, left_pts, right_pts,
                    lx1, rx1, lx2, rx2, lx_bot, rx_bot,
                    w, dt_loop,
                    previous_out.near_e, previous_out.mid_e, previous_out.far_e
                );
            } else {
                obstacle_controller.reset();
            }

            // ---- Roundabout bias table (EMA-smoothed) từ main.cpp ----
            float rb_bias_px = 0.0f;
            if (rb_state != RbState::NONE) {
                corner_assist.reset();
                switch (rb_state) {
                    case RbState::APPROACH:    rb_bias_px = +RB_BIAS_PX_RIGHT; break;
                    case RbState::IN_CIRCLE:   rb_bias_px = -RB_BIAS_PX_LEFT;  break;
                    case RbState::EXIT_PREP:   rb_bias_px = +RB_BIAS_PX_RIGHT; break;
                    case RbState::EXIT_SEARCH: rb_bias_px = 0.0f;              break; // turn_ctrl tự lo
                    default:                   rb_bias_px = 0.0f;              break;
                }
                rb_bias_smoothed = (1.0f - RB_BIAS_EMA) * rb_bias_smoothed + RB_BIAS_EMA * rb_bias_px;
            } else {
                rb_bias_smoothed = 0.0f;
            }

            // Tổng hợp góc sàn & lực kéo: Ưu tiên Roundabout -> TurnController -> ObstacleAvoidanceController -> CornerAssist
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
                    (turn_controller.is_unmarked_tjunction &&
                     turn_controller.state == v20::LaneTurnController::State::WAIT_CURVE &&
                     !turn_res.junction_opened &&
                     !(lx2 >= 0 && lx2 <= 15 && rx2 >= w - 15)));
                if (allow_corner_assist) {
                    const double cf = corner_assist.update(
                        lx1, rx1, lx2, rx2, lx_bot, rx_bot, w, dt_loop,
                        (dec.active_sign != "NONE"),
                        previous_out.near_e, previous_out.mid_e, previous_out.far_e
                    );
                    if (std::abs(cf) > std::abs(effective_floor)) effective_floor = cf;
                } else if (!hazard_stop) {
                    corner_assist.reset();
                }
            }
            applied_floor = effective_floor;

            // -------------------------------------------------------------
            // CẬP NHẬT MÁY TRẠNG THÁI PARKING (TOF 1 Trái, TOF 3 Phải, IMU Heading)
            // -------------------------------------------------------------
            const bool is_left_blocked  = (tof_left > 0.0f && tof_left < PARK_TOF_BLOCKED_MM);
            const bool is_right_blocked = (tof_right > 0.0f && tof_right < PARK_TOF_BLOCKED_MM);
            const bool is_left_free     = (tof_left < 0.0f || tof_left >= PARK_TOF_BLOCKED_MM);
            const bool is_right_free    = (tof_right < 0.0f || tof_right >= PARK_TOF_BLOCKED_MM);

            if (parking_active) {
                switch (park_state) {
                    case ParkingState::APPROACH:
                    case ParkingState::SLOT1_WAIT_BLOCKED: {
                        // Xe tiếp tục đi đến khi 2 TOF hai bên nhận thấy đang bị block ở ô 1
                        if (is_left_blocked && is_right_blocked) {
                            slot1_confirm_cnt++;
                            if (slot1_confirm_cnt >= 2) {
                                slot1_tof_left  = tof_left;
                                slot1_tof_right = tof_right;
                                park_state = ParkingState::SLOT2_SEARCH_FREE;
                                std::cout << "[PARK] >> SLOT 1: BOTH BLOCKED (L=" << tof_left
                                          << "mm, R=" << tof_right << "mm) -> Advancing to search free slot...\n";
                            }
                        } else {
                            slot1_confirm_cnt = 0;
                        }
                        break;
                    }

                    case ParkingState::SLOT2_SEARCH_FREE: {
                        // Cho xe chạy tiếp đến khi TOF bên trái thấy đang free và bên phải đang bị block (hoặc ngược lại)
                        if (is_left_free && is_right_blocked) {
                            slot2_confirm_cnt++;
                            if (slot2_confirm_cnt >= 2) {
                                park_free_side = -1; // Bên TRÁI trống
                                slot2_tof_left  = tof_left;
                                slot2_tof_right = tof_right;
                                park_state = ParkingState::SLOT3_WAIT_BLOCKED;
                                std::cout << "[PARK] >> SLOT 2: LEFT FREE (L=" << tof_left
                                          << "mm), RIGHT BLOCKED (R=" << tof_right
                                          << "mm) -> SAVED FREE SIDE: LEFT. Continuing to Slot 3...\n";
                            }
                        } else if (is_right_free && is_left_blocked) {
                            slot2_confirm_cnt++;
                            if (slot2_confirm_cnt >= 2) {
                                park_free_side = 1; // Bên PHẢI trống
                                slot2_tof_left  = tof_left;
                                slot2_tof_right = tof_right;
                                park_state = ParkingState::SLOT3_WAIT_BLOCKED;
                                std::cout << "[PARK] >> SLOT 2: RIGHT FREE (R=" << tof_right
                                          << "mm), LEFT BLOCKED (L=" << tof_left
                                          << "mm) -> SAVED FREE SIDE: RIGHT. Continuing to Slot 3...\n";
                            }
                        } else {
                            slot2_confirm_cnt = 0;
                        }
                        break;
                    }

                    case ParkingState::SLOT3_WAIT_BLOCKED: {
                        // Tiếp tục cho xe chạy đến khi thấy hai bên bị block tiếp thì dừng (mỗi bên có 3 ô)
                        if (is_left_blocked && is_right_blocked) {
                            slot3_confirm_cnt++;
                            if (slot3_confirm_cnt >= 2) {
                                slot3_tof_left  = tof_left;
                                slot3_tof_right = tof_right;
                                if (heading_ok) park_heading_0 = current_heading;
                                park_dwell_time = 0.0;
                                park_state = ParkingState::WAIT_FULL_STOP;
                                std::cout << "[PARK] >> SLOT 3: BOTH BLOCKED (L=" << tof_left
                                          << "mm, R=" << tof_right
                                          << "mm) -> STOP! Saved Base Heading=" << park_heading_0 << " deg\n";
                            }
                        } else {
                            slot3_confirm_cnt = 0;
                        }
                        break;
                    }

                    case ParkingState::WAIT_FULL_STOP: {
                        park_dwell_time += dt_loop;
                        if (heading_ok) park_heading_0 = current_heading;
                        if (park_dwell_time >= PARK_STOP_DWELL_S) {
                            park_state = ParkingState::REVERSE_1;
                            park_reverse_elapsed = 0.0;
                            std::cout << "[PARK] >> FULL STOP CONFIRMED (Heading_0=" << park_heading_0
                                      << " deg) -> START REVERSE_1 (Steer max to "
                                      << (park_free_side < 0 ? "LEFT" : "RIGHT") << ")\n";
                        }
                        break;
                    }

                    case ParkingState::REVERSE_1: {
                        park_reverse_elapsed += dt_loop;
                        // Bẻ lái tối đa về phía ô trống, lùi chậm cho đến khi heading lệch 45 độ so với heading_0
                        float delta_h = std::abs(normalizeAngle(current_heading - park_heading_0));
                        if (delta_h >= PARK_TARGET_DELTA_DEG || park_reverse_elapsed >= 7.0) {
                            park_state = ParkingState::REVERSE_2;
                            park_reverse_elapsed = 0.0;
                            std::cout << "[PARK] >> REVERSE_1 COMPLETE (Delta=" << delta_h
                                      << " deg >= 45 deg) -> START REVERSE_2 (Counter-steer opposite)\n";
                        }
                        break;
                    }

                    case ParkingState::REVERSE_2: {
                        park_reverse_elapsed += dt_loop;
                        // Bẻ lái ngược lại, lùi chậm cho đến khi heading trả về giá trị ban đầu +- 3 độ
                        float delta_h = std::abs(normalizeAngle(current_heading - park_heading_0));
                        if (delta_h <= PARK_ALIGN_TOL_DEG || park_reverse_elapsed >= 7.0) {
                            park_state = ParkingState::DONE;
                            std::cout << "[PARK] >> REVERSE_2 COMPLETE: ALIGNED (Delta=" << delta_h
                                      << " deg <= 3 deg) -> PARKING COMPLETED! <<<\n";
                        }
                        break;
                    }

                    case ParkingState::DONE: {
                        break;
                    }

                    default:
                        break;
                }
            }

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
                voter.Reset();
                navBrain.ClearLatchedSign();
                if (handled_sign != "NONE") navBrain.BlockSign(handled_sign, SIGN_BLOCK_FRAMES);
                sign_block_until_time = elapsed_total + SIGN_BLOCK_SECONDS;
                handled_sign = "NONE";
            }

            if (parking_active) {
                if (park_state == ParkingState::WAIT_FULL_STOP || park_state == ParkingState::DONE) {
                    speed = 0.0f;
                    final_angle = MapWheelAngleToServoCmd(0.0, INVERT_STEER);
                } else if (park_state == ParkingState::REVERSE_1) {
                    speed = PARK_REVERSE_SPEED;
                    double wheel_steer = (park_free_side < 0) ? -PARK_MAX_STEER_DEG : PARK_MAX_STEER_DEG;
                    final_angle = MapWheelAngleToServoCmd(wheel_steer, INVERT_STEER);
                } else if (park_state == ParkingState::REVERSE_2) {
                    speed = PARK_REVERSE_SPEED;
                    double wheel_steer = (park_free_side < 0) ? PARK_MAX_STEER_DEG : -PARK_MAX_STEER_DEG;
                    final_angle = MapWheelAngleToServoCmd(wheel_steer, INVERT_STEER);
                } else {
                    // APPROACH, SLOT1_WAIT_BLOCKED, SLOT2_SEARCH_FREE, SLOT3_WAIT_BLOCKED:
                    // Chạy chậm tiến thẳng tuyệt đối: kết hợp PID làn + Steer Trim chống lệch trái + Heading-Hold
                    speed = PARK_APPROACH_SPEED;
                    float heading_corr = 0.0f;
                    if (heading_ok) {
                        float h_err = normalizeAngle(current_heading - park_straight_heading);
                        heading_corr = std::clamp(-PARK_HEADING_KP * h_err, -PARK_HEADING_MAX_CORR_DEG, PARK_HEADING_MAX_CORR_DEG);
                    }
                    double park_steer = out.angle + PARK_STEER_TRIM_DEG + heading_corr;
                    park_steer = std::clamp(park_steer, -16.0, 16.0);
                    final_angle = MapWheelAngleToServoCmd(park_steer, INVERT_STEER);
                }
            } else if (dec.emergency_stop) {
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

                if (rb_state != RbState::NONE && out.valid) {
                    speed = std::clamp(speed, RB_SPEED_MIN_CMS, RB_SPEED_MAX_CMS);
                }
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

                // --- ROUNDABOUT TELEMETRY ---
                (rb_state != RbState::NONE),
                RbStateName(rb_state),
                rb_pass_count,

                // --- OBSTACLE AVOIDANCE TELEMETRY ---
                obs_res.state_str,
                obs_res.obstacle_side,
                obs_res.y_cut,
                obs_res.dist_px,
                obs_res.near_cut,
                obs_res.pass_count,

                // --- T-JUNCTION TELEMETRY ---
                turn_controller.t_junction_count,
                turn_res.tjunction_detected,

                // --- TRAFFIC LIGHT TELEMETRY ---
                navBrain.TrafficLightColor(),
                navBrain.TrafficLightStop(),
                navBrain.GetTLMetrics().top_brightness,
                navBrain.GetTLMetrics().mid_brightness,
                navBrain.GetTLMetrics().bot_brightness,
                navBrain.GetTLMetrics().diff_brightness,

                // --- PARKING TELEMETRY ---
                ParkingStateStr(park_state),
                park_free_side,
                tof_left,
                tof_right,
                current_heading,
                park_heading_0,
                std::abs(normalizeAngle(current_heading - park_heading_0)),

                ui_left_pts,
                ui_right_pts
            );
            
            // In terminal debug ngắn gọn
            std::cout << "[F " << totalFrames << "] "
                    << "Dev: " << std::fixed << std::setprecision(1) << turn_res.dev << "px | "
                    << "Junc: " << (turn_res.junction_opened ? "OPEN" : "WAIT") << " | "
                    << "StrFrames: " << turn_res.straight_frames << "/" << turn_controller.REQUIRED_STRAIGHT_FRAMES;
            if (rb_state != RbState::NONE) {
                std::cout << " | RB: " << RbStateName(rb_state)
                          << " (Pass " << rb_pass_count << "/2, Bias:" << std::fixed << std::setprecision(1) << rb_bias_smoothed << "px)";
            }
            if (obstacle_controller.active()) {
                std::cout << " | OBS: " << obs_res.state_str
                          << " (Y:" << obs_res.y_cut
                          << " D:" << obs_res.dist_px
                          << "px NearCut:" << (obs_res.near_cut ? "1" : "0") << ")";
            }
            if (navBrain.GetTLMetrics().raw_color != "NONE" || navBrain.TrafficLightColor() != "NONE") {
                std::cout << " | TL:" << navBrain.TrafficLightColor()
                          << "(T" << navBrain.GetTLMetrics().top_brightness
                          << "/M" << navBrain.GetTLMetrics().mid_brightness
                          << "/B" << navBrain.GetTLMetrics().bot_brightness
                          << " d" << navBrain.GetTLMetrics().diff_brightness << ")";
            }
            if (parking_active) {
                float delta_h = std::abs(normalizeAngle(current_heading - park_heading_0));
                std::cout << " | PARK: " << ParkingStateStr(park_state)
                          << "(Side:" << (park_free_side < 0 ? "L" : (park_free_side > 0 ? "R" : "-"))
                          << " TOF L:" << std::fixed << std::setprecision(0) << tof_left
                          << " R:" << tof_right
                          << " Hdg:" << std::setprecision(1) << current_heading
                          << " H0:" << park_heading_0
                          << " dH:" << delta_h << "deg)";
            }
            std::cout << " | P0(" << turn_res.p0.x << "," << turn_res.p0.y << ") "
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
                // PARKING SCANLINES UI (Hành lang phối cảnh 2 tia thẳng tuyệt đối)
                // ============================================================
                if (parking_active && park_corridor.latched) {
                    cv::line(overlay, cv::Point(park_corridor.x_left_bot, park_corridor.y_bot),
                                      cv::Point(park_corridor.x_left_far, park_corridor.y_far),
                                      cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
                    cv::line(overlay, cv::Point(park_corridor.x_right_bot, park_corridor.y_bot),
                                      cv::Point(park_corridor.x_right_far, park_corridor.y_far),
                                      cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

                    cv::circle(overlay, cv::Point(park_corridor.x_left_bot, park_corridor.y_bot), 5, cv::Scalar(0, 255, 255), -1);
                    cv::circle(overlay, cv::Point(park_corridor.x_left_far, park_corridor.y_far), 5, cv::Scalar(0, 255, 255), -1);
                    cv::circle(overlay, cv::Point(park_corridor.x_right_bot, park_corridor.y_bot), 5, cv::Scalar(0, 255, 255), -1);
                    cv::circle(overlay, cv::Point(park_corridor.x_right_far, park_corridor.y_far), 5, cv::Scalar(0, 255, 255), -1);

                    cv::putText(overlay, "PARK CORRIDOR (STRAIGHT 2-RAYS)",
                                cv::Point(std::max(10, park_corridor.x_left_far - 30), 45),
                                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
                }

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

                // ============================================================
                // T-JUNCTION GEOMETRIC DETECTION UI (chỉ hiện khi phát hiện hình học T-junction)
                // ============================================================
                if (turn_res.tjunction_detected) {
                    // Vẽ khung đỏ nổi bật toàn bề ngang ở khu vực tầng giữa
                    cv::rectangle(overlay, cv::Rect(0, ui_y2 - 10, frame.cols, 20), cv::Scalar(0, 0, 255), 2);
                    // Text cảnh báo T-junction
                    std::string tj_text = cv::format(">> T-JUNCTION DETECTED (#%d) <<", turn_controller.t_junction_count + 1);
                    cv::putText(overlay, tj_text,
                                cv::Point(static_cast<int>(frame.cols * 0.12), ui_y2 + 30),
                                cv::FONT_HERSHEY_SIMPLEX, 0.70, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
                    // Hiện thêm thông số hình học phía dưới
                    cv::putText(overlay, cv::format("GeoFrames: %d | Locked: %s",
                                turn_controller.tjunction_geo_confirm_frames,
                                turn_controller.tjunction_count_locked ? "YES" : "NO"),
                                cv::Point(15, ui_y2 + 55),
                                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
                }

                // ============================================================
                // OBSTACLE AVOIDANCE UI (vẽ vạch cắt mask, khoảng cách px, và trạng thái né / khóa)
                // ============================================================
                if (obstacle_controller.active()) {
                    // Vẽ đường cắt bậc thềm (Shelf Cut Line) màu cam neon tại hàng y_cut
                    if (obs_res.y_cut > 0) {
                        cv::line(overlay, cv::Point(0, obs_res.y_cut), cv::Point(frame.cols - 1, obs_res.y_cut),
                                 cv::Scalar(0, 140, 255), 2, cv::LINE_AA);
                        std::string cut_info = cv::format("OBSTACLE CUT (Y: %d | Dist: %dpx)", obs_res.y_cut, obs_res.dist_px);
                        cv::putText(overlay, cut_info, cv::Point(frame.cols - 270, obs_res.y_cut - 6),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 140, 255), 1, cv::LINE_AA);
                    }

                    // Banner hiển thị trạng thái né vật cản
                    if (obs_res.state == v20::ObstacleAvoidanceController::State::DODGE) {
                        std::string dodge_text = (obs_res.dodge_dir < 0)
                            ? cv::format(">> DODGE LEFT (Dist: %dpx) <<", obs_res.dist_px)
                            : cv::format(">> DODGE RIGHT (Dist: %dpx) <<", obs_res.dist_px);
                        cv::putText(overlay, dodge_text, cv::Point(static_cast<int>(frame.cols * 0.12), 110),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 165, 255), 2, cv::LINE_AA);
                    } else if (obs_res.state == v20::ObstacleAvoidanceController::State::PASSING) {
                        std::string pass_text = (obs_res.obstacle_side > 0)
                            ? cv::format(">> PASSING OBSTACLE #%d [LOCK RIGHT - KEEP STRAIGHT] <<", obs_res.pass_count + 1)
                            : cv::format(">> PASSING OBSTACLE #%d [LOCK LEFT - KEEP STRAIGHT] <<", obs_res.pass_count + 1);
                        cv::putText(overlay, pass_text, cv::Point(static_cast<int>(frame.cols * 0.05), 110),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.58, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
                    } else if (obs_res.state == v20::ObstacleAvoidanceController::State::CLEARING) {
                        cv::putText(overlay, cv::format(">> OBSTACLE #%d CLEARED -> STABLE <<", obs_res.pass_count),
                                    cv::Point(static_cast<int>(frame.cols * 0.12), 110),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
                    }
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

                // 2B. DRAW STRAIGHT PERSPECTIVE PARKING CORRIDOR (Bright Cyan/Yellow)
                if (parking_active && park_corridor.latched) {
                    cv::line(overlay, cv::Point(park_corridor.x_left_bot, park_corridor.y_bot),
                                      cv::Point(park_corridor.x_left_far, park_corridor.y_far),
                                      cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
                    cv::line(overlay, cv::Point(park_corridor.x_right_bot, park_corridor.y_bot),
                                      cv::Point(park_corridor.x_right_far, park_corridor.y_far),
                                      cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
                    cv::circle(overlay, cv::Point(park_corridor.x_left_bot, park_corridor.y_bot), 5, cv::Scalar(0, 255, 255), -1);
                    cv::circle(overlay, cv::Point(park_corridor.x_left_far, park_corridor.y_far), 5, cv::Scalar(0, 255, 255), -1);
                    cv::circle(overlay, cv::Point(park_corridor.x_right_bot, park_corridor.y_bot), 5, cv::Scalar(0, 255, 255), -1);
                    cv::circle(overlay, cv::Point(park_corridor.x_right_far, park_corridor.y_far), 5, cv::Scalar(0, 255, 255), -1);
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

                std::string sign_ui = (dec.active_sign != "NONE") ? dec.active_sign : handled_sign;
                auto tl_m = navBrain.GetTLMetrics();
                if (navBrain.TrafficLightColor() != "NONE" || tl_m.raw_color != "NONE") {
                    sign_ui += cv::format(" | TL:%s [T:%d M:%d B:%d d:%d]",
                                          navBrain.TrafficLightColor().c_str(),
                                          tl_m.top_brightness,
                                          tl_m.mid_brightness,
                                          tl_m.bot_brightness,
                                          tl_m.diff_brightness);
                }
                if (navBrain.PedestrianStop()) {
                    sign_ui += " [PED STOP]";
                } else if (navBrain.TrafficLightStop()) {
                    sign_ui += " [TL STOP]";
                }

                draw_text(overlay, cv::format("Frame: %ld", totalFrames), cv::Point(15, 25), cv::Scalar(255, 255, 255));
                draw_text(overlay, cv::format("Action: %s", action_to_str(dec.action)), cv::Point(15, 50), cv::Scalar(0, 255, 255));
                std::string voter_ui = cv::format("Sign: %s | Vtr:%s [C:%.2f H:%d]",
                                                  sign_ui.c_str(),
                                                  (committed_class >= 0 ? SignClassName(committed_class) : "NONE"),
                                                  voter.GetCommittedMaxConf(),
                                                  (committed_class >= 0 ? voter.GetHitCount(committed_class) : 0));
                draw_text(overlay, voter_ui, cv::Point(15, 75), cv::Scalar(0, 255, 0));
                draw_text(overlay, cv::format("Speed: %.1f", speed), cv::Point(15, 100), cv::Scalar(255, 255, 255));
                
                std::string turn_state_ui = turn_controller.is_roundabout_mode
                    ? std::string(turn_controller.get_roundabout_phase_str())
                    : std::string(state_to_str(turn_res.state));
                draw_text(overlay, cv::format("Turn State: %s", turn_state_ui.c_str()), cv::Point(15, 125), cv::Scalar(0, 165, 255));
                draw_text(overlay, cv::format("Turn Dir: %s", turn_res.direction.c_str()), cv::Point(15, 150), cv::Scalar(0, 165, 255)); 
                
                draw_text(overlay, cv::format("Turn Bias: %.1fpx", turn_res.steer_bias_px), cv::Point(15, 175), cv::Scalar(255, 255, 0));
                draw_text(overlay, cv::format("Turn Floor: %.1fdeg", applied_floor), cv::Point(15, 200), cv::Scalar(255, 255, 0)); // <--- ĐỔI DÒNG 749 NÀY                
                draw_text(overlay, cv::format("Near E: %.1f | Far E: %.1f", out.near_e, out.far_e), cv::Point(15, 225), cv::Scalar(255, 255, 255));
                draw_text(overlay, cv::format("Lane Angle: %.2f", out.angle), cv::Point(15, 250), cv::Scalar(255, 0, 255));
                if (obstacle_controller.active()) {
                    draw_text(overlay, cv::format("Obstacle: %s (Dist: %dpx)", obs_res.state_str.c_str(), obs_res.dist_px), cv::Point(15, 275), cv::Scalar(0, 200, 255));
                }
                if (navBrain.TrafficLightStop() && turn_controller.state != v20::LaneTurnController::State::IDLE) {
                    draw_text(overlay, cv::format(">> WAITING GREEN TO TURN: %s <<", turn_controller.direction.c_str()),
                              cv::Point(15, 300), cv::Scalar(0, 255, 255));
                }
                if (parking_active) {
                    float delta_h = std::abs(normalizeAngle(current_heading - park_heading_0));
                    draw_text(overlay, cv::format("PARK: %s | Free: %s | Speed: %.1f cm/s",
                              ParkingStateStr(park_state),
                              (park_free_side < 0 ? "LEFT" : (park_free_side > 0 ? "RIGHT" : "NONE")),
                              speed),
                              cv::Point(15, 300), cv::Scalar(0, 255, 255));
                    draw_text(overlay, cv::format("TOF1(L): %.0fmm | TOF3(R): %.0fmm (Blk<%.0f)",
                              tof_left, tof_right, PARK_TOF_BLOCKED_MM),
                              cv::Point(15, 325), cv::Scalar(0, 255, 255));
                    draw_text(overlay, cv::format("Heading: %.1f | Base: %.1f | Delta: %.1f deg",
                              current_heading, park_heading_0, delta_h),
                              cv::Point(15, 350), cv::Scalar(0, 255, 255));
                }
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
    if (tof_ok) tof.Stop();
    if (imu_ok) bno.close();
    if (SHOW_UI) cv::destroyAllWindows();
    std::cout << "\n[System] Stopped safely\n";
    return 0;
}

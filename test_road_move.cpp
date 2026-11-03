#include "hardware/Camera.h"
#include "hardware/Motor.h"
#include "hardware/Servo.h"
#include "common/Logger.h"

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <csignal>

// Cấu hình đường dẫn model
static const char *MODEL_PATH = "/home/car4/Downloads/road/road1.onnx";
// Biến toàn cục để bắt tín hiệu Ctrl+C
volatile std::sig_atomic_t g_stopSignal = 0;
void SignalHandler(int) { g_stopSignal = 1; }

// ============================================================
// ROAD MODEL CLASS
// ============================================================
class RoadModel {
public:
    bool Init(const std::string &modelPath) {
        try {
            env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "RoadSeg");
            Ort::SessionOptions options;
            options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            session_ = std::make_unique<Ort::Session>(*env_, modelPath.c_str(), options);

            Ort::AllocatorWithDefaultOptions allocator;
            auto inputName = session_->GetInputNameAllocated(0, allocator);
            inputName_ = inputName.get();

            auto inputInfo = session_->GetInputTypeInfo(0);
            inputShape_ = inputInfo.GetTensorTypeAndShapeInfo().GetShape();
            inputHeight_ = static_cast<int>(inputShape_[2]);
            inputWidth_ = static_cast<int>(inputShape_[3]);

            auto outputName = session_->GetOutputNameAllocated(0, allocator);
            outputName_ = outputName.get();

            std::cout << "Model loaded! Input: " << inputWidth_ << "x" << inputHeight_ << "\n";
            return true;
        } catch (const std::exception &e) {
            std::cerr << "Error loading model: " << e.what() << "\n";
            return false;
        }
    }

    bool Predict(const cv::Mat &frame, cv::Mat &mask) {
        cv::Mat blob;
        // Tự động resize, BGR->RGB, Normalize và chuyển sang CHW bằng OpenCV DNN
        cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(inputWidth_, inputHeight_), cv::Scalar(), true, false);
        
        std::vector<float> inputData(blob.begin<float>(), blob.end<float>());
        std::vector<int64_t> inputDims = {1, 3, inputHeight_, inputWidth_};
        auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memInfo, inputData.data(), inputData.size(), inputDims.data(), inputDims.size());

        const char *inputNames[] = {inputName_.c_str()};
        const char *outputNames[] = {outputName_.c_str()};
        auto outputs = session_->Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1);

        auto outputShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        int channels = static_cast<int>(outputShape[1]);
        int h = static_cast<int>(outputShape[2]);
        int w = static_cast<int>(outputShape[3]);
        float *outData = outputs[0].GetTensorMutableData<float>();

        cv::Mat smallMask(h, w, CV_8UC1);
        int size = h * w;

        // Xử lý 2 kênh (Nền và Đường)
        if (channels == 2) {
            for (int i = 0; i < size; i++) {
                smallMask.at<uchar>(i) = (outData[size + i] > outData[i]) ? 255 : 0;
            }
        } else {
            return false;
        }

        // Resize và làm mượt mask
        cv::resize(smallMask, mask, frame.size(), 0, 0, cv::INTER_LINEAR);
        cv::threshold(mask, mask, 127, 255, cv::THRESH_BINARY);
        
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

        return true;
    }

private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::string inputName_, outputName_;
    std::vector<int64_t> inputShape_;
    int inputWidth_, inputHeight_;
};

// ============================================================
// MAIN
// ============================================================
int main() {
    std::signal(SIGINT, SignalHandler);
    std::signal(SIGTERM, SignalHandler);

    // 1. Khởi tạo phần cứng
    Camera camera;
    if (!camera.Init()) {
        std::cerr << "Cannot init camera\n";
        return -1;
    }

    Motor motor;
    if (!motor.Init()) {
        std::cerr << "Cannot init motor\n";
        return -1;
    }

    Servo servo;
    if (!servo.Init() || !servo.SetPWMFreq(50)) {
        std::cerr << "Cannot init servo\n";
        return -1;
    }

    // Đảm bảo xe dừng an toàn lúc bắt đầu
    motor.SetSpeedCms(0.0f);
    servo.SetAngle(0);

    // 2. Khởi tạo AI
    RoadModel roadModel;
    if (!roadModel.Init(MODEL_PATH)) return -1;

    // 3. Biến điều khiển
    float currentSpeed = 0.0f;
    int currentSteer = 0;
    const float SPEED_STEP = 5.0f;   // Mỗi lần bấm W/S tăng/giảm 5 cm/s
    const int STEER_STEP = 15;        // Mỗi lần bấm A/D đánh lái 5 độ
    const float MAX_SPEED = 20.0f;   // Tốc độ tối đa (an toàn)
    const int MAX_STEER = 45;        // Góc lái tối đa

    std::cout << "====================================\n";
    std::cout << "      ROAD AI + KEYBOARD MOVE       \n";
    std::cout << "====================================\n";
    std::cout << "W/S : Speed +/- 5 cm/s\n";
    std::cout << "A/D : Steering Left/Right\n";
    std::cout << "C   : Center Steering\n";
    std::cout << "SPACE: STOP (Emergency)\n";
    std::cout << "ESC : Quit\n";
    std::cout << "====================================\n";

    cv::namedWindow("Road Mask", cv::WINDOW_NORMAL);
    cv::resizeWindow("Road Mask", 800, 450); // Thu nhỏ cửa sổ cho dễ nhìn

    cv::Mat frame, mask;

    // 4. Vòng lặp chính
    while (!g_stopSignal) {
        // Lấy ảnh
        camera.GetFrame(frame);
        if (frame.empty()) continue;

        // Chạy AI
        if (roadModel.Predict(frame, mask)) {
            cv::imshow("Road Mask", mask);
        }

        // Bắt phím (Dùng waitKey của cửa sổ Road Mask)
        int key = cv::waitKey(1) & 0xFF;

        if (key == 27 || key == 'q' || key == 'Q') { // ESC hoặc Q
            break;
        }
        else if (key == 'w' || key == 'W') {
            currentSpeed += SPEED_STEP;
            if (currentSpeed > MAX_SPEED) currentSpeed = MAX_SPEED;
            std::cout << "Speed: " << currentSpeed << " cm/s\n";
        }
        else if (key == 's' || key == 'S') {
            currentSpeed -= SPEED_STEP;
            if (currentSpeed < 0) currentSpeed = 0; // Không cho lùi ở chế độ test này
            std::cout << "Speed: " << currentSpeed << " cm/s\n";
        }
        else if (key == 'd' || key == 'D') {
            currentSteer -= STEER_STEP;
            if (currentSteer < -MAX_STEER) currentSteer = -MAX_STEER;
            std::cout << "Steer: " << currentSteer << " deg\n";
        }
        else if (key == 'a' || key == 'A') {
            currentSteer += STEER_STEP;
            if (currentSteer > MAX_STEER) currentSteer = MAX_STEER;
            std::cout << "Steer: " << currentSteer << " deg\n";
        }
        else if (key == 'c' || key == 'C') {
            currentSteer = 0;
            std::cout << "Steer: Centered\n";
        }
        else if (key == ' ' || key == 'x' || key == 'X') {
            currentSpeed = 0.0f;
            currentSteer = 0;
            std::cout << "EMERGENCY STOP!\n";
        }

        // Gửi lệnh xuống phần cứng
        motor.SetSpeedCms(currentSpeed);
        servo.SetAngle(currentSteer);
    }

    // 5. Dọn dẹp an toàn
    std::cout << "Stopping safely...\n";
    motor.SetSpeedCms(0.0f);
    servo.SetAngle(0);
    cv::destroyAllWindows();

    return 0;
}

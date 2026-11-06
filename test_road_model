#include "hardware/Camera.h"
#include "hardware/Motor.h"
#include "hardware/Servo.h"
#include "common/Logger.h"

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
#include <cstdint>
#include <limits>

// ============================================================
// CONFIGURATION
// ============================================================

static const char* MODEL_PATH = "/home/car4/Downloads/road/kga2.engine";

// Class index của Drivable Area (da_logits output có 2 kênh[cite: 1]: channel 0 = background, channel 1 = drivable area).
static constexpr int DA_CHANNEL = 1;

static constexpr float SPEED_STEP = 5.0f;
static constexpr int STEER_STEP = 15;

static constexpr float MAX_SPEED = 20.0f;
static constexpr int MAX_STEER = 45;

static constexpr float OVERLAY_ALPHA = 0.40f;
static constexpr int DEBUG_INTERVAL = 60;

// ============================================================
// GLOBAL
// ============================================================

volatile std::sig_atomic_t g_stopSignal = 0;

void SignalHandler(int)
{
    g_stopSignal = 1;
}

// ============================================================
// CUDA CHECK
// ============================================================

static void CheckCuda(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status)
        );
    }
}

// ============================================================
// TENSORRT LOGGER
// ============================================================

class TrtLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
            std::cerr << "[TensorRT] " << msg << std::endl;
        }
    }
};

static TrtLogger gTrtLogger;

// ============================================================
// READ ENGINE
// ============================================================

static std::vector<char> ReadEngineFile(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Cannot open engine: " + filename);
    }

    file.seekg(0, std::ios::end);
    const std::streamsize size = file.tellg();
    if (size <= 0)
    {
        throw std::runtime_error("Engine file is empty");
    }

    file.seekg(0, std::ios::beg);
    std::vector<char> data(static_cast<size_t>(size));
    if (!file.read(data.data(), size))
    {
        throw std::runtime_error("Cannot read engine file");
    }

    return data;
}

// ============================================================
// MODEL INTERFACE
// ============================================================

class IRoadModel
{
public:
    virtual ~IRoadModel() = default;
    virtual bool Init(const std::string& modelPath) = 0;
    virtual bool Predict(const cv::Mat& frame, cv::Mat& daMask) = 0;
};

// ============================================================
// TRILITENET TENSORRT MODEL IMPLEMENTATION
// ============================================================

class TriLiteRoadModel : public IRoadModel
{
public:
    TriLiteRoadModel() = default;

    ~TriLiteRoadModel() override
    {
        Cleanup();
    }

    bool Init(const std::string& modelPath) override
    {
        try
        {
            std::cout << "\n[TensorRT] Loading engine: " << modelPath << std::endl;

            std::vector<char> engineData = ReadEngineFile(modelPath);

            runtime_ = nvinfer1::createInferRuntime(gTrtLogger);
            if (!runtime_)
            {
                throw std::runtime_error("createInferRuntime failed");
            }

            engine_ = runtime_->deserializeCudaEngine(engineData.data(), engineData.size());
            if (!engine_)
            {
                throw std::runtime_error("deserializeCudaEngine failed");
            }

            context_ = engine_->createExecutionContext();
            if (!context_)
            {
                throw std::runtime_error("createExecutionContext failed");
            }

            const int nbBindings = engine_->getNbBindings();
            if (nbBindings < 2)
            {
                throw std::runtime_error("Engine must have at least 2 bindings");
            }

            buffers_.assign(nbBindings, nullptr);
            inputIndex_ = -1;
            outputIndex_ = -1;

            std::cout << "\n========== ENGINE BINDINGS ==========\n";
            for (int i = 0; i < nbBindings; ++i)
            {
                const char* bindingName = engine_->getBindingName(i);
                const bool isInput = engine_->bindingIsInput(i);
                const nvinfer1::Dims dims = engine_->getBindingDimensions(i);
                const nvinfer1::DataType type = engine_->getBindingDataType(i);

                size_t elementCount = 1;
                for (int j = 0; j < dims.nbDims; ++j)
                {
                    elementCount *= static_cast<size_t>(dims.d[j]);
                }

                std::cout << "Binding " << i << " [" << bindingName << "] "
                          << (isInput ? "INPUT" : "OUTPUT")
                          << " Elements: " << elementCount << "\n";

                if (isInput)
                {
                    inputIndex_ = i;
                    inputHeight_ = dims.d[2];
                    inputWidth_ = dims.d[3];
                    inputData_.resize(elementCount);
                }
                else
                {
                    // da_logits output [1, 2, H, W][cite: 1]
                    outputIndex_ = i;
                    outputHeight_ = dims.d[2];
                    outputWidth_ = dims.d[3];
                    outputData_.resize(elementCount);
                }

                CheckCuda(
                    cudaMalloc(&buffers_[i], elementCount * sizeof(float)),
                    "cudaMalloc binding"
                );
            }

            if (inputIndex_ < 0 || outputIndex_ < 0)
            {
                throw std::runtime_error("Input/Output binding not found");
            }

            CheckCuda(cudaStreamCreate(&stream_), "cudaStreamCreate");

            std::cout << "\n[TensorRT] Model initialized successfully!"
                      << "\nInput Size: " << inputWidth_ << "x" << inputHeight_
                      << "\nOutput Size: " << outputWidth_ << "x" << outputHeight_ << "\n\n";

            return true;
        }
        catch (const std::exception& e)
        {
            std::cerr << "\n[Init Error] " << e.what() << std::endl;
            Cleanup();
            return false;
        }
    }

    bool Preprocess(const cv::Mat& frame)
    {
        if (frame.empty() || frame.channels() != 3)
            return false;

        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(inputWidth_, inputHeight_), 0, 0, cv::INTER_LINEAR);

        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

        cv::Mat floatImg;
        rgb.convertTo(floatImg, CV_32FC3, 1.0 / 255.0);

        // HWC -> CHW conversion
        std::vector<cv::Mat> channels(3);
        cv::split(floatImg, channels);

        const size_t planeSize = static_cast<size_t>(inputWidth_) * inputHeight_;
        for (int c = 0; c < 3; ++c)
        {
            if (!channels[c].isContinuous())
            {
                channels[c] = channels[c].clone();
            }

            std::memcpy(
                inputData_.data() + static_cast<size_t>(c) * planeSize,
                channels[c].ptr<float>(),
                planeSize * sizeof(float)
            );
        }

        return true;
    }

    bool Predict(const cv::Mat& frame, cv::Mat& daMask) override
    {
        if (frame.empty() || !context_ || !stream_)
            return false;

        try
        {
            if (!Preprocess(frame))
                return false;

            // 1. Copy input H2D
            CheckCuda(
                cudaMemcpyAsync(
                    buffers_[inputIndex_],
                    inputData_.data(),
                    inputData_.size() * sizeof(float),
                    cudaMemcpyHostToDevice,
                    stream_
                ),
                "Copy input H2D"
            );

            // 2. Inference
            if (!context_->enqueueV2(buffers_.data(), stream_, nullptr))
            {
                throw std::runtime_error("TensorRT enqueueV2 failed");
            }

            // 3. Copy output D2H
            CheckCuda(
                cudaMemcpyAsync(
                    outputData_.data(),
                    buffers_[outputIndex_],
                    outputData_.size() * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    stream_
                ),
                "Copy output D2H"
            );

            CheckCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");

            const size_t planeSize = static_cast<size_t>(outputWidth_) * outputHeight_;
            const float* ch0 = outputData_.data();
            const float* ch1 = outputData_.data() + planeSize;

            const float* da = (DA_CHANNEL == 1) ? ch1 : ch0;
            const float* other = (DA_CHANNEL == 1) ? ch0 : ch1;

            // 4. Argmax segmentation mask
            cv::Mat maskSmall(outputHeight_, outputWidth_, CV_8UC1, cv::Scalar(0));
            int daPixels = 0;

            for (int y = 0; y < outputHeight_; ++y)
            {
                const size_t rowOffset = static_cast<size_t>(y) * outputWidth_;
                uchar* maskRow = maskSmall.ptr<uchar>(y);

                for (int x = 0; x < outputWidth_; ++x)
                {
                    const size_t index = rowOffset + x;
                    const bool isDA = da[index] > other[index];

                    maskRow[x] = isDA ? 255 : 0;
                    if (isDA) ++daPixels;
                }
            }

            // 5. Resize mask về lại kích thước khung hình gốc của camera
            cv::resize(maskSmall, daMask, frame.size(), 0, 0, cv::INTER_NEAREST);
            return true;
        }
        catch (const std::exception& e)
        {
            std::cerr << "[Predict Error] " << e.what() << std::endl;
            daMask.release();
            return false;
        }
    }

private:
    void Cleanup()
    {
        if (stream_) cudaStreamSynchronize(stream_);
        for (void*& buffer : buffers_)
        {
            if (buffer)
            {
                cudaFree(buffer);
                buffer = nullptr;
            }
        }
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
    std::vector<float> inputData_;
    std::vector<float> outputData_;

    int inputIndex_ = -1;
    int outputIndex_ = -1;
    int inputWidth_ = 0;
    int inputHeight_ = 0;
    int outputWidth_ = 0;
    int outputHeight_ = 0;
};

// ============================================================
// MAIN FUNCTION
// ============================================================

int main()
{
    std::signal(SIGINT, SignalHandler);
    std::signal(SIGTERM, SignalHandler);

    Camera camera;
    if (!camera.Init())
    {
        std::cerr << "Cannot initialize camera\n";
        return -1;
    }

    Motor motor;
    if (!motor.Init())
    {
        std::cerr << "Cannot initialize motor\n";
        return -1;
    }
    motor.SetUartLogging(false);

    Servo servo;
    if (!servo.Init() || !servo.SetPWMFreq(50))
    {
        std::cerr << "Cannot initialize servo\n";
        motor.SetSpeedCms(0.0f);
        return -1;
    }

    motor.SetSpeedCms(0.0f);
    servo.SetAngle(0);

    TriLiteRoadModel roadModel;
    if (!roadModel.Init(MODEL_PATH))
    {
        std::cerr << "Cannot initialize TensorRT model from engine\n";
        motor.SetSpeedCms(0.0f);
        servo.SetAngle(0);
        return -1;
    }

    float currentSpeed = 0.0f;
    int currentSteer = 0;

    cv::namedWindow("Road Overlay", cv::WINDOW_NORMAL);
    cv::resizeWindow("Road Overlay", 800, 450);

    cv::Mat frame;
    cv::Mat daMask;
    cv::Mat overlay;

    const int FPS_WINDOW = 30;
    std::vector<double> fpsHistory;
    double fps = 0.0;
    char fpsText[128];

    std::cout << "\n========== CONTROLS ==========\n"
              << "W : Increase speed\n"
              << "S : Decrease speed\n"
              << "A : Steer left\n"
              << "D : Steer right\n"
              << "C : Center steering\n"
              << "SPACE/X : Emergency stop\n"
              << "Q/ESC : Quit\n"
              << "==============================\n";

    while (!g_stopSignal)
    {
        const double tStart = static_cast<double>(cv::getTickCount());

        camera.GetFrame(frame);
        if (frame.empty())
        {
            currentSpeed = 0.0f;
            currentSteer = 0;
            motor.SetSpeedCms(0.0f);
            servo.SetAngle(0);
            cv::waitKey(1);
            continue;
        }

        const bool inferenceOK = roadModel.Predict(frame, daMask);

        if (inferenceOK)
        {
            cv::Mat greenLayer(frame.size(), frame.type(), cv::Scalar(0, 255, 0));
            overlay = frame.clone();
            greenLayer.copyTo(overlay, daMask);

            cv::addWeighted(frame, 1.0 - OVERLAY_ALPHA, overlay, OVERLAY_ALPHA, 0.0, overlay);
        }
        else
        {
            currentSpeed = 0.0f;
            currentSteer = 0;
            motor.SetSpeedCms(0.0f);
            servo.SetAngle(0);
            overlay = frame.clone();
        }

        // FPS Calculation
        const double tEnd = static_cast<double>(cv::getTickCount());
        const double dt = (tEnd - tStart) / cv::getTickFrequency();
        if (dt > 0.0)
        {
            fpsHistory.push_back(1.0 / dt);
            if (static_cast<int>(fpsHistory.size()) > FPS_WINDOW)
            {
                fpsHistory.erase(fpsHistory.begin());
            }
            double sum = 0.0;
            for (double val : fpsHistory) sum += val;
            fps = sum / fpsHistory.size();
        }

        std::snprintf(fpsText, sizeof(fpsText), "FPS %.1f | %s", fps, inferenceOK ? "OK" : "ERROR");
        cv::putText(overlay, fpsText, cv::Point(20, 45), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    inferenceOK ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);

        cv::imshow("Road Overlay", overlay);

        const int key = cv::waitKey(1) & 0xFF;
        if (key == 27 || key == 'q' || key == 'Q')
        {
            break;
        }

        if (inferenceOK)
        {
            if (key == 'w' || key == 'W')
                currentSpeed = std::min(currentSpeed + SPEED_STEP, MAX_SPEED);
            else if (key == 's' || key == 'S')
                currentSpeed = std::max(currentSpeed - SPEED_STEP, 0.0f);
            else if (key == 'a' || key == 'A')
                currentSteer = std::min(currentSteer + STEER_STEP, MAX_STEER);
            else if (key == 'd' || key == 'D')
                currentSteer = std::max(currentSteer - STEER_STEP, -MAX_STEER);
            else if (key == 'c' || key == 'C')
                currentSteer = 0;
            else if (key == ' ' || key == 'x' || key == 'X')
            {
                currentSpeed = 0.0f;
                currentSteer = 0;
            }
        }
        else
        {
            currentSpeed = 0.0f;
            currentSteer = 0;
        }

        motor.SetSpeedCms(currentSpeed);
        servo.SetAngle(currentSteer);
    }

    motor.SetSpeedCms(0.0f);
    servo.SetAngle(0);
    cv::destroyAllWindows();

    std::cout << "\n[System] Stopped safely\n";
    return 0;
}

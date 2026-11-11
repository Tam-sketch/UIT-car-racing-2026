// ============================================================
// SignDetector.h — YOLOv8 TensorRT detector (thư mục gốc)
// ============================================================
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <memory>

struct SignObject {
    int class_id = -1;
    std::string name;
    float confidence = 0.0f;
    cv::Rect box;
};

class SignDetector {
public:
    inline static const std::vector<std::string> CLASS_NAMES = {
        "construction_ahead", "no_left", "no_right", "parking",
        "roundabout", "traffic_light", "turn_left", "turn_right"
    };

    SignDetector() = default;
    ~SignDetector() { Cleanup(); }

    bool Init(const std::string& enginePath) {
        try {
            std::cout << "[SignDetector] Loading Engine: " << enginePath << "\n";
            std::ifstream file(enginePath, std::ios::binary);
            if (!file) {
                std::cerr << "[SignDetector] Cannot open: " << enginePath << "\n";
                return false;
            }
            file.seekg(0, std::ios::end);
            size_t size = file.tellg();
            file.seekg(0, std::ios::beg);
            std::vector<char> buffer(size);
            file.read(buffer.data(), size);

            runtime_ = nvinfer1::createInferRuntime(logger_);
            if (!runtime_) return false;
            engine_  = runtime_->deserializeCudaEngine(buffer.data(), buffer.size());
            if (!engine_)  return false;
            context_ = engine_->createExecutionContext();
            if (!context_) return false;

            const int nb = engine_->getNbBindings();
            buffers_.assign(nb, nullptr);
            for (int i = 0; i < nb; ++i) {
                nvinfer1::Dims dims = engine_->getBindingDimensions(i);
                size_t total = 1;
                for (int j = 0; j < dims.nbDims; ++j) total *= dims.d[j];
                cudaMalloc(&buffers_[i], total * sizeof(float));
                if (engine_->bindingIsInput(i)) {
                    inputIndex_ = i;
                    inputHeight_ = dims.d[2];
                    inputWidth_  = dims.d[3];
                    inputData_.resize(total);
                } else {
                    outputIndex_ = i;
                    outputData_.resize(total);
                }
            }
            cudaStreamCreate(&stream_);
            std::cout << "[SignDetector] Ready. Input " << inputWidth_
                      << "x" << inputHeight_ << "\n";
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[SignDetector Error] " << e.what() << "\n";
            Cleanup();
            return false;
        }
    }

    std::vector<SignObject> Detect(const cv::Mat& frame,
                                   float confThresh = 0.70f,
                                   float nmsThresh  = 0.45f) {
        std::vector<SignObject> results;
        if (frame.empty() || !context_ || !stream_) return results;

        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0,
                               cv::Size(inputWidth_, inputHeight_),
                               cv::Scalar(), true, false);
        std::memcpy(inputData_.data(), blob.ptr<float>(),
                    inputData_.size() * sizeof(float));

        cudaMemcpyAsync(buffers_[inputIndex_], inputData_.data(),
                        inputData_.size() * sizeof(float),
                        cudaMemcpyHostToDevice, stream_);
        context_->enqueueV2(buffers_.data(), stream_, nullptr);
        cudaMemcpyAsync(outputData_.data(), buffers_[outputIndex_],
                        outputData_.size() * sizeof(float),
                        cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);

        const int num_classes = (int)CLASS_NAMES.size();
        const int num_boxes = (int)outputData_.size() / (4 + num_classes);

        std::vector<cv::Rect> boxes;
        std::vector<float>    confidences;
        std::vector<int>      class_ids;

        const float x_scale = (float)frame.cols / inputWidth_;
        const float y_scale = (float)frame.rows / inputHeight_;

        for (int i = 0; i < num_boxes; ++i) {
            float max_score = 0.0f;
            int   best_cls  = -1;
            for (int c = 0; c < num_classes; ++c) {
                float s = outputData_[(4 + c) * num_boxes + i];
                if (s > max_score) { max_score = s; best_cls = c; }
            }
            if (max_score < confThresh || best_cls < 0) continue;

            float cx = outputData_[0 * num_boxes + i];
            float cy = outputData_[1 * num_boxes + i];
            float w  = outputData_[2 * num_boxes + i];
            float h  = outputData_[3 * num_boxes + i];

            int left = (int)((cx - w * 0.5f) * x_scale);
            int top  = (int)((cy - h * 0.5f) * y_scale);
            int bw   = (int)(w * x_scale);
            int bh   = (int)(h * y_scale);

            left = std::clamp(left, 0, frame.cols - 1);
            top  = std::clamp(top,  0, frame.rows - 1);
            bw   = std::clamp(bw,   1, frame.cols - left);
            bh   = std::clamp(bh,   1, frame.rows - top);

            boxes.push_back(cv::Rect(left, top, bw, bh));
            confidences.push_back(max_score);
            class_ids.push_back(best_cls);
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, confThresh, nmsThresh, indices);

        for (int idx : indices) {
            SignObject obj;
            obj.class_id   = class_ids[idx];
            obj.name       = CLASS_NAMES[class_ids[idx]];
            obj.confidence = confidences[idx];
            obj.box        = boxes[idx];
            results.push_back(obj);
        }
        return results;
    }

    void Draw(cv::Mat& frame, const std::vector<SignObject>& signs) {
        for (const auto& s : signs) {
            cv::rectangle(frame, s.box, cv::Scalar(0, 255, 0), 2);
            char text[64];
            std::snprintf(text, sizeof(text), "%s %.2f",
                          s.name.c_str(), s.confidence);
            int y = std::max(s.box.y - 5, 12);
            cv::putText(frame, text, {s.box.x, y},
                        cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 255, 0), 1);
        }
    }

private:
    void Cleanup() {
        if (stream_) { cudaStreamSynchronize(stream_);
                       cudaStreamDestroy(stream_); stream_ = nullptr; }
        for (void*& b : buffers_) if (b) { cudaFree(b); b = nullptr; }
        buffers_.clear();
        if (context_) { delete context_; context_ = nullptr; }
        if (engine_)  { delete engine_;  engine_  = nullptr; }
        if (runtime_) { delete runtime_; runtime_ = nullptr; }
    }

    class Logger : public nvinfer1::ILogger {
    public:
        void log(Severity sev, const char* msg) noexcept override {
            if (sev <= Severity::kWARNING)
                std::cerr << "[TRT-Sign] " << msg << "\n";
        }
    } logger_;

    nvinfer1::IRuntime*          runtime_ = nullptr;
    nvinfer1::ICudaEngine*       engine_  = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    cudaStream_t                 stream_  = nullptr;
    std::vector<void*>           buffers_;
    std::vector<float>           inputData_, outputData_;
    int inputIndex_ = -1, outputIndex_ = -1;
    int inputWidth_ = 0, inputHeight_ = 0;
};
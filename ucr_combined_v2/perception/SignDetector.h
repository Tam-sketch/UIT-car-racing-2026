// ============================================================
// ObjectDetector.h — TensorRT Object & Traffic Sign Detector (YOLOv8 Engine)
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

struct DetectObject {
    int class_id = -1;
    std::string name;
    float confidence = 0.0f;
    cv::Rect box;
};

class ObjectDetector {
public:
    // QUAN TRỌNG: Cập nhật danh sách này ĐÚNG THỨ TỰ (index 0, 1, 2...) 
    // như trong file data.yaml lúc bạn train model mới.
    inline static const std::vector<std::string> CLASS_NAMES = {
        "construction_ahead", 
        "no_left", 
        "no_right", 
        "parking",
        "roundabout", 
        "traffic_light", 
        "turn_left", 
        "turn_right",
        "walking_person" // <-- Thêm người đi bộ và các class khác vào đây
    };

    ObjectDetector() = default;

    ~ObjectDetector() {
        Cleanup();
    }

    bool Init(const std::string& enginePath) {
        try {
            std::cout << "[ObjectDetector] Loading Engine: " << enginePath << "\n";
            std::ifstream file(enginePath, std::ios::binary);
            if (!file) {
                std::cerr << "[ObjectDetector] Cannot open engine file: " << enginePath << "\n";
                return false;
            }

            file.seekg(0, std::ios::end);
            size_t size = file.tellg();
            file.seekg(0, std::ios::beg);
            std::vector<char> buffer(size);
            file.read(buffer.data(), size);

            runtime_ = nvinfer1::createInferRuntime(logger_);
            if (!runtime_) return false;

            engine_ = runtime_->deserializeCudaEngine(buffer.data(), buffer.size());
            if (!engine_) return false;

            context_ = engine_->createExecutionContext();
            if (!context_) return false;

            int nbBindings = engine_->getNbBindings();
            buffers_.assign(nbBindings, nullptr);

            for (int i = 0; i < nbBindings; ++i) {
                nvinfer1::Dims dims = engine_->getBindingDimensions(i);
                size_t total_elements = 1;
                for (int j = 0; j < dims.nbDims; ++j) {
                    total_elements *= dims.d[j];
                }

                cudaMalloc(&buffers_[i], total_elements * sizeof(float));

                if (engine_->bindingIsInput(i)) {
                    inputIndex_ = i;
                    inputHeight_ = dims.d[2];
                    inputWidth_  = dims.d[3];
                    inputData_.resize(total_elements);
                } else {
                    outputIndex_ = i;
                    outputData_.resize(total_elements);
                }
            }

            cudaStreamCreate(&stream_);
            std::cout << "[ObjectDetector] Engine Ready! Input size: " 
                      << inputWidth_ << "x" << inputHeight_ << "\n";
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[ObjectDetector Error] " << e.what() << "\n";
            Cleanup();
            return false;
        }
    }

    std::vector<DetectObject> Detect(const cv::Mat& frame, float confThresh = 0.70f, float nmsThresh = 0.45f) {
        std::vector<DetectObject> results;
        if (frame.empty() || !context_ || !stream_) return results;

        // 1. Preprocess ảnh theo chuẩn RGB 1/255.0
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, 
                               cv::Size(inputWidth_, inputHeight_), 
                               cv::Scalar(), true, false);
        std::memcpy(inputData_.data(), blob.ptr<float>(), inputData_.size() * sizeof(float));

        // 2. Chạy Inference qua TensorRT
        cudaMemcpyAsync(buffers_[inputIndex_], inputData_.data(), 
                        inputData_.size() * sizeof(float), cudaMemcpyHostToDevice, stream_);
        context_->enqueueV2(buffers_.data(), stream_, nullptr);
        cudaMemcpyAsync(outputData_.data(), buffers_[outputIndex_], 
                        outputData_.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);

        // 3. Postprocess YOLOv8 output: [1, 4 + num_classes, num_boxes]
        int num_classes = static_cast<int>(CLASS_NAMES.size());
        
        // Kiểm tra an toàn: Đảm bảo output size khớp với cấu trúc model
        if (outputData_.empty() || outputData_.size() % (4 + num_classes) != 0) {
             std::cerr << "[ObjectDetector] Cảnh báo: Kích thước output không khớp với số lượng class khai báo!\n";
             return results;
        }

        int num_boxes = static_cast<int>(outputData_.size()) / (4 + num_classes);

        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;

        float x_scale = static_cast<float>(frame.cols) / inputWidth_;
        float y_scale = static_cast<float>(frame.rows) / inputHeight_;

        for (int i = 0; i < num_boxes; ++i) {
            float max_score = 0.0f;
            int best_cls = -1;

            for (int c = 0; c < num_classes; ++c) {
                float score = outputData_[(4 + c) * num_boxes + i];
                if (score > max_score) {
                    max_score = score;
                    best_cls = c;
                }
            }

            if (max_score >= confThresh && best_cls != -1) {
                float cx = outputData_[0 * num_boxes + i];
                float cy = outputData_[1 * num_boxes + i];
                float w  = outputData_[2 * num_boxes + i];
                float h  = outputData_[3 * num_boxes + i];

                int left = static_cast<int>((cx - w / 2.0f) * x_scale);
                int top  = static_cast<int>((cy - h / 2.0f) * y_scale);
                int bw   = static_cast<int>(w * x_scale);
                int bh   = static_cast<int>(h * y_scale);

                // Giới hạn trong kích thước ảnh gốc tránh out-of-bounds
                left = std::clamp(left, 0, frame.cols - 1);
                top  = std::clamp(top, 0, frame.rows - 1);
                bw   = std::clamp(bw, 1, frame.cols - left);
                bh   = std::clamp(bh, 1, frame.rows - top);

                boxes.push_back(cv::Rect(left, top, bw, bh));
                confidences.push_back(max_score);
                class_ids.push_back(best_cls);
            }
        }

        // 4. Non-Maximum Suppression (NMS)
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, confThresh, nmsThresh, indices);

        for (int idx : indices) {
            DetectObject obj;
            obj.class_id = class_ids[idx];
            obj.name = CLASS_NAMES[class_ids[idx]];
            obj.confidence = confidences[idx];
            obj.box = boxes[idx];
            results.push_back(obj);
        }

        return results;
    }

    void Draw(cv::Mat& frame, const std::vector<DetectObject>& objects) {
        for (const auto& obj : objects) {
            // Đổi màu box nếu là người đi bộ (ví dụ: Đỏ cho người đi bộ, Xanh cho biển báo)
            cv::Scalar color = (obj.name == "walking_person") ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);

            cv::rectangle(frame, obj.box, color, 2);
            std::string text = obj.name + " " + std::to_string(obj.confidence).substr(0, 4);
            int baseline = 0;
            cv::Size sz = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            int y_text = std::max(obj.box.y, sz.height + 5);
            cv::rectangle(frame, cv::Point(obj.box.x, y_text - sz.height - 5),
                          cv::Point(obj.box.x + sz.width, y_text), color, cv::FILLED);
            cv::putText(frame, text, cv::Point(obj.box.x, y_text - 3),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }
    }

private:
    void Cleanup() {
        if (stream_) {
            cudaStreamSynchronize(stream_);
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
        for (void*& b : buffers_) {
            if (b) { cudaFree(b); b = nullptr; }
        }
        buffers_.clear();
        if (context_) { delete context_; context_ = nullptr; }
        if (engine_)  { delete engine_; engine_ = nullptr; }
        if (runtime_) { delete runtime_; runtime_ = nullptr; }
    }

    class Logger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override {
            if (severity <= Severity::kWARNING) {
                std::cerr << "[TRT-Detector] " << msg << "\n";
            }
        }
    } logger_;

    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    cudaStream_t stream_ = nullptr;

    std::vector<void*> buffers_;
    std::vector< float > inputData_, outputData_;
    int inputIndex_ = -1, outputIndex_ = -1;
    int inputWidth_ = 0, inputHeight_ = 0;
};

using SignObject   = DetectObject;
using SignDetector = ObjectDetector;

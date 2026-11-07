#pragma once
#include "perception/TrtSession.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <iostream>
#include <string>

namespace ucr {
namespace perception {

struct Detection {
    int   class_id;
    float confidence;
    cv::Rect bbox;
};

class SignDetector {
public:
    struct Config {
        float conf_threshold = 0.25f;
        float nms_threshold  = 0.45f;
    };

    // Hàm Init 1: Không dùng default argument (tránh lỗi GCC Jetson Nano)
    bool Init(const std::string& enginePath) {
        return Init(enginePath, Config{});
    }

    // Hàm Init 2: Nhận Config
    bool Init(const std::string& enginePath, const Config& cfg) {
        cfg_ = cfg;
        if (!session_.Init(enginePath)) return false;
        
        if (session_.NumInputs() < 1 || session_.NumOutputs() < 1) {
            std::cerr << "[SignDetector] Engine has no inputs or outputs\n";
            return false;
        }

        const auto& inB  = session_.InputBinding(0);
        const auto& outB = session_.OutputBinding(0);

        if (inB.dims.nbDims != 4) {
            std::cerr << "[SignDetector] Expected 4D NCHW input\n";
            return false;
        }

        inH_ = inB.dims.d[2];
        inW_ = inB.dims.d[3];
        
        // Giả định output YOLO: [1, 4 + num_classes, 8400]
        outNum_ = outB.dims.d[1];
        outAnchors_ = outB.dims.d[2];

        inBuf_.resize(static_cast<size_t>(3) * inW_ * inH_);
        outBuf_.resize(static_cast<size_t>(outNum_) * outAnchors_);

        std::cout << "[SignDetector] in " << inW_ << "x" << inH_ 
                  << " out " << outNum_ << "x" << outAnchors_ << "\n";
        return true;
    }

    bool Predict(const cv::Mat& bgrFrame, std::vector<Detection>& detections) {
        detections.clear();
        if (!session_.IsReady() || bgrFrame.empty()) return false;

        // 1. Tiền xử lý
        cv::Mat resized;
        cv::resize(bgrFrame, resized, cv::Size(inW_, inH_), 0, 0, cv::INTER_LINEAR);
        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        cv::Mat fimg;
        rgb.convertTo(fimg, CV_32FC3, 1.0 / 255.0);

        const size_t plane = static_cast<size_t>(inW_) * inH_;
        const float* src   = fimg.ptr<float>();
        for (size_t i = 0; i < plane; ++i) {
            inBuf_[i]             = src[i * 3 + 0];
            inBuf_[i + plane]     = src[i * 3 + 1];
            inBuf_[i + 2 * plane] = src[i * 3 + 2];
        }

        // 2. Chạy inference
        if (!session_.Run(inBuf_, outBuf_)) return false;

        // 3. Hậu xử lý (YOLOv8/v11 format: [1, 4+C, 8400])
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;

        float scale_x = static_cast<float>(bgrFrame.cols) / inW_;
        float scale_y = static_cast<float>(bgrFrame.rows) / inH_;

        for (int i = 0; i < outAnchors_; ++i) {
            float max_conf = 0.f;
            int best_cls = -1;
            
            // Duyệt qua các class (từ index 4 trở đi)
            for (int c = 4; c < outNum_; ++c) {
                float conf = outBuf_[c * outAnchors_ + i];
                if (conf > max_conf) {
                    max_conf = conf;
                    best_cls = c - 4;
                }
            }

            if (max_conf > cfg_.conf_threshold) {
                float cx = outBuf_[0 * outAnchors_ + i];
                float cy = outBuf_[1 * outAnchors_ + i];
                float w  = outBuf_[2 * outAnchors_ + i];
                float h  = outBuf_[3 * outAnchors_ + i];

                // Chuyển về tọa độ ảnh gốc
                float x = (cx - w / 2.f) * scale_x;
                float y = (cy - h / 2.f) * scale_y;
                float width = w * scale_x;
                float height = h * scale_y;

                boxes.push_back(cv::Rect(x, y, width, height));
                confidences.push_back(max_conf);
                class_ids.push_back(best_cls);
            }
        }

        // 4. NMS
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, cfg_.conf_threshold, cfg_.nms_threshold, indices);

        for (int idx : indices) {
            int cls = class_ids[idx];
            // Map construction_ahead (0) thành turn_left (6)
            if (cls == 0) cls = 6;
            detections.push_back({cls, confidences[idx], boxes[idx]});
        }
        return true;
    }

private:
    TrtSession session_;
    Config cfg_;
    int inW_ = 0, inH_ = 0;
    int outNum_ = 0, outAnchors_ = 0;
    std::vector<float> inBuf_, outBuf_;
};

} // namespace perception
} // namespace ucr
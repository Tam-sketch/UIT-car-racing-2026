#pragma once
#include "perception/TrtSession.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <algorithm>

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

    bool Init(const std::string& enginePath) {
        return Init(enginePath, Config{});
    }

    bool Init(const std::string& enginePath, const Config& cfg) {
        cfg_ = cfg;
        if (!session_.Init(enginePath)) return false;
        
        if (session_.NumInputs() < 1 || session_.NumOutputs() < 1) return false;

        const auto& inB  = session_.InputBinding(0);
        const auto& outB = session_.OutputBinding(0);

        if (inB.dims.nbDims != 4 || outB.dims.nbDims != 3) return false;

        inH_ = inB.dims.d[2];
        inW_ = inB.dims.d[3];
        
        // Xử lý layout: [1, 12, 3549] (YOLOv8) hoặc [1, 3549, 12]
        int d1 = outB.dims.d[1];
        int d2 = outB.dims.d[2];

        if (d1 < d2) {
            numChannels_ = d1;
            numAnchors_  = d2;
            strideC_     = numAnchors_;
            strideA_     = 1;
        } else {
            numChannels_ = d2;
            numAnchors_  = d1;
            strideC_     = 1;
            strideA_     = numChannels_;
        }

        inBuf_.resize(static_cast<size_t>(3) * inW_ * inH_);
        outBuf_.resize(static_cast<size_t>(numChannels_) * numAnchors_);

        std::cout << "[SignDetector] in " << inW_ << "x" << inH_ 
                  << " | out: " << numChannels_ << " ch, " << numAnchors_ << " anchors\n";
        return true;
    }

    bool Predict(const cv::Mat& bgrFrame, std::vector<Detection>& detections) {
        detections.clear();
        if (!session_.IsReady() || bgrFrame.empty()) return false;

        // 1. Letterbox Resize
        float r = std::min(static_cast<float>(inW_) / bgrFrame.cols, 
                           static_cast<float>(inH_) / bgrFrame.rows);
        int new_unpad_w = static_cast<int>(bgrFrame.cols * r);
        int new_unpad_h = static_cast<int>(bgrFrame.rows * r);
        
        cv::Mat resized;
        cv::resize(bgrFrame, resized, cv::Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LINEAR);

        int dw = inW_ - new_unpad_w;
        int dh = inH_ - new_unpad_h;
        int top = dh / 2;
        int bottom = dh - top;
        int left = dw / 2;
        int right = dw - left;

        cv::Mat padded;
        cv::copyMakeBorder(resized, padded, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

        // BGR -> RGB, Normalize
        cv::Mat rgb;
        cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
        cv::Mat fimg;
        rgb.convertTo(fimg, CV_32FC3, 1.0 / 255.0);

        // Copy vào Buffer (CHW)
        const size_t plane = static_cast<size_t>(inW_) * inH_;
        const float* src   = fimg.ptr<float>();
        for (size_t i = 0; i < plane; ++i) {
            inBuf_[i]             = src[i * 3 + 0];
            inBuf_[i + plane]     = src[i * 3 + 1];
            inBuf_[i + 2 * plane] = src[i * 3 + 2];
        }

        // 2. Inference
        if (!session_.Run(inBuf_, outBuf_)) return false;

        // 3. Hậu xử lý
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;

        for (int i = 0; i < numAnchors_; ++i) {
            float max_conf = 0.f;
            int best_cls = -1;
            
            for (int c = 4; c < numChannels_; ++c) {
                float conf = outBuf_[c * strideC_ + i * strideA_];
                if (conf > max_conf) {
                    max_conf = conf;
                    best_cls = c - 4;
                }
            }

            if (max_conf > cfg_.conf_threshold) {
                float cx = outBuf_[0 * strideC_ + i * strideA_];
                float cy = outBuf_[1 * strideC_ + i * strideA_];
                float w  = outBuf_[2 * strideC_ + i * strideA_];
                float h  = outBuf_[3 * strideC_ + i * strideA_];

                // Loại bỏ padding
                float x = (cx - w / 2.f - left) / r;
                float y = (cy - h / 2.f - top) / r;
                float width = w / r;
                float height = h / r;

                x = std::max(0.0f, x);
                y = std::max(0.0f, y);
                width = std::min(width, static_cast<float>(bgrFrame.cols) - x);
                height = std::min(height, static_cast<float>(bgrFrame.rows) - y);

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
            if (cls == 0) cls = 6; // Map construction_ahead sang turn_left
            detections.push_back({cls, confidences[idx], boxes[idx]});
        }
        return true;
    }

private:
    TrtSession session_;
    Config cfg_;
    int inW_ = 0, inH_ = 0;
    int numChannels_ = 0, numAnchors_ = 0;
    int strideC_ = 1, strideA_ = 1;
    std::vector<float> inBuf_, outBuf_;
};

} // namespace perception
} // namespace ucr

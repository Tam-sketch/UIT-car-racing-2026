// ============================================================
// perception/SignDetector.h
//
// YOLOv8/v5-style detection head.
//   Input : (1, 3, 416, 416)  NCHW  RGB /255
//   Output: (1, 4+8, 3549)    pixel coords, sigmoid'd class scores
//   NMS   : class-agnostic, done here (engine has no NMS baked in)
//
// Class indices (8):
//   0 construction_ahead, 1 no_left, 2 no_right, 3 parking,
//   4 roundabout, 5 traffic_light, 6 turn_left, 7 turn_right
//
// Preprocessing is letterboxed (preserves aspect, zero-pads to square).
// Coordinates are unletterboxed back to the original image frame.
// ============================================================
#pragma once

#include "perception/TrtSession.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace ucr {
namespace perception {

class SignDetector {
public:
    struct Detection {
        cv::Rect2f bbox;       // in original-frame pixel coords
        int        class_id;
        float      confidence;
    };

    struct Config {
        float conf_thresh = 0.5f;
        float iou_thresh  = 0.45f;
        int   input_size  = 416;
        // If true (YOLO standard), coords are in input-pixel units; else normalized 0-1.
        bool  coords_in_pixels = true;
    };

    bool Init(const std::string& enginePath) { return Init(enginePath, Config{}); }

    bool Init(const std::string& enginePath, const Config& cfg) {
        cfg_ = cfg;
        if (!session_.Init(enginePath)) return false;

        if (session_.NumInputs() < 1 || session_.NumOutputs() < 1) {
            std::cerr << "[SignDetector] engine has no I/O\n";
            return false;
        }
        const auto& inB  = session_.InputBinding(0);
        const auto& outB = session_.OutputBinding(0);

        if (inB.dims.nbDims != 4 || inB.dims.d[1] != 3) {
            std::cerr << "[SignDetector] expected NCHW input with 3 ch\n";
            return false;
        }
        if (outB.dims.nbDims != 3) {
            std::cerr << "[SignDetector] expected 3D output [1,C,N]\n";
            return false;
        }
        inH_  = inB.dims.d[2];
        inW_  = inB.dims.d[3];
        outC_ = outB.dims.d[1];   // 12 expected
        outN_ = outB.dims.d[2];   // 3549 expected

        if (inH_ != cfg_.input_size || inW_ != cfg_.input_size) {
            std::cerr << "[SignDetector] engine input " << inW_ << "x" << inH_
                      << " != requested " << cfg_.input_size << "\n";
            return false;
        }
        if (outC_ < 5 || outC_ > 32) {
            std::cerr << "[SignDetector] unexpected output C=" << outC_ << "\n";
            return false;
        }
        num_classes_ = outC_ - 4;

        inBuf_.resize(static_cast<size_t>(3) * inW_ * inH_);
        outBuf_.resize(static_cast<size_t>(outC_) * outN_);

        std::cout << "[SignDetector] in " << inW_ << "x" << inH_
                  << " out (" << outC_ << "," << outN_ << ") classes="
                  << num_classes_ << "\n";
        return true;
    }

    bool IsReady() const { return session_.IsReady(); }
    int  NumClasses() const { return num_classes_; }

    // Populates `out`. Clears `out` on entry. Returns false on inference error.
    bool Detect(const cv::Mat& bgr, std::vector<Detection>& out) {
        out.clear();
        if (!IsReady() || bgr.empty()) return false;

        // --- Letterbox to square ---
        const int W = bgr.cols, H = bgr.rows;
        const float scale = std::min((float)inW_ / W, (float)inH_ / H);
        const int   newW  = (int)std::lround(W * scale);
        const int   newH  = (int)std::lround(H * scale);
        const int   padX  = (inW_ - newW) / 2;
        const int   padY  = (inH_ - newH) / 2;

        cv::Mat canvas(inH_, inW_, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::Mat resized;
        cv::resize(bgr, resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);
        resized.copyTo(canvas(cv::Rect(padX, padY, newW, newH)));

        // BGR→RGB, /255, NCHW
        cv::Mat rgb;
        cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
        cv::Mat f;
        rgb.convertTo(f, CV_32FC3, 1.0 / 255.0);

        const size_t plane = static_cast<size_t>(inW_) * inH_;
        const float* src   = f.ptr<float>();
        for (size_t i = 0; i < plane; ++i) {
            inBuf_[i]             = src[i * 3 + 0];
            inBuf_[i + plane]     = src[i * 3 + 1];
            inBuf_[i + 2 * plane] = src[i * 3 + 2];
        }

        if (!session_.Run(inBuf_, outBuf_)) return false;

        // --- Parse raw detections ---
        std::vector<Detection> raw;
        raw.reserve(64);
        const float* d = outBuf_.data();

        for (int i = 0; i < outN_; ++i) {
            // Bbox: outBuf_[c*outN_ + i] for c in {0,1,2,3}
            float cx = d[0 * outN_ + i];
            float cy = d[1 * outN_ + i];
            float bw = d[2 * outN_ + i];
            float bh = d[3 * outN_ + i];
            if (!cfg_.coords_in_pixels) {
                cx *= inW_; cy *= inH_; bw *= inW_; bh *= inH_;
            }
            // Class scores
            int bestCls = -1;
            float bestSc = 0.f;
            for (int c = 0; c < num_classes_; ++c) {
                const float s = d[(4 + c) * outN_ + i];
                if (s > bestSc) { bestSc = s; bestCls = c; }
            }
            if (bestSc < cfg_.conf_thresh) continue;

            // Unletterbox
            float x1 = (cx - bw * 0.5f - padX) / scale;
            float y1 = (cy - bh * 0.5f - padY) / scale;
            float x2 = (cx + bw * 0.5f - padX) / scale;
            float y2 = (cy + bh * 0.5f - padY) / scale;
            x1 = std::clamp(x1, 0.0f, (float)W);
            y1 = std::clamp(y1, 0.0f, (float)H);
            x2 = std::clamp(x2, 0.0f, (float)W);
            y2 = std::clamp(y2, 0.0f, (float)H);
            if (x2 <= x1 || y2 <= y1) continue;

            raw.push_back({ cv::Rect2f(x1, y1, x2 - x1, y2 - y1), bestCls, bestSc });
        }

        // --- Class-agnostic NMS ---
        NMS(raw, cfg_.iou_thresh, out);
        return true;
    }

private:
    static float IoU(const cv::Rect2f& a, const cv::Rect2f& b) {
        const float x1 = std::max(a.x, b.x);
        const float y1 = std::max(a.y, b.y);
        const float x2 = std::min(a.x + a.width,  b.x + b.width);
        const float y2 = std::min(a.y + a.height, b.y + b.height);
        const float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
        const float uni = a.width * a.height + b.width * b.height - inter;
        return (uni > 0.f) ? inter / uni : 0.f;
    }

    static void NMS(std::vector<Detection>& dets, float iouThr,
                    std::vector<Detection>& keep) {
        if (dets.empty()) return;
        std::vector<int> ord(dets.size());
        std::iota(ord.begin(), ord.end(), 0);
        std::sort(ord.begin(), ord.end(), [&](int a, int b) {
            return dets[a].confidence > dets[b].confidence;
        });
        std::vector<char> gone(dets.size(), 0);
        for (size_t i = 0; i < ord.size(); ++i) {
            const int a = ord[i];
            if (gone[a]) continue;
            keep.push_back(dets[a]);
            for (size_t j = i + 1; j < ord.size(); ++j) {
                const int b = ord[j];
                if (gone[b]) continue;
                if (IoU(dets[a].bbox, dets[b].bbox) > iouThr) gone[b] = 1;
            }
        }
    }

    TrtSession         session_;
    Config             cfg_;
    int                inW_ = 0, inH_ = 0;
    int                outC_ = 0, outN_ = 0, num_classes_ = 0;
    std::vector<float> inBuf_, outBuf_;
};

}  // namespace perception
}  // namespace ucr
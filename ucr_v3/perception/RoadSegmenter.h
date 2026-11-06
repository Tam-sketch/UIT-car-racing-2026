// ============================================================
// perception/RoadSegmenter.h
//
// TwinLiteNet-compatible road segmentation wrapper.
//   - Owns a TrtSession
//   - BGR uint8 → RGB float32 /255 → NCHW float32
//   - Runs inference
//   - 2-channel argmax with confidence margin → binary mask
//
// Output mask is CV_8UC1 at the model's OUTPUT resolution
// (usually 180×320 for TwinLiteNet). Callers that need to
// visualize at camera resolution should resize themselves.
//
// Assumes NCHW I/O: input (1, 3, H, W), output (1, 2, H, W).
// For a different model (e.g. sigmoid 1-channel road mask),
// add a new Config field or a sibling class.
// ============================================================
#pragma once

#include "perception/TrtSession.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <string>
#include <vector>

namespace ucr {
namespace perception {

class RoadSegmenter {
public:
    struct Config {
        // 0: DA is channel 0 (unusual). 1: DA is channel 1 (TwinLiteNet standard).
        int   daChannel        = 1;
        // Pixels with (DA - BG) below this margin are classified background.
        // Protects against FP16 boundary flicker.
        float confidenceMargin = 0.02f;
    };

    bool Init(const std::string& enginePath, const Config& cfg = {}) {
        cfg_ = cfg;

        if (!session_.Init(enginePath)) return false;

        if (session_.NumInputs() < 1 || session_.NumOutputs() < 1) {
            std::cerr << "[RoadSegmenter] Engine has no inputs or outputs\n";
            return false;
        }

        const auto& inB  = session_.InputBinding(0);
        const auto& outB = session_.OutputBinding(0);

        if (inB.dims.nbDims != 4 || outB.dims.nbDims != 4) {
            std::cerr << "[RoadSegmenter] Expected 4D NCHW bindings\n";
            return false;
        }
        if (inB.dims.d[1] != 3) {
            std::cerr << "[RoadSegmenter] Expected 3 input channels, got "
                      << inB.dims.d[1] << "\n";
            return false;
        }
        if (outB.dims.d[1] != 2) {
            std::cerr << "[RoadSegmenter] Expected 2 output channels, got "
                      << outB.dims.d[1] << "\n";
            return false;
        }

        inH_  = inB.dims.d[2];
        inW_  = inB.dims.d[3];
        outH_ = outB.dims.d[2];
        outW_ = outB.dims.d[3];

        inBuf_.resize(static_cast<size_t>(3) * inW_ * inH_);
        outBuf_.resize(static_cast<size_t>(2) * outW_ * outH_);

        std::cout << "[RoadSegmenter] in " << inW_ << "x" << inH_
                  << " out " << outW_ << "x" << outH_ << "\n";
        return true;
    }

    bool IsReady() const { return session_.IsReady(); }

    int InputWidth()   const { return inW_; }
    int InputHeight()  const { return inH_; }
    int OutputWidth()  const { return outW_; }
    int OutputHeight() const { return outH_; }

    // Returns mask at model output resolution (CV_8UC1, 0 or 255).
    bool Predict(const cv::Mat& bgrFrame, cv::Mat& maskOut) {
        if (!IsReady() || bgrFrame.empty()) return false;
        try {
            cv::Mat resized;
            if (bgrFrame.cols != inW_ || bgrFrame.rows != inH_)
                cv::resize(bgrFrame, resized, cv::Size(inW_, inH_), 0, 0, cv::INTER_LINEAR);
            else
                resized = bgrFrame;

            cv::Mat rgb;
            cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
            cv::Mat fimg;
            rgb.convertTo(fimg, CV_32FC3, 1.0 / 255.0);

            const size_t plane = static_cast<size_t>(inW_) * inH_;
            const float* src   = fimg.ptr<float>();
            for (size_t i = 0; i < plane; ++i) {
                inBuf_[i]             = src[i * 3 + 0];   // R
                inBuf_[i + plane]     = src[i * 3 + 1];   // G
                inBuf_[i + 2 * plane] = src[i * 3 + 2];   // B
            }

            if (!session_.Run(inBuf_, outBuf_)) return false;

            const size_t outPlane = static_cast<size_t>(outW_) * outH_;
            const float* ch0 = outBuf_.data();
            const float* ch1 = outBuf_.data() + outPlane;
            const float* da  = (cfg_.daChannel == 1) ? ch1 : ch0;
            const float* bg  = (cfg_.daChannel == 1) ? ch0 : ch1;

            maskOut.create(outH_, outW_, CV_8UC1);
            for (int y = 0; y < outH_; ++y) {
                uchar*       row  = maskOut.ptr<uchar>(y);
                const float* rda  = da + y * outW_;
                const float* rbg  = bg + y * outW_;
                for (int x = 0; x < outW_; ++x) {
                    row[x] = (rda[x] - rbg[x] > cfg_.confidenceMargin) ? 255 : 0;
                }
            }
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[RoadSegmenter] Predict error: " << e.what() << "\n";
            maskOut.release();
            return false;
        }
    }

private:
    TrtSession         session_;
    Config             cfg_;
    int                inW_ = 0, inH_ = 0;
    int                outW_ = 0, outH_ = 0;
    std::vector<float> inBuf_, outBuf_;
};

}  // namespace perception
}  // namespace ucr
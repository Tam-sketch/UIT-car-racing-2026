// ============================================================
// perception/SignDetector.h
//
// 8-class YOLO sign detector on top of TrtSession. Header-only,
// matching the RoadSegmenter.h pattern.
//
// Layout is auto-detected once at Init() from the output binding
// dims, so the same code handles:
//   [1, 4+nc, A]  → channels-first (Ultralytics raw export)
//   [1, A, 4+nc]  → anchors-first (rare)
//   [1, N, 6]     → NMS baked in   (Ultralytics nms=True export)
//
// Class ids come from SignClass.h — the enum order there MUST
// match the model's training names dict or every decision is wrong.
// ============================================================
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "perception/SignClass.h"
#include "perception/TrtSession.h"

namespace ucr {
namespace perception {

// ------------------------------------------------------------
// Free helpers (inline — header-only)
// ------------------------------------------------------------

inline float SignIoU(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float x1 = std::max(a.x, b.x);
    const float y1 = std::max(a.y, b.y);
    const float x2 = std::min(a.x + a.width,  b.x + b.width);
    const float y2 = std::min(a.y + a.height, b.y + b.height);
    const float inter = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
    const float uni   = a.width * a.height + b.width * b.height - inter;
    return uni > 0.f ? inter / uni : 0.f;
}

inline void SignNMS(std::vector<struct SignDetection>& dets, float iou_thresh);

// ------------------------------------------------------------
// Detection record
// ------------------------------------------------------------

struct SignDetection {
    int        class_id;    // SIGN_* from SignClass.h
    float      confidence;  // 0..1
    cv::Rect2f bbox;        // in ORIGINAL frame coordinates
};

// NMS needs SignDetection defined — declare after the struct.
inline void SignNMS(std::vector<SignDetection>& dets, float iou_thresh) {
    std::sort(dets.begin(), dets.end(),
              [](const SignDetection& a, const SignDetection& b) {
                  return a.confidence > b.confidence;
              });
    std::vector<SignDetection> kept;
    kept.reserve(dets.size());
    for (std::size_t i = 0; i < dets.size(); ++i) {
        bool suppressed = false;
        for (std::size_t k = 0; k < kept.size(); ++k) {
            if (kept[k].class_id == dets[i].class_id &&
                SignIoU(kept[k].bbox, dets[i].bbox) > iou_thresh) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) kept.push_back(dets[i]);
    }
    dets.swap(kept);
}

// ------------------------------------------------------------
// SignDetector
// ------------------------------------------------------------

class SignDetector {
public:
    struct Config {
        float conf_thresh = 0.60f;
        float iou_thresh  = 0.45f;
        int   input_size  = 416;   // must match the exported engine
    };

    SignDetector() = default;
    ~SignDetector() = default;
    SignDetector(const SignDetector&)            = delete;
    SignDetector& operator=(const SignDetector&) = delete;

    // GCC 7.5 safe: overload split, no `= {}` default on nested Config.
    bool Init(const std::string& engine_path) {
        return Init(engine_path, Config());
    }

    bool Init(const std::string& engine_path, const Config& cfg) {
        cfg_ = cfg;
        initialized_ = false;

        if (!session_.Init(engine_path)) {
            std::cerr << "[SignDetector] TrtSession::Init failed for: "
                      << engine_path << "\n";
            return false;
        }

        if (!DetectLayout()) {
            std::cerr << "[SignDetector] Cannot determine output layout; aborting\n";
            session_.Shutdown();
            return false;
        }

        const auto& ib = session_.InputBinding(0);
        const std::size_t expected_input =
            static_cast<std::size_t>(3) * cfg_.input_size * cfg_.input_size;

        if (ib.elemCount != expected_input) {
            std::cerr << "[SignDetector] Input elem count mismatch: engine has "
                      << ib.elemCount << " but SIGN_INPUT_SIZE="
                      << cfg_.input_size << " implies " << expected_input
                      << ". Adjust SIGN_INPUT_SIZE in RobotConfig.h.\n";
            session_.Shutdown();
            return false;
        }
        if (ib.dims.nbDims != 4 ||
            ib.dims.d[0] != 1 || ib.dims.d[1] != 3 ||
            ib.dims.d[2] != cfg_.input_size || ib.dims.d[3] != cfg_.input_size) {
            std::cerr << "[SignDetector] Unexpected input dims (expected [1,3,"
                      << cfg_.input_size << "," << cfg_.input_size << "])\n";
            session_.Shutdown();
            return false;
        }

        input_buf_.assign(expected_input, 0.0f);
        initialized_ = true;

        std::cout << "[SignDetector] Ready: engine=" << engine_path
                  << "  input=" << cfg_.input_size << "x" << cfg_.input_size
                  << "  nc=" << num_classes_
                  << "  anchors=" << num_anchors_ << "\n";
        return true;
    }

    bool IsReady() const { return initialized_; }

    // Fills `out` with detections in ORIGINAL frame coords.
    // Returns false ONLY on backend failure. Returns true with empty out
    // when no detection exceeded conf_thresh.
    bool Detect(const cv::Mat& bgr, std::vector<SignDetection>& out) {
        out.clear();
        if (!initialized_ || bgr.empty()) return false;

        float scale = 1.0f;
        int   padx = 0, pady = 0;
        Preprocess(bgr, scale, padx, pady);

        if (!session_.Run(input_buf_, output_buf_)) return false;
        if (output_buf_.empty()) return false;

        const float* p = output_buf_.data();
        switch (layout_) {
            case Layout::CHANNELS_FIRST: ParseChannelsFirst(p, scale, padx, pady, out); break;
            case Layout::ANCHORS_FIRST:  ParseAnchorsFirst (p, scale, padx, pady, out); break;
            case Layout::NMS_BAKED:      ParseNmsBaked     (p, scale, padx, pady, out); break;
        }
        return true;
    }

private:
    enum class Layout { CHANNELS_FIRST, ANCHORS_FIRST, NMS_BAKED };

    bool DetectLayout() {
        if (session_.NumOutputs() < 1) return false;

        const auto& od = session_.OutputBinding(0).dims;

        int d1 = -1, d2 = -1;
        if (od.nbDims == 3) {
            if (od.d[0] != 1) {
                std::cerr << "[SignDetector] batch dim != 1\n";
                return false;
            }
            d1 = od.d[1];
            d2 = od.d[2];
        } else if (od.nbDims == 2) {
            d1 = od.d[0];
            d2 = od.d[1];
        } else {
            std::cerr << "[SignDetector] unexpected output rank: "
                      << od.nbDims << "\n";
            return false;
        }

        std::cout << "[SignDetector] Output dims: [" << d1 << ", " << d2 << "]\n";

        if (d2 == 6 || d1 == 6) {
            layout_      = Layout::NMS_BAKED;
            num_anchors_ = (d2 == 6) ? d1 : d2;
            num_classes_ = SIGN_CLASS_COUNT;
            std::cout << "[SignDetector] Layout: NMS_BAKED, N="
                      << num_anchors_ << "\n";
        } else if (d1 <= 20) {
            layout_      = Layout::CHANNELS_FIRST;
            num_classes_ = d1 - 4;
            num_anchors_ = d2;
            std::cout << "[SignDetector] Layout: CHANNELS_FIRST, nc="
                      << num_classes_ << " A=" << num_anchors_ << "\n";
        } else if (d2 <= 20) {
            layout_      = Layout::ANCHORS_FIRST;
            num_classes_ = d2 - 4;
            num_anchors_ = d1;
            std::cout << "[SignDetector] Layout: ANCHORS_FIRST, nc="
                      << num_classes_ << " A=" << num_anchors_ << "\n";
        } else {
            std::cerr << "[SignDetector] cannot classify output layout\n";
            return false;
        }

        if (num_classes_ != SIGN_CLASS_COUNT) {
            std::cerr << "[SignDetector] WARNING: model has " << num_classes_
                      << " classes, SignClass enum has " << SIGN_CLASS_COUNT
                      << ". Class ids will be mis-mapped unless the training "
                         "names dict matches the enum order.\n";
        }
        return true;
    }

    void Preprocess(const cv::Mat& bgr,
                    float& scale_out, int& padx_out, int& pady_out) {
        const int target = cfg_.input_size;
        const int src_w  = bgr.cols;
        const int src_h  = bgr.rows;

        const float s     = std::min(static_cast<float>(target) / src_w,
                                     static_cast<float>(target) / src_h);
        const int   new_w = static_cast<int>(std::round(src_w * s));
        const int   new_h = static_cast<int>(std::round(src_h * s));
        const int   padx  = (target - new_w) / 2;
        const int   pady  = (target - new_h) / 2;

        scale_out = s;
        padx_out  = padx;
        pady_out  = pady;

        cv::Mat resized;
        cv::resize(bgr, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

        cv::Mat lb(target, target, CV_8UC3, cv::Scalar(114, 114, 114));
        resized.copyTo(lb(cv::Rect(padx, pady, new_w, new_h)));

        cv::Mat rgb;
        cv::cvtColor(lb, rgb, cv::COLOR_BGR2RGB);

        const float inv255 = 1.0f / 255.0f;
        const int   hw     = target * target;
        for (int y = 0; y < target; ++y) {
            const cv::Vec3b* row = rgb.ptr<cv::Vec3b>(y);
            for (int x = 0; x < target; ++x) {
                const cv::Vec3b& p = row[x];
                input_buf_[0 * hw + y * target + x] = p[0] * inv255;
                input_buf_[1 * hw + y * target + x] = p[1] * inv255;
                input_buf_[2 * hw + y * target + x] = p[2] * inv255;
            }
        }
    }

    void ParseChannelsFirst(const float* p, float scale, int padx, int pady,
                            std::vector<SignDetection>& out) {
        const int A  = num_anchors_;
        const int nc = num_classes_;
        for (int a = 0; a < A; ++a) {
            int   best_cls = -1;
            float best     = 0.f;
            for (int c = 0; c < nc; ++c) {
                const float s = p[(4 + c) * A + a];
                if (s > best) { best = s; best_cls = c; }
            }
            if (best < cfg_.conf_thresh || best_cls < 0) continue;

            const float cx = p[0 * A + a];
            const float cy = p[1 * A + a];
            const float w  = p[2 * A + a];
            const float h  = p[3 * A + a];

            SignDetection d;
            d.class_id   = best_cls;
            d.confidence = best;
            d.bbox = cv::Rect2f(
                (cx - w * 0.5f - padx) / scale,
                (cy - h * 0.5f - pady) / scale,
                w / scale,
                h / scale);
            out.push_back(d);
        }
        SignNMS(out, cfg_.iou_thresh);
    }

    void ParseAnchorsFirst(const float* p, float scale, int padx, int pady,
                           std::vector<SignDetection>& out) {
        const int A  = num_anchors_;
        const int nc = num_classes_;
        for (int a = 0; a < A; ++a) {
            const float* row = p + static_cast<std::size_t>(a) * (4 + nc);
            int   best_cls = -1;
            float best     = 0.f;
            for (int c = 0; c < nc; ++c) {
                if (row[4 + c] > best) { best = row[4 + c]; best_cls = c; }
            }
            if (best < cfg_.conf_thresh || best_cls < 0) continue;

            const float cx = row[0];
            const float cy = row[1];
            const float w  = row[2];
            const float h  = row[3];

            SignDetection d;
            d.class_id   = best_cls;
            d.confidence = best;
            d.bbox = cv::Rect2f(
                (cx - w * 0.5f - padx) / scale,
                (cy - h * 0.5f - pady) / scale,
                w / scale,
                h / scale);
            out.push_back(d);
        }
        SignNMS(out, cfg_.iou_thresh);
    }

    void ParseNmsBaked(const float* p, float scale, int padx, int pady,
                       std::vector<SignDetection>& out) {
        const int N = num_anchors_;
        for (int i = 0; i < N; ++i) {
            const float* r = p + static_cast<std::size_t>(i) * 6;
            if (r[4] < cfg_.conf_thresh) continue;

            const float x1 = (r[0] - padx) / scale;
            const float y1 = (r[1] - pady) / scale;
            const float x2 = (r[2] - padx) / scale;
            const float y2 = (r[3] - pady) / scale;

            SignDetection d;
            d.class_id   = static_cast<int>(r[5]);
            d.confidence = r[4];
            d.bbox       = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
            out.push_back(d);
        }
    }

    TrtSession   session_;
    Config       cfg_;
    bool         initialized_ = false;
    Layout       layout_      = Layout::CHANNELS_FIRST;
    int          num_anchors_ = 0;
    int          num_classes_ = 0;
    std::vector<float> input_buf_;
    std::vector<float> output_buf_;
};

}  // namespace perception
}  // namespace ucr
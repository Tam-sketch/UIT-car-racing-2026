#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "perception/TrtSession.h"

namespace ucr {
namespace perception {

// Class order MUST match the training names dict, or the enum indices
// will silently point at the wrong detections.
enum SignClass : int {
    SIGN_CONSTRUCTION_AHEAD = 0,
    SIGN_NO_LEFT            = 1,
    SIGN_NO_RIGHT           = 2,
    SIGN_PARKING            = 3,
    SIGN_ROUNDABOUT         = 4,
    SIGN_TRAFFIC_LIGHT      = 5,
    SIGN_TURN_LEFT          = 6,
    SIGN_TURN_RIGHT         = 7,
    SIGN_CLASS_COUNT        = 8,
};

inline const char* SignClassName(int id) {
    static const char* names[SIGN_CLASS_COUNT] = {
        "construction_ahead", "no_left", "no_right", "parking",
        "roundabout", "traffic_light", "turn_left", "turn_right"
    };
    return (id >= 0 && id < SIGN_CLASS_COUNT) ? names[id] : "unknown";
}

struct SignDetection {
    int        class_id;   // SIGN_*
    float      confidence;
    cv::Rect2f bbox;       // in ORIGINAL frame coordinates
};

class SignDetector {
public:
    struct Config {
        float conf_thresh = 0.60f;
        float iou_thresh  = 0.45f;
        int   input_size  = 416;   // square letterbox target; engine was exported at 416
    };

    SignDetector() = default;
    ~SignDetector() = default;
    SignDetector(const SignDetector&)            = delete;
    SignDetector& operator=(const SignDetector&) = delete;

    // GCC 7.5 safe: overload split (no `= {}` default arg on nested Config)
    bool Init(const std::string& engine_path);
    bool Init(const std::string& engine_path, const Config& cfg);

    // Fills `out` with detections in the ORIGINAL frame's pixel space.
    // Returns false ONLY on backend failure (engine not ready, CUDA error,
    // shape mismatch). Returns true with an empty vector when nothing
    // exceeded `conf_thresh`.
    bool Detect(const cv::Mat& bgr, std::vector<SignDetection>& out);

    bool IsReady() const { return initialized_; }

private:
    enum class Layout { CHANNELS_FIRST, ANCHORS_FIRST, NMS_BAKED };

    bool DetectLayout();
    void Preprocess(const cv::Mat& bgr, float& scale_out, int& padx_out, int& pady_out);

    void ParseChannelsFirst(const float* p, float scale, int padx, int pady,
                            std::vector<SignDetection>& out);
    void ParseAnchorsFirst (const float* p, float scale, int padx, int pady,
                            std::vector<SignDetection>& out);
    void ParseNmsBaked     (const float* p, float scale, int padx, int pady,
                            std::vector<SignDetection>& out);

    TrtSession  session_;
    Config      cfg_;
    bool        initialized_ = false;

    Layout layout_      = Layout::CHANNELS_FIRST;
    int    num_anchors_ = 0;
    int    num_classes_ = 0;

    std::vector<float> input_buf_;   // NCHW FP32, reused each call
    std::vector<float> output_buf_;  // reused each call
};

}  // namespace perception
}  // namespace ucr
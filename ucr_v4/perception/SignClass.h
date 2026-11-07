#pragma once
// Lightweight sign class table.
// Included by SignDetector (produces them) and TurnStateMachine (consumes them).
// No TensorRT, no OpenCV, no external deps — so a control-side consumer
// never has to link against NvInfer just to compare an int.

namespace ucr {
namespace perception {

// Class indices MUST match the training `names` dict of huy_sign.engine.
// If the model was exported with a different ordering, swap these
// constants here — do NOT swap them at the consumer sites.
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

}  // namespace perception
}  // namespace ucr
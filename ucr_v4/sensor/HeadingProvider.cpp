// sensor/HeadingProvider.cpp
#include "sensor/HeadingProvider.h"
#include <cmath>

namespace ucr { namespace sensor {

static float wrap360(float v) {
    v = std::fmod(v, 360.0f);
    if (v < 0.0f) v += 360.0f;
    return v;
}

void HeadingProvider::ResetBaseline(float deg) {
    wrap_base_ = wrap360(deg);
}

float HeadingProvider::Compute(const BNO055& bno,
                               const BNO055::Data& data,
                               double /*dt_s*/) {
    float raw = 0.0f;
    bool  raw_valid = true;

    switch (cfg_.source) {
    case Config::Source::EULER_RAW:
        raw = static_cast<float>(data.eulerDeg.x);
        break;
    case Config::Source::EULER_VEHICLE_YAW:
        raw = 360.0f - static_cast<float>(data.eulerDeg.x);
        break;
    case Config::Source::RELATIVE_YAW: {
        const float y = static_cast<float>(bno.relativeYawDeg());
        // Detect "E38 not fixed": relative yaw frozen across ticks while
        // the state machine is actively asking. If frozen for > 5 calls,
        // we flag invalid so the caller falls back to eulerDeg.x.
        if (rel_seen_ && std::fabs(y - rel_seen_prev_) < 1e-3f) {
            raw_valid = false;  // treat as unavailable
        } else {
            rel_seen_prev_ = y;
            rel_seen_      = true;
        }
        raw = y;
        break;
    }
    }

    if (!raw_valid) return std::nanf("");

    float h = raw * cfg_.sign + cfg_.offset_deg;
    if (cfg_.wrap_to_360) h = wrap360(h);
    last_relative_yaw_ = h;
    return h;
}

}} // namespace
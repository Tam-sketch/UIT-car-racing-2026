// sensor/HeadingProvider.h
#pragma once
#include "sensor/BNO055.h"

namespace ucr { namespace sensor {

class HeadingProvider {
public:
    enum class Source {
        EULER_RAW,          // raw eulerDeg.x (0..360 as output by chip)
        EULER_VEHICLE_YAW,  // 360 - eulerDeg.x  (per test_bno055.cpp convention)
        RELATIVE_YAW,       // integrateYawRate() output; requires E38 wiring
    };

    struct Config {
        Source source            = Source::EULER_RAW;
        float  sign              = 1.0f;   // +1 or -1 — flip per bench test
        float  offset_deg        = 0.0f;   // added after sign, then wrapped
        bool   wrap_to_360       = true;
    };

    HeadingProvider() = default;

    // GCC 7.5 safe: overload split, no `= {}` default on nested Config (E49)
    void Configure(const Config& cfg) { cfg_ = cfg; }

    // Call once per control tick. `dt_s` is only used by RELATIVE_YAW.
    // Returns heading in [0, 360) if wrap_to_360, else raw signed value.
    // Returns NaN if the source is RELATIVE_YAW and `bno.relativeYawDeg()`
    // has not been advancing (i.e. E38 not fixed) — caller must check.
    float Compute(const BNO055& bno, const BNO055::Data& data, double dt_s);

    // Convenience: reset relative yaw to `deg` and zero the internal wrap
    // baseline. Call this when the state machine enters APPROACH if you want
    // every turn to be measured from 0 (not needed with wrapping math, but
    // it makes logs easier to read).
    void ResetBaseline(float deg = 0.0f);

private:
    Config cfg_;
    float  last_relative_yaw_ = 0.0f;
    float  rel_seen_prev_     = 0.0f;
    bool   rel_seen_          = false;
    float  wrap_base_         = 0.0f;
};

}} // namespace
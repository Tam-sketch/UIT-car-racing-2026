#pragma once
#include <cmath>
#include "config/RobotConfig.h"

namespace ucr { namespace control {

class TurnController {
public:
    struct Output { float speed; float angle; bool done; };

    void Start(int dir, float start_heading) {
        dir_ = dir;
        start_heading_ = start_heading;
        active_ = true;
    }

    Output Step(float current_heading) {
        if (!active_) return {0.f, 0.f, true};

        float d = AngleDiff(current_heading, start_heading_);
        float abs_d = std::fabs(d);

        float speed;
        if (abs_d < ucr::config::FORCE_TURN_PHASE_DEG) speed = ucr::config::FORCE_APPROACH_SPEED;
        else if (abs_d < ucr::config::FORCE_EXIT_PHASE_DEG) speed = ucr::config::FORCE_TURNING_SPEED;
        else speed = ucr::config::FORCE_EXIT_SPEED;

        if (abs_d >= ucr::config::HEADING_RESET_DEG) {
            active_ = false;
            return {0.f, 0.f, true};
        }

        return {speed, dir_ * ucr::config::FORCE_ANGLE, false};
    }

private:
    static float AngleDiff(float a, float b) {
        float d = a - b;
        while (d > 180.f)  d -= 360.f;
        while (d < -180.f) d += 360.f;
        return d;
    }
    int   dir_ = 0;
    float start_heading_ = 0.f;
    bool  active_ = false;
};

} } // namespace
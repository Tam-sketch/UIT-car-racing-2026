#include "control/TurnStateMachine.h"

#include <algorithm>
#include <cmath>

#include "perception/SignDetector.h"   // for SIGN_TURN_LEFT / SIGN_TURN_RIGHT

namespace ucr {
namespace control {

TurnStateMachine::TurnStateMachine(SignVoter& voter) : voter_(voter) {}

float TurnStateMachine::WrapDelta(float d) {
    while (d >  180.0f) d -= 360.0f;
    while (d < -180.0f) d += 360.0f;
    return d;
}

int TurnStateMachine::DecideDir(int sign) const {
    if (sign == perception::SIGN_TURN_LEFT)  return -1;
    if (sign == perception::SIGN_TURN_RIGHT) return +1;
    // no_left / no_right / new classes: no turn triggered.
    // (The Round-3 T-junction "must take right because no_left" rule
    //  is not implemented here — deferred with the new-sign work.)
    return 0;
}

void TurnStateMachine::ResetApproach() {
    braking_             = false;
    px_mid_peak_         = -1.0f;
    last_drop_from_peak_ = 0.0f;
}

TurnStateMachine::Output TurnStateMachine::Update(int   sign_committed,
                                                  float heading,
                                                  int   px_mid,
                                                  bool  valid) {
    Output out;   // angle_deg = NaN, speed_cms = NaN, in_turn_ctrl = false

    // 1) EMA of px_mid only while no sign is committed (matches v28)
    if (sign_committed < 0) {
        if (px_mid_ema_ < 0.0f) {
            px_mid_ema_ = static_cast<float>(px_mid);
        } else {
            px_mid_ema_ = TURN_BASELINE_ALPHA * px_mid_ema_
                        + (1.0f - TURN_BASELINE_ALPHA) * static_cast<float>(px_mid);
        }
    }

    // ---------------- NORMAL ----------------
    if (state_ == NORMAL) {
        if (sign_committed >= 0) {
            const int d = DecideDir(sign_committed);
            if (d != 0) {
                turn_dir_       = d;
                sign_at_commit_ = sign_committed;

                const float raw_base = (px_mid_ema_ >= 0.0f)
                                     ? px_mid_ema_
                                     : static_cast<float>(px_mid);
                frozen_baseline_ = std::max(TURN_BASELINE_FLOOR, raw_base);
                brake_thresh_    = frozen_baseline_ * TURN_BRAKE_MULT;
                turn_thresh_     = frozen_baseline_ * TURN_TURN_MULT;

                ResetApproach();
                state_ = APPROACH;

                // TODO (E45 CsvLogger): log NORMAL -> APPROACH
            } else {
                // no_left / no_right / new classes: block + reset,
                // otherwise the same committed sign would re-trigger
                // every frame it stays visible.
                voter_.Block(sign_committed);
                voter_.Reset();
            }
        }
        return out;
    }

    // ---------------- APPROACH ----------------
    if (state_ == APPROACH) {
        if (!valid) return out;   // hold state, don't change actuation

        // track running peak while above brake threshold
        if (px_mid > brake_thresh_) {
            if (px_mid_peak_ < 0.0f || static_cast<float>(px_mid) > px_mid_peak_) {
                px_mid_peak_ = static_cast<float>(px_mid);
            }
        }
        last_drop_from_peak_ = (px_mid_peak_ >= 0.0f)
                             ? (px_mid_peak_ - static_cast<float>(px_mid))
                             : 0.0f;

        // --- v28 trigger: meaningful drop below running peak ---
        if (px_mid_peak_ >= 0.0f
            && px_mid_peak_ > turn_thresh_
            && (px_mid_peak_ - static_cast<float>(px_mid)) >= TURN_PEAK_DROP_PX)
        {
            state_         = TURNING;
            entry_heading_ = heading;
            // TODO (E45): log APPROACH -> TURNING
            return out;
        }

        // --- bail: peak never crossed turn_thresh and px_mid is back
        //     near baseline → false positive, unblock so it can re-arm ---
        if (px_mid_peak_ >= 0.0f
            && px_mid_peak_ <= turn_thresh_
            && static_cast<float>(px_mid) < frozen_baseline_ * TURN_BAIL_FRAC_OF_BASELINE)
        {
            voter_.Block(sign_at_commit_);
            voter_.Reset();

            state_          = NORMAL;
            turn_dir_       = 0;
            sign_at_commit_ = -1;
            px_mid_ema_     = -1.0f;
            ResetApproach();

            // TODO (E45): log APPROACH -> NORMAL (bail)
            return out;
        }

        // --- approach braking ---
        if (px_mid > brake_thresh_) {
            if (!braking_) {
                braking_ = true;
                // TODO (E45): log APPROACH -> BRAKING
            }
            out.speed_cms = TURN_BRAKING_SPEED;
            return out;
        }
        if (braking_) {
            braking_ = false;
            // TODO (E45): log BRAKING -> APPROACH
        }
        return out;
    }

    // ---------------- TURNING ----------------
    if (state_ == TURNING) {
        const float delta    = WrapDelta(heading - entry_heading_);
        const float progress = std::fabs(delta) / std::max(TURN_TARGET_YAW, 1.0f);

        if (progress >= 1.0f) {
            state_     = NORMAL;
            turn_dir_  = 0;

            if (sign_at_commit_ >= 0) {
                voter_.Block(sign_at_commit_);
                voter_.Reset();
            }
            sign_at_commit_ = -1;
            px_mid_ema_     = -1.0f;
            ResetApproach();

            // TODO (E45): log TURNING -> NORMAL
            return out;
        }

        float angle;
        if (progress < TURN_TAPER_START) {
            angle = TURN_SHARP_ANGLE * static_cast<float>(turn_dir_);
        } else {
            const float t = (progress - TURN_TAPER_START)
                          / (1.0f - TURN_TAPER_START);
            angle = TURN_SHARP_ANGLE * static_cast<float>(turn_dir_) * (1.0f - t);
        }
        out.angle_deg    = angle;
        out.speed_cms    = TURN_TURNING_SPEED;
        out.in_turn_ctrl = true;
        return out;
    }

    return out;
}

}  // namespace control
}  // namespace ucr
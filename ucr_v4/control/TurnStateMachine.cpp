#include "control/TurnStateMachine.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "perception/SignClass.h"   // for SIGN_TURN_LEFT / SIGN_TURN_RIGHT

namespace ucr {
namespace control {

TurnStateMachine::TurnStateMachine(SignVoter& voter) : voter_(voter) {}

float TurnStateMachine::WrapDelta(float d) {
    while (d >  180.0f) d -= 360.0f;
    while (d < -180.0f) d += 360.0f;
    return d;
}

int TurnStateMachine::DecideDir(int sign) const {
    if (sign == ucr::perception::SIGN_TURN_LEFT)  return -1;
    if (sign == ucr::perception::SIGN_TURN_RIGHT) return +1;
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
            px_mid_ema_ = ucr::config::TURN_BASELINE_ALPHA * px_mid_ema_
                        + (1.0f - ucr::config::TURN_BASELINE_ALPHA) * static_cast<float>(px_mid);
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
                frozen_baseline_ = std::max(ucr::config::TURN_BASELINE_FLOOR, raw_base);
                brake_thresh_    = frozen_baseline_ * ucr::config::TURN_BRAKE_MULT;
                turn_thresh_     = frozen_baseline_ * ucr::config::TURN_TURN_MULT;

                ResetApproach();
                state_ = APPROACH;

                std::fprintf(stderr,
                    "[TSM] NORMAL->APPROACH sign=%d dir=%+d px_mid=%d base=%.0f brTh=%.0f tnTh=%.0f\n",
                    sign_committed, d, px_mid,
                    frozen_baseline_, brake_thresh_, turn_thresh_);
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
            && (px_mid_peak_ - static_cast<float>(px_mid)) >= ucr::config::TURN_PEAK_DROP_PX)
        {
            state_         = TURNING;
            entry_heading_ = heading;
            std::fprintf(stderr,
                "[TSM] APPROACH->TURNING peak=%.0f px_mid=%d drop=%.0f entry_hdg=%.2f\n",
                px_mid_peak_, px_mid, last_drop_from_peak_, heading);
            return out;
        }

        // --- bail: peak never crossed turn_thresh and px_mid is back
        //     near baseline → false positive, unblock so it can re-arm ---
        if (px_mid_peak_ >= 0.0f
            && px_mid_peak_ <= turn_thresh_
            && static_cast<float>(px_mid) < frozen_baseline_ * ucr::config::TURN_BAIL_FRAC_OF_BASELINE)
        {
            voter_.Block(sign_at_commit_);
            voter_.Reset();

            state_          = NORMAL;
            turn_dir_       = 0;
            sign_at_commit_ = -1;
            px_mid_ema_     = -1.0f;
            ResetApproach();

            std::fprintf(stderr,
                "[TSM] APPROACH->NORMAL (BAIL) peak=%.0f tnTh=%.0f px_mid=%d bailTh=%.0f — "
                "peak was %.0f below trigger\n",
                px_mid_peak_, turn_thresh_, px_mid,
                frozen_baseline_ * ucr::config::TURN_BAIL_FRAC_OF_BASELINE,
                turn_thresh_ - px_mid_peak_);


            return out;
        }

        // --- approach braking ---
        if (px_mid > brake_thresh_) {
            if (!braking_) {
                braking_ = true;
                // TODO (E45): log APPROACH -> BRAKING
            }
            out.speed_cms = ucr::config::TURN_BRAKING_SPEED;
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
        const float progress = std::fabs(delta) / std::max(ucr::config::TURN_TARGET_YAW, 1.0f);

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

            std::fprintf(stderr,
                "[TSM] TURNING->NORMAL delta=%.1f progress=%.2f exit_hdg=%.2f\n",
                delta, progress, heading);
            return out;
        }

        float angle;
        if (progress < ucr::config::TURN_TAPER_START) {
            angle = ucr::config::TURN_SHARP_ANGLE * static_cast<float>(turn_dir_);
        } else {
            const float t = (progress - ucr::config::TURN_TAPER_START)
                          / (1.0f - ucr::config::TURN_TAPER_START);
            angle = ucr::config::TURN_SHARP_ANGLE * static_cast<float>(turn_dir_) * (1.0f - t);
        }
        out.angle_deg    = angle;
        out.speed_cms    = ucr::config::TURN_TURNING_SPEED;
        out.in_turn_ctrl = true;
        return out;
    }

    return out;
}

}  // namespace control
}  // namespace ucr
#pragma once

#include <limits>

#include "control/SignVoter.h"
#include "config/RobotConfig.h"

namespace ucr {
namespace control {

// Port of v28.py TurnStateMachine.
//
// States: NORMAL -> APPROACH -> TURNING -> NORMAL.
// No timers. APPROACH -> TURNING fires on a peak-drop of px_mid; TURNING
// -> NORMAL fires on a measured heading delta >= TURN_TARGET_YAW.
//
// Angle convention: Output::angle_deg is in the SAME sign as the PID
// output (positive = right, matching sim AVControl and v20 out.angle).
// The caller is responsible for the negation to real-left convention
// before calling RealAngleToServo(), exactly as for the PID path.
class TurnStateMachine {
public:
    enum State { NORMAL, APPROACH, TURNING };

    struct Output {
        // NaN when the SM is not controlling steering this tick.
        float angle_deg   = std::numeric_limits<float>::quiet_NaN();
        // NaN when the SM imposes no speed cap this tick.
        float speed_cms   = std::numeric_limits<float>::quiet_NaN();
        // True only during TURNING.
        bool  in_turn_ctrl = false;
    };

    explicit TurnStateMachine(SignVoter& voter);

    // Call once per control tick.
    //
    // sign_committed : SignVoter::GetCommitted(); -1 if none
    // heading_deg    : measured yaw in degrees (0..360 wrapped or unbounded —
    //                  WrapDelta handles both)
    // px_mid         : nonzero pixels in the 35..65% mask band
    // valid          : true when the road mask passed its validity gate
    Output Update(int   sign_committed,
                  float heading_deg,
                  int   px_mid,
                  bool  valid);

    State GetState()            const { return state_; }
    int   GetTurnDir()          const { return turn_dir_; }
    bool  IsBraking()           const { return braking_; }
    float GetFrozenBaseline()   const { return frozen_baseline_; }
    float GetBrakeThresh()      const { return brake_thresh_; }
    float GetTurnThresh()       const { return turn_thresh_; }
    float GetPxMidPeak()        const { return px_mid_peak_; }
    float GetDropFromPeak()     const { return last_drop_from_peak_; }

private:
    static float WrapDelta(float d);
    int   DecideDir(int sign) const;
    void  ResetApproach();

    SignVoter& voter_;

    State state_               = NORMAL;
    int   turn_dir_            = 0;                         // -1 L, +1 R, 0 none
    float entry_heading_       = 0.0f;
    int   sign_at_commit_      = -1;
    float px_mid_ema_          = -1.0f;                     // -1 = uninit
    float frozen_baseline_     = ucr::config::TURN_BASELINE_FLOOR;
    float brake_thresh_        = ucr::config::TURN_BASELINE_FLOOR * ucr::config::TURN_BRAKE_MULT;
    float turn_thresh_         = ucr::config::TURN_BASELINE_FLOOR * ucr::config::TURN_TURN_MULT;
    bool  braking_             = false;
    float px_mid_peak_         = -1.0f;
    float last_drop_from_peak_ = 0.0f;
};

}  // namespace control
}  // namespace ucr
// ============================================================
// control/TurnStateMachine.h
//
// Port of pray.py's force-turn + voter-reset logic.
//
// Clock contract:
//   - UpdateVoter(classIds, ctrlTick) is called ONCE per sign-detect
//     (i.e. every SIGN_DETECT_EVERY control ticks).
//   - Update(ctrlTick, heading, roadMask) is called ONCE per control tick.
//   - SetTopo(topo) is called once per sign-detect, after SkeletonTopology.
//
// Output angle is in v20's out.angle convention (+ = right turn).
// Caller converts to real-angle left-positive by negating, then clamps.
// ============================================================
#pragma once

#include "control/SignVoter.h"
#include "perception/SkeletonTopology.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace ucr {
namespace control {

struct TurnConfig {
    // --- Gate ---
    float turn_condition_y_lo_ratio = 0.69f;
    float turn_condition_coverage   = 0.78f;
    float force_angle               = 22.0f;   // was 25 in sim, 22 fits MAX_REAL_ANGLE=30
    // --- Adaptive force speed (by |Δheading| from force start) ---
    float force_approach_speed   = 8.0f;
    float force_turning_speed    = 5.0f;
    float force_exit_speed       = 12.0f;
    float force_turn_phase_deg   = 15.0f;
    float force_exit_phase_deg   = 50.0f;
    // --- Reset ---
    float heading_reset_deg        = 56.0f;
    int   reset_debounce_ticks     = 150;
    int   sign_snapshot_hold_ticks = 5;
    // --- Speed caps ---
    float slow_speed      = 10.0f;   // was 15 in sim
    float max_speed       = 20.0f;   // matches MAX_SPEED_CMS
    int   sign_hold_ticks = 30;
    
    // Relative widening threshold. cov(far band) must exceed cov(near band)
    // by this fraction for the gate to fire. 0.08 = 8 percentage-point widening.
    // Raise (0.10–0.15) if force-turn fires too early; lower (0.05) if it never fires.
    float turn_opening_margin = 0.08f;

    // Absolute floor on near-band coverage. Protects against mask dropout while
    // the car is off the lane. Raise if false triggers during occlusion.
    float turn_near_min_coverage = 0.55f;

    // --- Class indices (8-class engine) ---
    int idx_construction_ahead = 0;
    int idx_no_left            = 1;
    int idx_no_right           = 2;
    int idx_parking            = 3;
    int idx_roundabout         = 4;
    int idx_traffic_light      = 5;
    int idx_turn_left          = 6;
    int idx_turn_right         = 7;
};

class TurnStateMachine {
public:
    struct Output {
        bool   in_force_turn     = false;
        int    force_dir         = 0;
        float  force_angle_deg   = 0.0f;
        float  force_speed_cms   = 0.0f;
        float  max_speed_cap_cms = 0.0f;
        bool   sign_committed    = false;
        int    sign_id           = -1;
        const char* phase        = "-";
        float  heading   = 0.0f;
        float  d_heading = 0.0f;
        int    topo_L = 0, topo_S = 1, topo_R = 0;
        bool   topo_reset_fired  = false;
    };

    TurnStateMachine(const TurnConfig& cfg,
                     int activateFrames,
                     int voterTimeoutTicks)
        : cfg_(cfg)
        , voter_(activateFrames, voterTimeoutTicks) {}

    // ---- Detect-rate calls (every SIGN_DETECT_EVERY ticks) ----
    void UpdateVoter(const std::vector<int>& classIds, uint32_t ctrlTick) {
        voter_.Update(classIds, ctrlTick);
    }
    void SetTopo(const perception::TopoResult& t) { lastTopo_ = t; }

    // ---- Control-rate call (every control tick) ----
    Output Update(uint32_t ctrlTick, float headingDeg, const cv::Mat& roadMask) {
        Output o;
        o.heading        = headingDeg;
        o.sign_id        = voter_.GetCommitted();
        o.sign_committed = (o.sign_id >= 0);
        o.topo_L = (int)lastTopo_.left;
        o.topo_S = (int)lastTopo_.straight;
        o.topo_R = (int)lastTopo_.right;

        const bool signSeenRecently =
            (voter_.LastSeenTick() >= 0) &&
            (int)ctrlTick - voter_.LastSeenTick() <= cfg_.sign_hold_ticks;

        if (voter_.JustTimedOut()) {
            ResetForce();
            lastCommit_ = -1;
            snapTopoValid_ = false;
        }

        if (o.sign_id >= 0 && o.sign_id != lastCommit_) {
            snapTopo_ = lastTopo_;
            snapTopoValid_ = true;
            lastCommit_ = o.sign_id;
        }

        if (topoReset_.Update(lastTopo_, cfg_.reset_debounce_ticks)) {
            voter_.Reset();
            ResetForce();
            lastCommit_ = -1;
            snapTopoValid_ = false;
            o.topo_reset_fired = true;
        }

        float dHeading = 0.0f;
        if (forceActive_ && forceHeading0Valid_) {
            dHeading = AngleDiffDeg(headingDeg, forceHeading0_);
            if (std::abs(dHeading) >= cfg_.heading_reset_deg) {
                voter_.Reset();
                ResetForce();
                lastCommit_ = -1;
                snapTopoValid_ = false;
                dHeading = 0.0f;
            }
        }
        o.d_heading = dHeading;

        const int framesSinceSeen = (voter_.LastSeenTick() >= 0)
                                  ? (int)ctrlTick - voter_.LastSeenTick()
                                  : (1 << 30);
        const perception::TopoResult topoDecision =
            (o.sign_id >= 0 && snapTopoValid_ &&
             framesSinceSeen >= cfg_.sign_snapshot_hold_ticks)
            ? snapTopo_ : lastTopo_;

        const int forceCandidate = DecideForceDir(o.sign_id, topoDecision);
        if (!forceActive_ && forceCandidate != 0 &&
            CheckTurnCondition(roadMask)) {
            forceActive_       = true;
            forceDir_          = forceCandidate;
            forceHeading0_     = headingDeg;
            forceHeading0Valid_ = true;
            forcePhase_        = "ENTRY";
        }

        if (forceActive_) {
            const float absDh = std::abs(dHeading);
            if (absDh < cfg_.force_turn_phase_deg)      { forcePhase_ = "ENTRY";   o.force_speed_cms = cfg_.force_approach_speed; }
            else if (absDh < cfg_.force_exit_phase_deg) { forcePhase_ = "TURNING"; o.force_speed_cms = cfg_.force_turning_speed; }
            else                                        { forcePhase_ = "EXIT";    o.force_speed_cms = cfg_.force_exit_speed; }

            o.in_force_turn     = true;
            o.force_dir         = forceDir_;
            o.force_angle_deg   = (float)forceDir_ * cfg_.force_angle;
            o.phase             = forcePhase_;
            o.max_speed_cap_cms = cfg_.slow_speed;
        } else {
            o.max_speed_cap_cms =
                (o.sign_committed || signSeenRecently) ? cfg_.slow_speed : cfg_.max_speed;
        }
        return o;
    }

    void Reset() {
        voter_.Reset();
        ResetForce();
        topoReset_.Reset();
        lastCommit_ = -1;
        snapTopoValid_ = false;
    }

private:
    static float AngleDiffDeg(float a, float b) {
        return std::fmod(a - b + 540.0f, 360.0f) - 180.0f;
    }

    bool CheckTurnCondition(const cv::Mat& mask) const {
        if (mask.empty()) return false;
        const int h = mask.rows, w = mask.cols;
        if (h < 10) return false;

        // Three bands, bottom to top (image coords: y=h is bottom/near, y=0 is top/far).
        //   far  band: y ∈ [0.55h, 0.72h)  ← where the junction mouth is visible
        //   near band: y ∈ [0.82h, h)      ← pure lane, always ≈lane width
        //   bottom band (for the absolute safety net) = near band
        const int yFarTop  = (int)(h * 0.55f);
        const int yFarBot  = (int)(h * 0.72f);
        const int yNearTop = (int)(h * 0.82f);
        if (yFarTop >= yFarBot || yFarBot >= yNearTop || yNearTop >= h) return false;

        const int areaFar  = (yFarBot - yFarTop) * w;
        const int areaNear = (h - yNearTop) * w;
        if (areaFar <= 0 || areaNear <= 0) return false;

        const float covFar  = (float)cv::countNonZero(mask.rowRange(yFarTop, yFarBot)) / areaFar;
        const float covNear = (float)cv::countNonZero(mask.rowRange(yNearTop, h))       / areaNear;

        // Gate condition:
        //   (a) the junction band has more road than the near band (relative widening)
        //   (b) the near band is dominated by road (car is actually on the lane;
        //       protects against firing during a mask dropout)
        const float opening = covFar - covNear;
        return (opening >= cfg_.turn_opening_margin) && (covNear >= 0.55f);
    }

    int DecideForceDir(int commit, const perception::TopoResult& topo) const {
        if (commit < 0) return 0;
        if (commit == cfg_.idx_turn_left)  return -1;
        if (commit == cfg_.idx_turn_right) return +1;
        const bool hasS = topo.straight;
        if (commit == cfg_.idx_no_left)  return hasS ? 0 : +1;
        if (commit == cfg_.idx_no_right) return hasS ? 0 : -1;
        return 0;   // construction_ahead / parking / roundabout / traffic_light: handled later
    }

    void ResetForce() {
        forceActive_ = false;
        forceDir_ = 0;
        forceHeading0Valid_ = false;
        forcePhase_ = "-";
    }

    class TopoResetDetector {
    public:
        bool Update(const perception::TopoResult& t, int debounce) {
            const int nbits = (int)t.left + (int)t.straight + (int)t.right;
            if (nbits >= 2) { armed_ = true; stable_ = 0; return false; }
            if (armed_ && nbits == 1) {
                if (++stable_ >= debounce) { armed_ = false; stable_ = 0; return true; }
            } else { stable_ = 0; }
            return false;
        }
        void Reset() { armed_ = false; stable_ = 0; }
    private:
        bool armed_ = false;
        int  stable_ = 0;
    };

    TurnConfig  cfg_;
    SignVoter   voter_;
    TopoResetDetector topoReset_;

    perception::TopoResult lastTopo_{};
    perception::TopoResult snapTopo_{};
    bool snapTopoValid_ = false;
    int  lastCommit_ = -1;

    bool  forceActive_ = false;
    int   forceDir_ = 0;
    float forceHeading0_ = 0.0f;
    bool  forceHeading0Valid_ = false;
    const char* forcePhase_ = "-";
};

}  // namespace control
}  // namespace ucr
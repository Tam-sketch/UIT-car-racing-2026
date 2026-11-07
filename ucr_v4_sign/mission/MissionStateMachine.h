#pragma once
#include "control/LaneFollowerV20.h"
#include "control/SignVoter.h"
#include "control/TurnController.h"
#include "perception/SignDetector.h"
#include "config/RobotConfig.h"
#include <opencv2/opencv.hpp>

namespace ucr {
namespace mission {

enum class State { 
    LANE_FOLLOW, 
    FORCE_TURN 
};

class MissionStateMachine {
public:
    struct Command { 
        float speed; 
        float angle; 
    };

    // Constructor nhận Config của LaneFollower để truyền post_pid_gain = 1.0
    explicit MissionStateMachine(const v20::Config& lane_cfg = v20::Config())
        : lane_follower_(lane_cfg) {}

    Command Update(const cv::Mat& mask, 
                   const std::vector<perception::Detection>& signs,
                   float heading, int frame_idx) {
        
        std::vector<int> cls_ids;
        for (const auto& d : signs) cls_ids.push_back(d.class_id);
        voter_.Update(cls_ids, frame_idx);

        int commit = voter_.Committed();

        if (state_ == State::FORCE_TURN) {
            auto out = turn_ctrl_.Step(heading);
            if (out.done) {
                state_ = State::LANE_FOLLOW;
                voter_.ClearCommitted();
                lane_follower_.softReset(); // Đã có hàm này nhờ patch ở trên
                auto fo = lane_follower_.step(mask);
                return {fo.speed, fo.angle};
            }
            return {out.speed, out.angle};
        }

        if (commit != -1) {
            int dir = DecideForceDir(commit);
            if (dir != 0 && CheckTurnCondition(mask)) {
                state_ = State::FORCE_TURN;
                turn_ctrl_.Start(dir, heading);
                lane_follower_.resetTracking();
                auto out = turn_ctrl_.Step(heading);
                return {out.speed, out.angle};
            }
        }

        auto fo = lane_follower_.step(mask);
        return {fo.speed, fo.angle};
    }

private:
    static int DecideForceDir(int cls) {
        switch (cls) {
            case 6: return -1;  // turn_left
            case 7: return +1;  // turn_right
            case 0: return -1;  // construction_ahead -> left
            case 1: return +1;  // no_left -> right
            case 2: return -1;  // no_right -> left
            default: return 0;  // traffic_light, parking, roundabout
        }
    }

    static bool CheckTurnCondition(const cv::Mat& mask) {
        if (mask.empty()) return false;
        int h = mask.rows, w = mask.cols;
        int y_lo = static_cast<int>(h * config::TURN_CONDITION_Y_LO_RATIO);
        cv::Rect roi(0, y_lo, w, h - y_lo);
        if (roi.area() == 0) return false;
        cv::Mat region = mask(roi);
        int road_px = cv::countNonZero(region);
        return static_cast<float>(road_px) / roi.area() >= config::TURN_CONDITION_COVERAGE;
    }

    State state_ = State::LANE_FOLLOW;
    control::SignVoter      voter_;
    control::TurnController turn_ctrl_;
    v20::LaneFollower       lane_follower_; // Sửa thành v20::
};

} // namespace mission
} // namespace ucr
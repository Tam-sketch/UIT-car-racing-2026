#pragma once
#include "control/LaneFollowerV20.h"
#include "control/TurnController.h"
#include "perception/SignDetector.h"
#include "config/RobotConfig.h"
#include <opencv2/opencv.hpp>
#include <vector>

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

    Command Update(const cv::Mat& mask, 
                   const std::vector<perception::Detection>& signs,
                   float heading, int frame_idx) {
        
        // 1. Tìm biển báo có độ tin cậy cao nhất trong frame hiện tại
        int commit = -1;
        float max_conf = 0.0f;
        for (const auto& d : signs) {
            if (d.confidence > max_conf) {
                max_conf = d.confidence;
                commit = d.class_id;
            }
        }

        // 2. Nếu đang trong quá trình rẽ cưỡng bức (FORCE_TURN)
        if (state_ == State::FORCE_TURN) {
            auto out = turn_ctrl_.Step(heading);
            if (out.done) {
                state_ = State::LANE_FOLLOW;
                lane_follower_.softReset();
                auto fo = lane_follower_.step(mask);
                return {fo.speed, fo.angle};
            }
            return {out.speed, out.angle};
        }

        // 3. Nếu đang bám làn (LANE_FOLLOW), kiểm tra xem có biển báo nào không
        if (commit != -1) {
            int dir = DecideForceDir(commit);
            if (dir != 0 && CheckTurnCondition(mask)) {
                std::cout << "[Mission] Kich hoat FORCE_TURN (Class ID: " 
                          << commit << ", Conf: " << max_conf << ")\n";
                state_ = State::FORCE_TURN;
                turn_ctrl_.Start(dir, heading);
                lane_follower_.resetTracking();
                auto out = turn_ctrl_.Step(heading);
                return {out.speed, out.angle};
            }
        }

        // 4. Mặc định: Bám làn bằng PID
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
    control::TurnController turn_ctrl_;
    v20::LaneFollower       lane_follower_;
};

} // namespace mission
} // namespace ucr
#pragma once
#include <map>
#include <set>
#include <vector>
#include "config/RobotConfig.h"

namespace ucr {
namespace control {

class SignVoter {
public:
    SignVoter(int activate = config::VOTER_ACTIVATE_FRAMES, 
              int timeout  = config::VOTER_TIMEOUT_FRAMES)
        : activate_(activate), timeout_(timeout) {}

    void Update(const std::vector<int>& classes, int frame_idx) {
        std::set<int> classes_set(classes.begin(), classes.end());
        if (!classes_set.empty()) last_seen_frame_ = frame_idx;

        if (last_seen_frame_ >= 0 && (frame_idx - last_seen_frame_) > timeout_) {
            Reset();
            return;
        }

        for (auto it = candidate_cnt_.begin(); it != candidate_cnt_.end(); ) {
            if (classes_set.find(it->first) == classes_set.end())
                it = candidate_cnt_.erase(it);
            else
                ++it;
        }

        for (int cls : classes_set) {
            candidate_cnt_[cls]++;
            if (candidate_cnt_[cls] >= activate_ && activated_.find(cls) == activated_.end()) {
                activated_.insert(cls);
                counts_[cls] = 0;
            }
        }

        for (int cls : classes_set) {
            if (activated_.count(cls)) counts_[cls]++;
        }

        if (!counts_.empty()) {
            int best = -1, best_cnt = -1;
            for (auto& pair : counts_) {
                if (pair.second > best_cnt) { 
                    best = pair.first; 
                    best_cnt = pair.second; 
                }
            }
            if (best != -1) committed_ = best;
        }
    }

    int  Committed() const { return committed_; }
    void ClearCommitted() { committed_ = -1; }
    
    void Reset() {
        counts_.clear(); 
        activated_.clear(); 
        candidate_cnt_.clear();
        committed_ = -1; 
        last_seen_frame_ = -1;
    }

private:
    int activate_, timeout_;
    std::map<int,int> counts_;
    std::set<int>     activated_;
    std::map<int,int> candidate_cnt_;
    int committed_       = -1;
    int last_seen_frame_ = -1;
};

} // namespace control
} // namespace ucr
// ============================================================
// control/SignVoter.h
// Port của pray.py::SignVoter — chống nhiễu single-frame
// ============================================================
#pragma once

#include <algorithm>
#include <map>
#include <set>
#include <vector>

namespace ucr {
namespace control {

class SignVoter {
public:
    SignVoter(int activateFrames, int timeoutFrames)
        : activateFrames_(activateFrames), timeoutFrames_(timeoutFrames) {}

    void Update(const std::vector<int>& classIds, uint32_t tickIdx) {
        justTimedOut_ = false;
        std::set<int> classes(classIds.begin(), classIds.end());
        if (!classes.empty()) lastSeenTick_ = (int)tickIdx;

        if (lastSeenTick_ >= 0 &&
            (int)tickIdx - lastSeenTick_ > timeoutFrames_) {
            Reset();
            justTimedOut_ = true;
            return;
        }
        for (auto& kv : candidateCnt_) if (!classes.count(kv.first)) kv.second = 0;
        for (int c : classes) {
            int& n = candidateCnt_[c];
            ++n;
            if (n >= activateFrames_ && !activated_.count(c)) {
                activated_.insert(c);
                counts_[c] = 0;
            }
        }
        for (int c : classes) if (activated_.count(c)) ++counts_[c];
        if (!counts_.empty()) {
            int best = committed_;
            int bestN = -1;
            for (const auto& kv : counts_) {
                if (kv.second > bestN) { bestN = kv.second; best = kv.first; }
            }
            committed_ = best;
        }
    }

    int  GetCommitted() const { return committed_; }
    bool JustTimedOut() const { return justTimedOut_; }
    int  LastSeenTick() const { return lastSeenTick_; }

    void Reset() {
        counts_.clear();
        activated_.clear();
        candidateCnt_.clear();
        committed_ = -1;
        lastSeenTick_ = -1;
    }

private:
    int activateFrames_, timeoutFrames_;
    std::map<int, int> counts_;
    std::set<int>      activated_;
    std::map<int, int> candidateCnt_;
    int  committed_    = -1;
    int  lastSeenTick_ = -1;
    bool justTimedOut_ = false;
};

}  // namespace control
}  // namespace ucr
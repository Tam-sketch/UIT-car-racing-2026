// ============================================================
// control/SignVoter.h
//
// Port of pray.py::SignVoter.
//   - Classes must be seen `activateFrames` consecutive detect-updates before
//     they enter the count pool.
//   - Once in the pool, per-class `counts` accumulate; the current max wins.
//   - If no sign seen for > `timeoutFrames` ticks, the voter resets and
//     `JustTimedOut()` returns true for one Update cycle.
//
// The caller must pass `tickIdx` in control-tick units, and must ONLY call
// Update() on ticks when a detect actually ran (i.e. every SIGN_DETECT_EVERY).
// This makes `timeoutFrames` a real wall-clock timeout regardless of detect
// rate (100 ticks = 10 s at CONTROL_PERIOD_MS=100).
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

        // Zero candidate counter for any class not in this update.
        for (auto& kv : candidateCnt_) if (!classes.count(kv.first)) kv.second = 0;

        // Increment candidate counters, activate on threshold.
        for (int c : classes) {
            int& n = candidateCnt_[c];
            ++n;
            if (n >= activateFrames_ && !activated_.count(c)) {
                activated_.insert(c);
                counts_[c] = 0;
            }
        }

        // Increment pooled counts for activated classes seen this update.
        for (int c : classes) if (activated_.count(c)) ++counts_[c];

        // Commit = argmax(counts); keep previous on tie.
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
#pragma once

#include <map>
#include <vector>

namespace ucr {
namespace control {

// Port of v28.py SignVoter.
//
// Commits to a class after `min_frames` detections of that class (with
// `max_gap` frame tolerance), and releases the commit after `timeout`
// frames. Block() adds a class to a decaying blacklist so the same sign
// cannot re-arm immediately after a turn completes.
//
// Update() takes the raw per-frame class id list (already sorted by
// confidence descending, if you want the first non-blacklisted to win).
class SignVoter {
public:
    SignVoter(int min_frames        = 4,
              int max_gap           = 3,
              int timeout           = 300,
              int blacklist_frames  = 300);

    // Feed every detection frame (or every Nth — must match the control
    // tick period, since `min_frames` is in detection frames).
    void Update(const std::vector<int>& class_ids);

    // -1 means "nothing committed right now".
    int GetCommitted()  const { return committed_; }

    // Frames since the current commit started (0 when nothing committed).
    int GetCommitAge()  const { return commit_age_; }

    void Block(int cls);
    void Reset();

private:
    int min_frames_;
    int max_gap_;
    int timeout_;
    int blacklist_frames_;

    int current_class_ = -1;
    int count_         = 0;
    int gap_           = 0;
    int committed_     = -1;
    int age_           = 0;
    int commit_age_    = 0;

    std::map<int, int> blacklist_;   // class -> frames remaining
};

}  // namespace control
}  // namespace ucr
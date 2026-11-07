#include "control/SignVoter.h"

namespace ucr {
namespace control {

SignVoter::SignVoter(int min_frames, int max_gap,
                     int timeout, int blacklist_frames)
    : min_frames_(min_frames),
      max_gap_(max_gap),
      timeout_(timeout),
      blacklist_frames_(blacklist_frames) {}

void SignVoter::Block(int cls) {
    if (cls < 0) return;
    blacklist_[cls] = blacklist_frames_;
}

void SignVoter::Reset() {
    current_class_ = -1;
    count_         = 0;
    gap_           = 0;
    committed_     = -1;
    age_           = 0;
    commit_age_    = 0;
}

void SignVoter::Update(const std::vector<int>& class_ids) {
    // 1) age blacklist
    for (std::map<int, int>::iterator it = blacklist_.begin();
         it != blacklist_.end(); ) {
        if (--it->second <= 0) it = blacklist_.erase(it);
        else                   ++it;
    }

    // 2) age committed (before the no-detection early return,
    //    matching v28 exactly)
    if (committed_ >= 0) {
        ++age_;
        ++commit_age_;
        if (age_ >= timeout_) {
            committed_  = -1;
            age_        = 0;
            commit_age_ = 0;
        }
    }

    // 3) filter blacklisted classes; first remaining wins
    int cls = -1;
    for (size_t i = 0; i < class_ids.size(); ++i) {
        if (blacklist_.find(class_ids[i]) == blacklist_.end()) {
            cls = class_ids[i];
            break;
        }
    }

    if (cls < 0) {
        ++gap_;
        if (gap_ > max_gap_) {
            current_class_ = -1;
            count_         = 0;
        }
        return;
    }

    if (cls == current_class_) {
        ++count_;
        gap_ = 0;
    } else {
        current_class_ = cls;
        count_         = 1;
        gap_           = 0;
    }

    if (count_ >= min_frames_ && committed_ != cls) {
        committed_  = cls;
        age_        = 0;
        commit_age_ = 0;
    }
}

}  // namespace control
}  // namespace ucr
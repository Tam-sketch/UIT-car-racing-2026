// ============================================================
// SignVoter.h — Intelligent High-Speed Sign Voting & Arbitration
// ============================================================
//5
#pragma once

#include <algorithm>
#include <map>
#include <set>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <string>

namespace ucr {
namespace control {

struct SignObservation {
    int   class_id   = -1;
    float confidence = 0.0f;
    int   area       = 0;

    SignObservation() = default;
    SignObservation(int id, float conf, int a = 0)
        : class_id(id), confidence(conf), area(a) {}
};

class SignVoter {
public:
    // FAST-PATH: Khi conf cao (>= 0.60f), CHỐT NGAY LẬP TỨC ở frame đầu tiên (0 frame delay)!
    static constexpr float FAST_PATH_CONF       = 0.60f;
    // Ngưỡng conf tối thiểu để bắt đầu theo dõi và tích lũy điểm từ xa
    static constexpr float MIN_VOTE_CONF        = 0.30f;
    // Số frame miss liên tiếp trước khi reset số lần nháy của class đó (Leaky Bucket chống rớt)
    static constexpr int   MAX_TOLERATED_MISSES = 8;

    SignVoter(int activateFrames = 2, int timeoutFrames = 60)
        : activateFrames_(activateFrames), timeoutFrames_(timeoutFrames) {}

    // Cập nhật với danh sách quan sát đầy đủ (class_id + confidence + area)
    void Update(const std::vector<SignObservation>& obsList, uint32_t tickIdx) {
        justTimedOut_ = false;

        std::set<int> seen_classes;
        for (const auto& obs : obsList) {
            if (obs.class_id >= 0 && obs.confidence >= MIN_VOTE_CONF) {
                seen_classes.insert(obs.class_id);
            }
        }

        if (!seen_classes.empty()) {
            lastSeenTick_ = static_cast<int>(tickIdx);
        }

        // Quá timeout dài (mặc định 60 ticks ~ 6.0s) không thấy bất kỳ biển nào -> Reset an toàn
        if (lastSeenTick_ >= 0 && (static_cast<int>(tickIdx) - lastSeenTick_ > timeoutFrames_)) {
            Reset();
            justTimedOut_ = true;
            return;
        }

        // 1. CẬP NHẬT CÁC BIỂN XUẤT HIỆN Ở FRAME HIỆN TẠI
        for (const auto& obs : obsList) {
            if (obs.class_id < 0 || obs.confidence < MIN_VOTE_CONF) continue;
            int c = obs.class_id;

            hitCount_[c]++;
            consecutiveHits_[c]++;
            missCount_[c] = 0;
            if (obs.confidence > maxConf_[c]) {
                maxConf_[c] = obs.confidence;
            }

            // Trọng số cộng dồn điểm theo độ tự tin:
            // Conf >= 0.85: +15.0 điểm (cực mạnh, áp đảo)
            // Conf >= 0.70: +10.0 điểm
            // Conf >= 0.60: +6.0 điểm
            // Conf >= 0.45: +3.0 điểm
            // Conf >= 0.30: +1.5 điểm
            float weight = 1.5f;
            if (obs.confidence >= 0.85f)      weight = 15.0f;
            else if (obs.confidence >= 0.70f) weight = 10.0f;
            else if (obs.confidence >= 0.60f) weight = 6.0f;
            else if (obs.confidence >= 0.45f) weight = 3.0f;

            scores_[c] += weight;

            // --- CƠ CHẾ KÍCH HOẠT (ACTIVATION) ---
            // 1. FAST-PATH: Conf cao (>= 0.60) -> KÍCH HOẠT VÀ CHỐT NGAY LẬP TỨC (0-frame latency)!
            // Dù xuất hiện ở cuối khung hình, chuẩn bị đi hết biển, 1 frame conf cao là ăn ngay!
            if (obs.confidence >= FAST_PATH_CONF) {
                activated_.insert(c);
                if (committed_ < 0 || obs.confidence >= maxConf_[committed_]) {
                    committed_ = c;
                }
            }
            // 2. NHẬN LIÊN TỤC: Nhận 2 frame liên tiếp (dù conf thấp 0.30-0.59) -> Chốt ngay!
            else if (consecutiveHits_[c] >= activateFrames_) {
                activated_.insert(c);
                if (committed_ < 0) {
                    committed_ = c;
                }
            }
            // 3. NHẤP NHÁY / TẦN SUẤT CAO: Nháy nhiều lần (hitCount >= 3) -> Kích hoạt vào so tài!
            else if (hitCount_[c] >= 3) {
                activated_.insert(c);
            }
        }

        // 2. XỬ LÝ BIỂN BỊ MISS Ở FRAME HIỆN TẠI (LEAKY BUCKET)
        // Khi camera rung/chói làm hụt 1-2 frame, KHÔNG reset số lần nháy hay xóa biển!
        // Chỉ khi miss liên tiếp > MAX_TOLERATED_MISSES (8 frames) mới giảm điểm.
        for (auto& kv : hitCount_) {
            int c = kv.first;
            if (!seen_classes.count(c)) {
                consecutiveHits_[c] = 0; // Hụt frame thì reset chuỗi liên tiếp
                missCount_[c]++;
                if (missCount_[c] >= MAX_TOLERATED_MISSES) {
                    kv.second = 0;
                    scores_[c] = std::max(0.0f, scores_[c] - 2.0f);
                }
            }
        }

        // 3. SO SÁNH TRỌNG SỐ & PHÂN ĐỊNH BIỂN CHIẾN THẮNG (ARGMAX)
        // Ưu tiên: Điểm tích lũy + Số lần nháy * 4.0 + Max Conf * 10.0
        if (!activated_.empty()) {
            int best_c = committed_;
            float best_score = -1.0f;

            if (committed_ >= 0 && scores_.count(committed_)) {
                // Biển đang chốt có bias bảo vệ +5.0 điểm để chống bị giật lật bởi 1 frame nhiễu
                best_score = scores_[committed_] + hitCount_[committed_] * 4.0f + maxConf_[committed_] * 10.0f + 5.0f;
            }

            for (int c : activated_) {
                float total = scores_[c] + hitCount_[c] * 4.0f + maxConf_[c] * 10.0f;
                if (total > best_score) {
                    best_score = total;
                    best_c = c;
                }
            }
            committed_ = best_c;
        }
    }

    // Overload khi không có biển nào xuất hiện ở frame hiện tại
    void Update(uint32_t tickIdx) {
        Update(std::vector<SignObservation>{}, tickIdx);
    }

    // Overload tương thích ngược (vector class_id)
    void Update(const std::vector<int>& classIds, uint32_t tickIdx) {
        std::vector<SignObservation> obsList;
        obsList.reserve(classIds.size());
        for (int c : classIds) {
            obsList.emplace_back(c, 0.75f, 0);
        }
        Update(obsList, tickIdx);
    }

    int   GetCommitted() const { return committed_; }
    float GetCommittedMaxConf() const {
        if (committed_ >= 0 && maxConf_.count(committed_)) return maxConf_.at(committed_);
        return 0.0f;
    }
    int   GetHitCount(int c) const {
        auto it = hitCount_.find(c);
        return (it != hitCount_.end()) ? it->second : 0;
    }
    int   GetConsecutiveHits(int c) const {
        auto it = consecutiveHits_.find(c);
        return (it != consecutiveHits_.end()) ? it->second : 0;
    }
    float GetScore(int c) const {
        auto it = scores_.find(c);
        return (it != scores_.end()) ? it->second : 0.0f;
    }
    bool  JustTimedOut() const { return justTimedOut_; }
    int   LastSeenTick() const { return lastSeenTick_; }

    void Reset() {
        scores_.clear();
        hitCount_.clear();
        consecutiveHits_.clear();
        missCount_.clear();
        maxConf_.clear();
        activated_.clear();
        committed_ = -1;
        lastSeenTick_ = -1;
    }

private:
    int activateFrames_;
    int timeoutFrames_;

    std::map<int, float> scores_;
    std::map<int, int>   hitCount_;
    std::map<int, int>   consecutiveHits_;
    std::map<int, int>   missCount_;
    std::map<int, float> maxConf_;
    std::set<int>        activated_;

    int  committed_    = -1;
    int  lastSeenTick_ = -1;
    bool justTimedOut_ = false;
};

}  // namespace control
}  // namespace ucr

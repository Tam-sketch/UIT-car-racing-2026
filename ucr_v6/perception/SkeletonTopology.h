// ============================================================
// perception/SkeletonTopology.h
//
// Port of pray.py's skeleton_topology / TopoLatch.
// Detects presence of {left, straight, right} branches at a
// junction using skeleton thinning + arm tracing.
//
// Uses a self-contained Zhang-Suen thinning (no opencv_contrib
// needed — OpenCV 4.1.1 base build on Jetson is enough).
// ============================================================
#pragma once

#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <vector>

namespace ucr {
namespace perception {

struct TopoResult {
    bool left     = false;
    bool straight = true;   // safe default when nothing detected
    bool right    = false;
    int  n_arms   = 0;
    int  junction_y = -1;
    int  jx         = -1;
    bool ok         = false;
};

struct SkeletonConfig {
    int   min_arm_size        = 5;
    int   junction_dilate     = 3;
    float dx_left_thresh      = 0.12f;
    float dx_right_thresh     = 0.12f;
    int   mask_close_kernel   = 5;
    float mask_min_area_ratio = 0.005f;
    int   s_verify_x_half     = 15;
    int   s_verify_row_min_px = 3;
    int   s_verify_margin     = 8;
};

// Zhang-Suen thinning. Input binary (0/255), output binary (0/255).
inline cv::Mat ZhangSuenThinning(const cv::Mat& binary) {
    cv::Mat img = (binary > 0);                          // 0/255, CV_8UC1
    cv::Mat prev = cv::Mat::zeros(img.size(), CV_8UC1);  // match size/type
    cv::Mat diff;
    do {
        // --- Sub-iter 1 ---
        cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);
        for (int y = 1; y < img.rows - 1; ++y) {
            const uchar* r0 = img.ptr<uchar>(y - 1);
            const uchar* r1 = img.ptr<uchar>(y);
            const uchar* r2 = img.ptr<uchar>(y + 1);
            uchar* m = marker.ptr<uchar>(y);
            for (int x = 1; x < img.cols - 1; ++x) {
                if (!r1[x]) continue;
                const int p2 = r0[x], p3 = r0[x + 1], p4 = r1[x + 1];
                const int p5 = r2[x + 1], p6 = r2[x], p7 = r2[x - 1];
                const int p8 = r1[x - 1], p9 = r0[x - 1];
                const int A = (!p2 && p3) + (!p3 && p4) + (!p4 && p5) + (!p5 && p6)
                            + (!p6 && p7) + (!p7 && p8) + (!p8 && p9) + (!p9 && p2);
                const int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                if (A == 1 && B >= 2 && B <= 6 &&
                    !(p2 && p4 && p6) && !(p4 && p6 && p8))
                    m[x] = 1;
            }
        }
        img.setTo(0, marker);   // ← replaces `img &= ~marker`

        // --- Sub-iter 2 ---
        marker.setTo(0);
        for (int y = 1; y < img.rows - 1; ++y) {
            const uchar* r0 = img.ptr<uchar>(y - 1);
            const uchar* r1 = img.ptr<uchar>(y);
            const uchar* r2 = img.ptr<uchar>(y + 1);
            uchar* m = marker.ptr<uchar>(y);
            for (int x = 1; x < img.cols - 1; ++x) {
                if (!r1[x]) continue;
                const int p2 = r0[x], p3 = r0[x + 1], p4 = r1[x + 1];
                const int p5 = r2[x + 1], p6 = r2[x], p7 = r2[x - 1];
                const int p8 = r1[x - 1], p9 = r0[x - 1];
                const int A = (!p2 && p3) + (!p3 && p4) + (!p4 && p5) + (!p5 && p6)
                            + (!p6 && p7) + (!p7 && p8) + (!p8 && p9) + (!p9 && p2);
                const int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                if (A == 1 && B >= 2 && B <= 6 &&
                    !(p2 && p4 && p8) && !(p2 && p6 && p8))
                    m[x] = 1;
            }
        }
        img.setTo(0, marker);

        cv::absdiff(img, prev, diff);   // both valid now
        prev = img.clone();
    } while (cv::countNonZero(diff) > 0);

    return img;   // already 0/255
}

// Full pipeline: clean → thin → junction → arms → L/S/R.
inline TopoResult ComputeTopology(const cv::Mat& mask, const SkeletonConfig& cfg) {
    TopoResult r;
    if (mask.empty()) return r;

    // --- Clean: morphological close + largest component ---
    cv::Mat bin = (mask > 0);
    const int k = cfg.mask_close_kernel;
    cv::Mat kEl = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k));
    cv::morphologyEx(bin, bin, cv::MORPH_CLOSE, kEl);

    cv::Mat labels, stats, cent;
    int n = cv::connectedComponentsWithStats(bin, labels, stats, cent, 8);
    if (n > 2) {
        int best = 1, bestArea = 0;
        for (int i = 1; i < n; ++i) {
            int a = stats.at<int>(i, cv::CC_STAT_AREA);
            if (a > bestArea) { bestArea = a; best = i; }
        }
        bin = (labels == best);
    }
    const int H = bin.rows, W = bin.cols;
    if (cv::countNonZero(bin) < cfg.mask_min_area_ratio * H * W) {
        r.straight = true;
        return r;
    }

    // --- Thinning ---
    cv::Mat skel = ZhangSuenThinning(bin);
    if (cv::countNonZero(skel) < 15) {
        r.straight = true;
        return r;
    }

    // --- Junction pixels: skeleton pixels with >=3 8-neighbors ---
    std::vector<int> jys, jxs;
    for (int y = 1; y < H - 1; ++y) {
        const uchar* r0 = skel.ptr<uchar>(y - 1);
        const uchar* r1 = skel.ptr<uchar>(y);
        const uchar* r2 = skel.ptr<uchar>(y + 1);
        for (int x = 1; x < W - 1; ++x) {
            if (!r1[x]) continue;
            int nb = (r0[x-1]>0) + (r0[x]>0) + (r0[x+1]>0)
                   + (r1[x-1]>0) +           (r1[x+1]>0)
                   + (r2[x-1]>0) + (r2[x]>0) + (r2[x+1]>0);
            if (nb >= 3) { jys.push_back(y); jxs.push_back(x); }
        }
    }
    if (jys.empty()) {
        r.straight = true;
        r.ok = true;
        return r;
    }

    // Junction point = max y (closest to car).
    int jy = jys[0];
    for (int y : jys) if (y > jy) jy = y;
    int sumX = 0, cntX = 0;
    for (size_t i = 0; i < jys.size(); ++i) if (jys[i] == jy) { sumX += jxs[i]; ++cntX; }
    int jx = cntX > 0 ? sumX / cntX : W / 2;
    r.junction_y = jy;
    r.jx = jx;

    // --- Arm tracing: remove junction pixels (dilated), then CC ---
    cv::Mat skelArms = skel.clone();
    cv::Mat jMask = cv::Mat::zeros(skel.size(), CV_8UC1);
    for (int y : jys) { }
    for (size_t i = 0; i < jys.size(); ++i) jMask.at<uchar>(jys[i], jxs[i]) = 255;
    cv::Mat kd = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                           cv::Size(cfg.junction_dilate, cfg.junction_dilate));
    cv::dilate(jMask, jMask, kd);
    skelArms.setTo(0, jMask);

    cv::Mat armLabels;
    int armN = cv::connectedComponents(skelArms, armLabels, 8);

    bool L = false, R = false;
    int armTopLeft = -1, armTopRight = -1;
    int nArms = 0;
    for (int lid = 1; lid < armN; ++lid) {
        std::vector<int> ys, xs;
        for (int y = 0; y < H; ++y) {
            const int* rl = armLabels.ptr<int>(y);
            for (int x = 0; x < W; ++x) if (rl[x] == lid) { ys.push_back(y); xs.push_back(x); }
        }
        if ((int)ys.size() < cfg.min_arm_size) continue;
        nArms++;
        // farthest point from junction
        int bi = 0; int bestD = -1;
        for (size_t i = 0; i < ys.size(); ++i) {
            int dy = ys[i] - jy, dx = xs[i] - jx;
            int d2 = dy * dy + dx * dx;
            if (d2 > bestD) { bestD = d2; bi = (int)i; }
        }
        const int ey = ys[bi], ex = xs[bi];
        if (ey - jy > 5) continue;   // arm below junction → ignore
        int yMin = *std::min_element(ys.begin(), ys.end());
        const float dxn = (float)(ex - jx) / (float)W;
        if (dxn < -cfg.dx_left_thresh) {
            L = true;
            if (armTopLeft < 0 || yMin < armTopLeft) armTopLeft = yMin;
        } else if (dxn > cfg.dx_right_thresh) {
            R = true;
            if (armTopRight < 0 || yMin < armTopRight) armTopRight = yMin;
        }
    }
    r.n_arms = nArms;

    // --- Straight verification: road continues upward from junction? ---
    bool S = false;
    int baselineY = -1;
    int stripTopY = jy - 5;
    int armTopMin = -1;
    if (armTopLeft >= 0) armTopMin = armTopLeft;
    if (armTopRight >= 0 && (armTopMin < 0 || armTopRight < armTopMin)) armTopMin = armTopRight;

    if (armTopMin >= 0 && jy > 5) {
        baselineY = armTopMin;
        const int xLo = std::max(0, jx - cfg.s_verify_x_half);
        const int xHi = std::min(W, jx + cfg.s_verify_x_half);
        int y = jy - 5;
        while (y >= 0) {
            const uchar* rb = bin.ptr<uchar>(y);
            int cnt = 0;
            for (int x = xLo; x < xHi; ++x) if (rb[x]) ++cnt;
            if (cnt >= cfg.s_verify_row_min_px) { stripTopY = y; --y; }
            else break;
        }
        if (stripTopY <= baselineY - cfg.s_verify_margin) S = true;
    }

    if (!L && !S && !R) S = true;   // no branch at all → straight

    r.left     = L;
    r.straight = S;
    r.right    = R;
    r.ok       = true;
    return r;
}

// Persistence filter: latch the raw topo key after `confirmFrames` stable matches.
class TopoLatch {
public:
    explicit TopoLatch(int confirmFrames) : confirm_(confirmFrames) {}

    TopoResult Update(const TopoResult& raw) {
        const std::array<char, 3> k = { (char)raw.left, (char)raw.straight, (char)raw.right };
        if (k == latchedKey_) { pending_.fill(0); cnt_ = 0; return MakeLatched(); }
        if (k == pending_) { ++cnt_; }
        else { pending_ = k; cnt_ = 1; }
        if (cnt_ >= confirm_) { latchedKey_ = k; pending_.fill(0); cnt_ = 0; }
        return MakeLatched();
    }

    void Reset() {
        latchedKey_ = { 0, 1, 0 };   // S default
        pending_.fill(0); cnt_ = 0;
    }

private:
    TopoResult MakeLatched() const {
        TopoResult r;
        r.left     = latchedKey_[0];
        r.straight = latchedKey_[1];
        r.right    = latchedKey_[2];
        r.ok       = true;
        return r;
    }
    int confirm_;
    std::array<char, 3> latchedKey_ = { 0, 1, 0 };
    std::array<char, 3> pending_    = { 0, 0, 0 };
    int cnt_ = 0;
};

}  // namespace perception
}  // namespace ucr
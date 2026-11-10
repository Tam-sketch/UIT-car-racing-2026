// --------------------------------------------------------
// Roundabout state machine (Option 1 — minimal)
//
// Ý tưởng:
//   1. Sign "roundabout" → APPROACH
//   2. Curve tăng cao   → IN_CIRCLE (PID tự chạy vòng)
//   3. Curve sụt thấp   → EXITING   (ép lái phải)
//   4. Sau 6-15 ticks   → NONE      (về PID bình thường)
// --------------------------------------------------------
inline constexpr float RB_CURVE_ENTER_THRESH  = 30.0f;   // |far_e-near_e| > → enter circle
inline constexpr float RB_CURVE_EXIT_THRESH   = 12.0f;   // < → exit candidate
inline constexpr int   RB_CURVE_ENTER_TICKS   = 3;       // streak để confirm enter
inline constexpr int   RB_CURVE_EXIT_TICKS    = 3;       // streak để confirm exit
inline constexpr int   RB_MIN_CIRCLE_TICKS    = 5;       // min time in circle
inline constexpr float RB_EXIT_STEER_DEG      = 10.0f;   // right steer (positive, negate in code)
inline constexpr int   RB_EXIT_STEER_TICKS    = 6;       // force steer duration
inline constexpr int   RB_EXIT_TIMEOUT_TICKS  = 15;      // hard exit
inline constexpr int   RB_APPROACH_TIMEOUT    = 100;     // 10s; give up if no curve
inline constexpr int   RB_COOLDOWN_TICKS      = 20;      // 2s after exit
inline constexpr float RB_SPEED_CAP_CMS       = 12.0f;   // speed cap during roundabout
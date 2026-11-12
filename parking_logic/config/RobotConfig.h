// ============================================================
// config/RobotConfig.h
// ============================================================
#pragma once

namespace ucr {
namespace config {

// ============================================================
// SECTION A — ACTIVE IN CURRENT BUILD
// ============================================================

// --------------------------------------------------------
// Model paths
// --------------------------------------------------------
inline constexpr char ROAD_ENGINE_PATH[] =
    "/home/car4/Downloads/road/kga2.engine";
inline constexpr char SIGN_ENGINE_PATH[] =
    "/home/car4/Downloads/sign/signv4.engine";

// --------------------------------------------------------
// Hardware limits
// --------------------------------------------------------
inline constexpr float MAX_SPEED_CMS = 30.0f;
inline constexpr float MIN_SPEED_CMS = 20.0f;

// --------------------------------------------------------
// Servo calibration
// --------------------------------------------------------
inline constexpr float SERVO_LEFT_SLOPE    = 0.96f;
inline constexpr float SERVO_LEFT_OFFSET   = -6.2f;
inline constexpr float SERVO_RIGHT_SLOPE   = 0.71f;
inline constexpr float SERVO_RIGHT_OFFSET  = 2.5f;

inline constexpr float MAX_REAL_ANGLE_DEG = 30.0f;
inline constexpr int   MAX_SERVO_CMD      = 55;

inline float RealAngleToServo(float real_deg) {
    if (real_deg > 0.0f)
        return (real_deg - SERVO_LEFT_OFFSET) / SERVO_LEFT_SLOPE;
    if (real_deg < 0.0f)
        return (real_deg - SERVO_RIGHT_OFFSET) / SERVO_RIGHT_SLOPE;
    return 0.0f;
}

inline float ServoToRealAngle(float servo_cmd) {
    if (servo_cmd > 0.0f)
        return SERVO_LEFT_SLOPE * servo_cmd + SERVO_LEFT_OFFSET;
    if (servo_cmd < 0.0f)
        return SERVO_RIGHT_SLOPE * servo_cmd + SERVO_RIGHT_OFFSET;
    return 0.0f;
}

// --------------------------------------------------------
// Control loop
// --------------------------------------------------------
inline constexpr int   CONTROL_PERIOD_MS = 100;   // 10 Hz
inline constexpr float MAX_ACCEL_CMS2    = 20.0f;

// --------------------------------------------------------
// Road model
// --------------------------------------------------------
inline constexpr int   ROAD_DA_CHANNEL  = 1;
inline constexpr float ROAD_CONF_MARGIN = 0.02f;

// --------------------------------------------------------
// Lane follower tuning
// --------------------------------------------------------
inline constexpr float  CURVE_SCALE        = 150.0f;
inline constexpr double POST_PID_GAIN      = 2.2;
inline constexpr int    MIN_ROWS           = 5;
inline constexpr double COVERAGE_THRESHOLD = 0.05;

// --------------------------------------------------------
// UI
// --------------------------------------------------------
inline constexpr bool SHOW_UI = true;

// ============================================================
// SECTION B — TURN / SIGN INTEGRATION
// ============================================================

// --------------------------------------------------------
// Sign model
// --------------------------------------------------------
inline constexpr float SIGN_CONF_THRESH = 0.7f;
inline constexpr float SIGN_IOU_THRESH  = 0.45f;
inline constexpr int   SIGN_INPUT_SIZE  = 416;

namespace sign_class {
    inline constexpr int CONSTRUCTION_AHEAD = 0;
    inline constexpr int NO_LEFT            = 1;
    inline constexpr int NO_RIGHT           = 2;
    inline constexpr int PARKING            = 3;
    inline constexpr int ROUNDABOUT         = 4;
    inline constexpr int TRAFFIC_LIGHT      = 5;
    inline constexpr int TURN_LEFT          = 6;
    inline constexpr int TURN_RIGHT         = 7;
    inline constexpr int WALKING_PERSON     = 8;
}

inline constexpr const char* SignClassName(int id) {
    switch (id) {
        case 0: return "construct";
        case 1: return "no_left";
        case 2: return "no_right";
        case 3: return "parking";
        case 4: return "roundabout";
        case 5: return "traffic_light";
        case 6: return "turn_left";
        case 7: return "turn_right";
        case 8: return "walking_person";
        default: return "-";
    }
}

// --------------------------------------------------------
// Traffic light HSV classifier
// --------------------------------------------------------
namespace tl {
    inline constexpr int RED_H1_LO = 0;
    inline constexpr int RED_H1_HI = 10;
    inline constexpr int RED_H2_LO = 170;
    inline constexpr int RED_H2_HI = 180;
    inline constexpr int YEL_H_LO  = 15;
    inline constexpr int YEL_H_HI  = 35;
    inline constexpr int GRN_H_LO  = 40;
    inline constexpr int GRN_H_HI  = 90;
    inline constexpr int S_MIN     = 100;
    inline constexpr int V_MIN     = 250;
    inline constexpr int MIN_LIT_PIXELS = 10; 
    inline constexpr float ROI_EXPAND   = 1.05f;
}

inline constexpr int PERSON_STOP_HANGOVER_TICKS = 20;   // ~2 s
inline constexpr int RED_LIGHT_HANGOVER_TICKS   = 10;   // ~1 s

// --------------------------------------------------------
// Sign detect + voter timers
// --------------------------------------------------------
inline constexpr int SIGN_DETECT_EVERY             = 2;    
inline constexpr int SIGN_VOTER_ACTIVATE_FRAMES    = 2;    
inline constexpr int SIGN_VOTER_TIMEOUT_FRAMES     = 60;   
inline constexpr int SIGN_HOLD_FRAMES              = 18;   
inline constexpr int SIGN_SNAPSHOT_HOLD_FRAMES     = 3;    
inline constexpr int TOPO_CONFIRM_FRAMES           = 3;    
inline constexpr int RESET_DEBOUNCE_FRAMES         = 90;   

// --------------------------------------------------------
// V21 Opening Detection & Bezier
// --------------------------------------------------------
inline constexpr int   OPEN_THRESH             = 15;    
inline constexpr int   OPEN_MIN_HEIGHT         = 10;    
inline constexpr int   OPEN_SMOOTH_WIN         = 5;     
inline constexpr int   OPEN_BASE_WIN           = 50;    

inline constexpr int   BEZIER_CORNER_RADIUS    = 40; 
inline constexpr int   BEZIER_MIN_TURN_DIST    = 4;
inline constexpr int   BEZIER_EXTEND_LEN       = 80;

// --------------------------------------------------------
// V21 Turn Completion & PID Blending
// --------------------------------------------------------
inline constexpr int   TURN_MIN_TICKS          = 25;  
inline constexpr int   TURN_MAX_TICKS          = 100; 
inline constexpr int   TURN_STRAIGHT_TICKS     = 5;
inline constexpr int   TURN_SYNTH_WIDTH_PX     = 60;

inline constexpr double TURN_LOOKAHEAD_Y_RATIO = 0.30; 
inline constexpr double TURN_NEAR_Y_RATIO      = 0.85;

inline constexpr double TURN_FAR_WEIGHT        = 0.85;
inline constexpr int    TURN_RAMP_TICKS        = 8; 
inline constexpr double TURN_ERROR_CAP         = 60.0;

// --------------------------------------------------------
// Heading (BNO055) transform
// --------------------------------------------------------
inline constexpr float HEADING_SIGN       = 1.0f;
inline constexpr float HEADING_OFFSET_DEG = 0.0f;

// --------------------------------------------------------
// PARKING PARAMETERS (MỚI THÊM)
// --------------------------------------------------------
inline constexpr float TOF_BLOCKED_MM = 300.0f;     // 30 cm = 300 mm
inline constexpr float PARKING_SPEED_CMS = 8.0f;    // Tốc độ tiến/lùi chậm
inline constexpr float PARKING_STEER_ANGLE = 28.0f; // Góc đánh lái khi lùi
inline constexpr float PARKING_TARGET_HEADING = 45.0f;
inline constexpr float PARKING_HEADING_TOLERANCE = 3.0f;

}  // namespace config
}  // namespace ucr
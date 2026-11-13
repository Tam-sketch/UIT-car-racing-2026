// ============================================================
// TelemetryIntegration.h — UCR 2026 UDP Telemetry (Single Header)
//
// Tự chứa (standalone), chỉ phụ thuộc OpenCV.
// Fix sẵn mã PEDESTRIAN_STOP = 13 (spec yêu cầu nhưng code cũ thiếu).
//
// CÁCH DÙNG (3 bước):
//
//   1) Include:
//        #include "TelemetryIntegration.h"
//
//   2) Init 1 lần trong main:
//        telemetry::Helper telem;
//        telem.Init("192.168.1.100", 9999, "TEAM_NAME");
//
//   3) Trong main loop, mỗi frame gọi:
//        telemetry::StateInputs in;
//        in.pedestrian_detected  = /* bool */;
//        in.red_light_detected   = /* bool */;
//        in.green_light_detected = /* bool */;
//        in.in_roundabout        = /* bool */;
//        in.active_sign_class    = /* int class id, -1 = none */;
//        in.last_seen_sign_class = /* int class id, -1 = none */;
//        in.sign_turn_left       = 6;    // id trong model của bạn
//        in.sign_turn_right      = 7;
//        in.sign_no_left         = 1;
//        in.sign_no_right        = 2;
//        in.sign_construction    = 0;
//        in.sign_roundabout      = 4;
//        in.sign_parking         = 3;
//
//        const double now_s = std::chrono::duration<double>(
//            std::chrono::steady_clock::now().time_since_epoch()).count();
//
//        telem.TrackInputs(in, now_s);              // cập nhật state
//        telem.Tick(frame, appliedSpeed, angle, now_s); // gửi UDP @ 20Hz
//
// Xong. State, priority, hangovers, JPEG encode, header build, rate-limit
// đều được xử lý trong header này.
// ============================================================
#pragma once

#include <opencv2/opencv.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace telemetry {

// ============================================================
// 1) VehicleState — 14 mã đầy đủ (0..13)
// ============================================================
enum class VehicleState : int32_t {
    NONE              = 0,   // Bình thường / không có tín hiệu
    GREEN_LIGHT       = 1,   // Đèn xanh
    RED_LIGHT         = 2,   // Đèn đỏ
    ROAD_WORK         = 3,   // Công trường thi công
    BRIDGE            = 4,   // Lên cầu
    ROUNDABOUT        = 5,   // Vòng xuyến
    OBSTACLE          = 6,   // Vật cản
    TURN_RIGHT        = 7,   // Biển rẽ phải
    NO_TURN_RIGHT     = 8,   // Biển cấm rẽ phải
    TURN_LEFT         = 9,   // Biển rẽ trái
    NO_TURN_LEFT      = 10,  // Biển cấm rẽ trái
    PARKING_SIGN      = 11,  // Biển đỗ xe
    PARKING           = 12,  // Đang đỗ xe
    PEDESTRIAN_STOP   = 13   // Dừng vì người đi bộ
};

inline const char* VehicleStateName(int s) {
    switch (s) {
        case 0:  return "NONE";
        case 1:  return "GREEN_LIGHT";
        case 2:  return "RED_LIGHT";
        case 3:  return "ROAD_WORK";
        case 4:  return "BRIDGE";
        case 5:  return "ROUNDABOUT";
        case 6:  return "OBSTACLE";
        case 7:  return "TURN_RIGHT";
        case 8:  return "NO_TURN_RIGHT";
        case 9:  return "TURN_LEFT";
        case 10: return "NO_TURN_LEFT";
        case 11: return "PARKING_SIGN";
        case 12: return "PARKING";
        case 13: return "PEDESTRIAN_STOP";
        default: return "UNKNOWN";
    }
}

// ============================================================
// 2) Client — UDP socket (fire-and-forget, no ACK)
// ============================================================
class Client {
public:
    Client() = default;
    ~Client() { Close(); }
    Client(const Client&) = delete;
    Client& operator=(const Client&) = delete;

    bool Init(const std::string& serverIp, int port, const std::string& carId) {
        Close();
        carId_ = carId.substr(0, 15);   // max 15 ký tự ASCII
        seq_   = 0;

        sock_ = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (sock_ < 0) return false;

        std::memset(&addr_, 0, sizeof(addr_));
        addr_.sin_family = AF_INET;
        addr_.sin_port   = htons(static_cast<uint16_t>(port));
        if (::inet_pton(AF_INET, serverIp.c_str(), &addr_.sin_addr) != 1) {
            Close();
            return false;
        }
        return true;
    }

    void Close() {
        if (sock_ >= 0) {
            ::close(sock_);
            sock_ = -1;
        }
    }

    bool IsReady() const { return sock_ >= 0; }

    // Full packet: 40-byte header + JPEG body
    bool SendFrame(const cv::Mat& frame,
                   float speedCms,
                   int steeringDeg,
                   int vehicleState,
                   int jpegW = 480,
                   int jpegH = 288,
                   int jpegQuality = 55) {
        if (sock_ < 0 || frame.empty()) return false;

        // Resize nếu cần
        cv::Mat resized;
        if (frame.cols != jpegW || frame.rows != jpegH) {
            cv::resize(frame, resized, cv::Size(jpegW, jpegH));
        } else {
            resized = frame;
        }

        // Encode JPEG
        std::vector<uchar> jpg;
        const std::vector<int> params = {
            cv::IMWRITE_JPEG_QUALITY, jpegQuality
        };
        if (!cv::imencode(".jpg", resized, jpg, params)) return false;
        if (jpg.size() > 60000) return false;   // spec: max 60KB

        // Build packet
        const uint32_t jlen = static_cast<uint32_t>(jpg.size());
        std::vector<uint8_t> buf(40u + jlen);
        buildHeader(buf.data(), speedCms, steeringDeg, vehicleState, jlen);
        std::memcpy(buf.data() + 40, jpg.data(), jlen);

        return sendRaw(buf.data(), buf.size());
    }

    // State-only packet: 40-byte header, no JPEG
    // Spec: speed=0, steering=0, jpeg_len=0
    bool SendStateOnly(int vehicleState) {
        if (sock_ < 0) return false;
        uint8_t buf[40];
        buildHeader(buf, 0.0f, 0, vehicleState, 0u);
        return sendRaw(buf, 40);
    }

private:
    int          sock_ = -1;
    sockaddr_in  addr_{};
    std::string  carId_;
    uint32_t     seq_ = 0;

    // Little-endian write helpers
    static void putU32LE(uint8_t* p, uint32_t v) {
        p[0] = static_cast<uint8_t>( v        & 0xFF);
        p[1] = static_cast<uint8_t>((v >> 8)  & 0xFF);
        p[2] = static_cast<uint8_t>((v >> 16) & 0xFF);
        p[3] = static_cast<uint8_t>((v >> 24) & 0xFF);
    }
    static void putI32LE(uint8_t* p, int32_t v) {
        putU32LE(p, static_cast<uint32_t>(v));
    }
    static void putF32LE(uint8_t* p, float v) {
        uint32_t u = 0;
        std::memcpy(&u, &v, 4);
        putU32LE(p, u);
    }

    void buildHeader(uint8_t* buf,
                     float speedCms,
                     int steeringDeg,
                     int vehicleState,
                     uint32_t jpegLen) {
        // [0..3] magic "UCRC"
        buf[0] = 'U'; buf[1] = 'C'; buf[2] = 'R'; buf[3] = 'C';

        // [4..19] car_id, null-padded 16 bytes
        std::memset(buf + 4, 0, 16);
        std::memcpy(buf + 4, carId_.data(),
                    std::min(carId_.size(), static_cast<size_t>(15)));

        // [20..23] sequence
        putU32LE(buf + 20, ++seq_);

        // [24..27] speed (float cm/s)
        putF32LE(buf + 24, speedCms);

        // [28..31] steering (int32 deg)
        putI32LE(buf + 28, steeringDeg);

        // [32..35] state (int32)
        putI32LE(buf + 32, vehicleState);

        // [36..39] jpeg_len
        putU32LE(buf + 36, jpegLen);
    }

    bool sendRaw(const uint8_t* data, size_t len) {
        const ssize_t sent = ::sendto(sock_, data, len, 0,
            reinterpret_cast<const sockaddr*>(&addr_), sizeof(addr_));
        return sent == static_cast<ssize_t>(len);
    }
};

// ============================================================
// 3) StateInputs — dữ liệu vào từ code của bạn
// ============================================================
struct StateInputs {
    // Detection flags (current frame)
    bool pedestrian_detected  = false;
    bool red_light_detected   = false;
    bool green_light_detected = false;

    // Behavior flags
    bool in_roundabout   = false;
    bool bridge_active   = false;
    bool parking_active  = false;

    // Sign class (dùng -1 nếu không có)
    int active_sign_class    = -1;  // đang trigger turn
    int last_seen_sign_class = -1;  // vừa thấy, chưa turn

    // Class ID mapping (override nếu model của bạn dùng ID khác)
    int sign_turn_left    = 6;
    int sign_turn_right   = 7;
    int sign_no_left      = 1;
    int sign_no_right     = 2;
    int sign_construction = 0;
    int sign_roundabout   = 4;
    int sign_parking      = 3;
};

// ============================================================
// 4) StateTracker — priority + hangover
// ============================================================
class StateTracker {
public:
    // Hangover durations (seconds)
    double person_hangover_s = 2.0;
    double red_hangover_s    = 1.0;
    double green_hangover_s  = 1.0;

    void TrackTimers(const StateInputs& in, double now_s) {
        if (in.pedestrian_detected)  last_person_s_ = now_s;
        if (in.red_light_detected)   last_red_s_    = now_s;
        if (in.green_light_detected) last_green_s_  = now_s;
    }

    // Priority:
    //   person > red > parking_active > green > roundabout > bridge
    //   > active_sign_turn > last_seen_sign > NONE
    int Compute(const StateInputs& in, double now_s) const {
        if (now_s - last_person_s_ <= person_hangover_s) return 13;
        if (now_s - last_red_s_    <= red_hangover_s)    return 2;
        if (in.parking_active)                            return 12;
        if (now_s - last_green_s_  <= green_hangover_s)  return 1;
        if (in.in_roundabout)                             return 5;
        if (in.bridge_active)                             return 4;

        if (in.active_sign_class >= 0) {
            const int s = mapSign(in.active_sign_class, in);
            if (s > 0) return s;
        }
        if (in.last_seen_sign_class >= 0) {
            const int s = mapSign(in.last_seen_sign_class, in);
            if (s > 0) return s;
        }
        return 0;
    }

private:
    double last_person_s_ = -1e9;
    double last_red_s_    = -1e9;
    double last_green_s_  = -1e9;

    static int mapSign(int cls, const StateInputs& in) {
        if (cls == in.sign_turn_left)    return 9;
        if (cls == in.sign_turn_right)   return 7;
        if (cls == in.sign_no_left)      return 10;
        if (cls == in.sign_no_right)     return 8;
        if (cls == in.sign_construction) return 3;
        if (cls == in.sign_roundabout)   return 5;
        if (cls == in.sign_parking)      return 11;
        return 0;
    }
};

// ============================================================
// 5) Helper — combine Client + Tracker + rate limiter
// ============================================================
class Helper {
public:
    // Tuning
    int send_interval_ms = 50;   // 20 Hz
    int jpeg_width       = 480;
    int jpeg_height      = 288;
    int jpeg_quality     = 55;

    // Optional: log 1Hz
    bool log_1hz = false;

    bool Init(const std::string& ip, int port, const std::string& carId) {
        return client_.Init(ip, port, carId);
    }
    bool IsReady() const { return client_.IsReady(); }

    // Gọi mỗi frame (hoặc mỗi tick) để cập nhật state
    void TrackInputs(const StateInputs& in, double now_s) {
        tracker_.TrackTimers(in, now_s);
        last_inputs_ = in;
        current_state_ = tracker_.Compute(in, now_s);
    }

    // Gọi mỗi frame — gửi UDP rate-limited @ send_interval_ms
    // Returns true nếu gửi thành công trong tick này
    bool Tick(const cv::Mat& frame, float speedCms, int steeringDeg, double now_s) {
        if (!client_.IsReady()) return false;

        const double ms = (now_s - last_sent_s_) * 1000.0;
        if (ms < send_interval_ms) return false;

        const bool ok = client_.SendFrame(
            frame, speedCms, steeringDeg, current_state_,
            jpeg_width, jpeg_height, jpeg_quality);

        if (ok) {
            last_sent_s_ = now_s;
            last_sent_state_ = current_state_;
        }
        return ok;
    }

    // Force gửi state-only ngay (không chờ rate limit, không có ảnh)
    // Dùng khi state thay đổi quan trọng (red light → phanh ngay)
    bool SendStateNow(int state) {
        return client_.SendStateOnly(state);
    }

    // Get state hiện tại (0..13)
    int CurrentState() const { return current_state_; }

private:
    Client       client_;
    StateTracker tracker_;
    StateInputs  last_inputs_;
    int          current_state_    = 0;
    int          last_sent_state_  = -1;
    double       last_sent_s_      = -1e9;
};

}  // namespace telemetry
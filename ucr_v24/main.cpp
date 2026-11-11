            char text1[192], text2[192];
            std::snprintf(text1, sizeof(text1),
                "FPS %.1f | spd %.1f/%.1f | PID %.1f | out %d | kf %.2f%s",
                fps, applied_speed, out.speed, out.angle_pre_gain,
                (int)std::lround(out.angle), out.kept_frac,
                (!out.valid && inferOK) ? " INV" : "");
            std::snprintf(text2, sizeof(text2),
                "sign %s | %s | near %.1f far %.1f cov %.3f",
                ucr::config::SignClassName(voter.GetCommitted()),
                out.state_str.c_str(),
                out.near_e, out.far_e, out.coverage);
            cv::putText(overlay, text1, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.55,
                        cv::Scalar(0, 255, 0), 2);
            cv::putText(overlay, text2, {10, 55}, cv::FONT_HERSHEY_SIMPLEX, 0.55,
                        cv::Scalar(0, 255, 0), 2);
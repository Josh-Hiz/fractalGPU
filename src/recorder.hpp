// Frame recorder: pipes raw RGB frames to ffmpeg over popen.
//
// Capture happens after the fractal quad is drawn but before ImGui, so the
// recording is the rendered scene without the UI overlay.
//
// Pacing: each pump() call duplicates the current frame as many times as
// needed to keep up with the target FPS, so the video plays back in real
// time even when rendering is slower than the target frame rate.
#pragma once

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

class Recorder {
  public:
    enum class Quality { Low = 0, Medium = 1, High = 2, VeryHigh = 3 };

    static const char *qualityName(Quality q) {
        switch (q) {
        case Quality::Low:      return "Low";
        case Quality::Medium:   return "Medium";
        case Quality::High:     return "High";
        case Quality::VeryHigh: return "Very High";
        }
        return "?";
    }

    bool start(int w, int h, int fps, Quality q, const std::string &path) {
        m_err.clear();
        if (m_pipe)
            return false;

        // ffmpeg + yuv420p require even dimensions.
        m_w = w & ~1;
        m_h = h & ~1;
        if (m_w < 2 || m_h < 2) {
            m_err = "window too small";
            return false;
        }
        m_fps = fps < 1 ? 1 : fps;
        m_frames = 0;
        m_path = path;

        int crf = 23;
        const char *preset = "medium";
        switch (q) {
        case Quality::Low:      crf = 28; preset = "fast";   break;
        case Quality::Medium:   crf = 23; preset = "medium"; break;
        case Quality::High:     crf = 18; preset = "medium"; break;
        case Quality::VeryHigh: crf = 14; preset = "slow";   break;
        }

        // Path is auto-generated (timestamp-only chars), so no shell-quoting risk.
        char cmd[1024];
        std::snprintf(cmd, sizeof(cmd),
            "ffmpeg -hide_banner -loglevel error -y "
            "-f rawvideo -pixel_format rgb24 -video_size %dx%d -framerate %d "
            "-i - -vf vflip -c:v libx264 -preset %s -crf %d "
            "-pix_fmt yuv420p '%s'",
            m_w, m_h, m_fps, preset, crf, path.c_str());

        m_pipe = popen(cmd, "w");
        if (!m_pipe) {
            m_err = "popen failed (is ffmpeg installed and on PATH?)";
            return false;
        }

        m_start = std::chrono::steady_clock::now();
        m_nextFrame = m_start;
        return true;
    }

    // Write `rgb` (width*height*3 bytes, bottom-up from glReadPixels) to the
    // pipe as many times as needed to match wall-clock pacing at target FPS.
    void pump(const unsigned char *rgb) {
        if (!m_pipe)
            return;
        using namespace std::chrono;
        auto now = steady_clock::now();
        auto frameDur = duration_cast<steady_clock::duration>(
            duration<double>(1.0 / (double)m_fps));

        size_t bytes = (size_t)m_w * (size_t)m_h * 3;
        // Cap duplicates per call so a long stall (e.g. user dragged the
        // window) doesn't dump thousands of identical frames at once.
        int writes = 0;
        while (now >= m_nextFrame && writes < 16) {
            if (std::fwrite(rgb, 1, bytes, m_pipe) != bytes) {
                m_err = "pipe write failed";
                stop();
                return;
            }
            m_frames++;
            m_nextFrame += frameDur;
            writes++;
        }
        // If we hit the cap, snap forward so we don't carry a huge backlog.
        if (writes >= 16 && now > m_nextFrame)
            m_nextFrame = now;
    }

    void stop() {
        if (!m_pipe)
            return;
        pclose(m_pipe);
        m_pipe = nullptr;
    }

    bool isActive() const { return m_pipe != nullptr; }
    int  width()    const { return m_w; }
    int  height()   const { return m_h; }
    int  fps()      const { return m_fps; }
    int  frames()   const { return m_frames; }
    double elapsedSec() const {
        if (!isActive())
            return 0.0;
        using namespace std::chrono;
        return duration<double>(steady_clock::now() - m_start).count();
    }
    const std::string &path()  const { return m_path; }
    const std::string &error() const { return m_err; }

  private:
    FILE *m_pipe = nullptr;
    int m_w = 0, m_h = 0;
    int m_fps = 30;
    int m_frames = 0;
    std::string m_path;
    std::string m_err;
    std::chrono::steady_clock::time_point m_start;
    std::chrono::steady_clock::time_point m_nextFrame;
};

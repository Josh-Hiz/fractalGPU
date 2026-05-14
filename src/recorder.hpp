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
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

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
        // Spawn the background writer so fwrite() to the (potentially slow)
        // ffmpeg pipe never blocks the main render/UI loop.
        m_stopWriter = false;
        m_writer = std::thread([this] { writerLoop(); });
        return true;
    }

    // True when wall-clock has reached the next-frame slot. Use this to gate
    // animation / readback so each call to pump() writes exactly one fresh
    // frame, never a duplicate.
    bool dueForFrame() const {
        if (!m_pipe)
            return false;
        return std::chrono::steady_clock::now() >= m_nextFrame;
    }

    // Write one frame. Caller is expected to gate on dueForFrame() and supply
    // a freshly-rendered image — duplicates would create freeze-and-jump
    // artifacts in the video.
    //
    // Hot path: copy `rgb` into a queued buffer and return. The background
    // writer thread does the actual fwrite(). Bounded queue + cv backpressure
    // prevents memory blowup if ffmpeg can't keep up; in the common case the
    // queue is empty when we push.
    void writeFrame(const unsigned char *rgb) {
        if (!m_pipe)
            return;
        size_t bytes = (size_t)m_w * (size_t)m_h * 3;
        std::vector<unsigned char> buf(rgb, rgb + bytes);
        {
            std::unique_lock<std::mutex> lk(m_mtx);
            m_cvSpace.wait(lk, [&] {
                return m_queue.size() < kQueueCap || m_stopWriter;
            });
            if (m_stopWriter)
                return;
            m_queue.push(std::move(buf));
        }
        m_cvWork.notify_one();

        m_frames++;
        using namespace std::chrono;
        auto frameDur = duration_cast<steady_clock::duration>(
            duration<double>(1.0 / (double)m_fps));
        m_nextFrame += frameDur;
        // If we've fallen far behind wall-clock (rendering stall), snap the
        // schedule forward to "now" so we don't try to catch up by writing a
        // burst of frames — that would just compress motion into a blur.
        auto now = steady_clock::now();
        if (now > m_nextFrame + frameDur)
            m_nextFrame = now;
    }

    void stop() {
        if (!m_pipe)
            return;
        // Signal writer to drain the queue and exit, then close the pipe.
        {
            std::lock_guard<std::mutex> lk(m_mtx);
            m_stopWriter = true;
        }
        m_cvWork.notify_all();
        m_cvSpace.notify_all();
        if (m_writer.joinable())
            m_writer.join();
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
    // Background writer: drains queued buffers into the ffmpeg pipe so the
    // main thread is never blocked by a slow encoder.
    void writerLoop() {
        for (;;) {
            std::vector<unsigned char> buf;
            {
                std::unique_lock<std::mutex> lk(m_mtx);
                m_cvWork.wait(lk,
                              [&] { return !m_queue.empty() || m_stopWriter; });
                if (m_queue.empty() && m_stopWriter)
                    return;
                buf = std::move(m_queue.front());
                m_queue.pop();
            }
            // Tell the producer there's space, then do the (potentially slow)
            // pipe write outside the mutex.
            m_cvSpace.notify_one();
            if (!m_pipe)
                return;
            if (std::fwrite(buf.data(), 1, buf.size(), m_pipe) != buf.size()) {
                m_err = "pipe write failed";
                return;
            }
        }
    }

    static constexpr size_t kQueueCap = 8;

    FILE *m_pipe = nullptr;
    int m_w = 0, m_h = 0;
    int m_fps = 30;
    int m_frames = 0;
    std::string m_path;
    std::string m_err;
    std::chrono::steady_clock::time_point m_start;
    std::chrono::steady_clock::time_point m_nextFrame;

    std::thread m_writer;
    std::mutex m_mtx;
    std::condition_variable m_cvWork;  // signals queued frames to writer
    std::condition_variable m_cvSpace; // signals queue space to producer
    std::queue<std::vector<unsigned char>> m_queue;
    bool m_stopWriter = false;
};

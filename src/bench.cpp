// Headless benchmark frontend for the CPU and GPU fractal renderers.
//
// Spins up a hidden GLFW window so the GPU path can register a real
// GL_RGBA32F texture for CUDA-GL interop (falls through to the host-upload
// fallback if registration fails — same code path as the main app).
//
// For each (backend, fractal, mode, resolution) scenario we run W warmup
// frames + F measured frames, advancing the camera azimuth each frame so
// rays don't repeat the same path. Per frame we record:
//   * frame_ms : wall-clock around renderer.render() — end-to-end including
//                memcpy on the GPU fallback path.
//   * kern_ms  : GPU only, from the renderer's internal cudaEvent — pure
//                kernel + sync time.
// Aggregates: mean, median, stddev, min, max, p99.
//
// CSV output is append-mode with a tag column so you can checkout/build/run
// across versions and accumulate rows for diff plotting.

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>

#include "config.hpp"
#include "renderer_cpu.hpp"
#ifdef FRACTAL_USE_CUDA
#include "renderer_gpu.hpp"
#endif

namespace {

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------
struct Stats {
    double mean = 0, median = 0, stddev = 0, min_ = 0, max_ = 0, p99 = 0;
};

Stats computeStats(std::vector<double> v) {
    Stats s;
    if (v.empty())
        return s;
    std::sort(v.begin(), v.end());
    s.min_ = v.front();
    s.max_ = v.back();
    s.median = v[v.size() / 2];
    size_t p99idx = (size_t)std::min<double>(
        (double)v.size() - 1.0, std::ceil(0.99 * (double)v.size()) - 1.0);
    s.p99 = v[p99idx];
    double sum = 0.0;
    for (double x : v)
        sum += x;
    s.mean = sum / (double)v.size();
    double sq = 0.0;
    for (double x : v)
        sq += (x - s.mean) * (x - s.mean);
    s.stddev = std::sqrt(sq / (double)v.size());
    return s;
}

// ---------------------------------------------------------------------------
// Camera helper (replicates main.cpp's OrbitCamera so we vary view per frame)
// ---------------------------------------------------------------------------
struct OrbitState {
    float dist = 3.0f;
    float azimuth = 0.0f;
    float elevation = 0.3f;
    void apply(RenderParams &p) {
        p.camera.target = {0.0f, 0.0f, 0.0f};
        p.camera.position = glm::vec3(
            dist * std::cos(elevation) * std::sin(azimuth),
            dist * std::sin(elevation),
            dist * std::cos(elevation) * std::cos(azimuth));
        p.camera.up = {0.0f, 1.0f, 0.0f};
    }
};

// ---------------------------------------------------------------------------
// Scenario definitions
// ---------------------------------------------------------------------------
struct Scenario {
    std::string name;
    FractalType ft;
    RenderMode mode;
    int width;
    int height;
    int maxSteps;
    int volSteps;     // unused if mode == Surface
    float orbitDist;
    bool volSmem = true; // volumetric kernel variant: smem two-pass vs. fused
};

const char *fractalName(FractalType t) {
    switch (t) {
    case FractalType::Mandelbulb: return "Mandelbulb";
    case FractalType::Mandelbox:  return "Mandelbox";
    case FractalType::Julia:      return "Julia";
    }
    return "?";
}
const char *modeName(RenderMode m) {
    return m == RenderMode::Surface ? "Surface" : "Volumetric";
}

void buildScenarios(std::vector<Scenario> &out, const std::string &set,
                    int wOverride, int hOverride) {
    auto defaultDist = [](FractalType ft) {
        if (ft == FractalType::Mandelbox) return 5.0f;
        if (ft == FractalType::Julia)     return 2.0f;
        return 3.0f;
    };
    auto add = [&](const char *name, FractalType ft, RenderMode m,
                   int w, int h, int steps, int volSteps,
                   bool volSmem = true) {
        if (wOverride > 0) w = wOverride;
        if (hOverride > 0) h = hOverride;
        out.push_back({name, ft, m, w, h, steps, volSteps,
                       defaultDist(ft), volSmem});
    };

    if (set == "quick") {
        add("low",  FractalType::Mandelbulb, RenderMode::Surface,    640, 360, 128, 0);
        add("low",  FractalType::Mandelbox,  RenderMode::Surface,    640, 360, 128, 0);
        add("med",  FractalType::Mandelbulb, RenderMode::Surface,   1280, 720, 128, 0);
        add("vol",  FractalType::Mandelbulb, RenderMode::Volumetric,1280, 720, 128, 24);
    } else if (set == "full") {
        for (auto ft : {FractalType::Mandelbulb,
                        FractalType::Mandelbox,
                        FractalType::Julia}) {
            add("low", ft, RenderMode::Surface,    640,  360, 128, 0);
            add("med", ft, RenderMode::Surface,   1280,  720, 128, 0);
            add("hi",  ft, RenderMode::Surface,   1920, 1080, 128, 0);
            add("vol", ft, RenderMode::Volumetric,1280,  720, 128, 24);
        }
    } else if (set == "steps") {
        // Sweep maxSteps to see scaling; one fractal/resolution.
        for (int s : {32, 64, 128, 256, 512}) {
            char nm[16]; std::snprintf(nm, sizeof(nm), "s%d", s);
            add(nm, FractalType::Mandelbulb, RenderMode::Surface, 1280, 720, s, 0);
        }
    } else if (set == "vol") {
        // Sweep volumetric steps (smem variant only — pair with vol-compare
        // for the smem-vs-fused A/B).
        for (int v : {8, 16, 24, 32, 64}) {
            char nm[16]; std::snprintf(nm, sizeof(nm), "v%d", v);
            add(nm, FractalType::Mandelbulb, RenderMode::Volumetric,
                1280, 720, 128, v);
        }
    } else if (set == "vol-compare") {
        // A/B the smem two-pass kernel vs. the fused single-pass kernel
        // across a few step counts. Names are tagged so CSV rows stay
        // distinguishable.
        for (int v : {8, 16, 24, 32, 64}) {
            char a[24], b[24];
            std::snprintf(a, sizeof(a), "v%d-smem",  v);
            std::snprintf(b, sizeof(b), "v%d-fused", v);
            add(a, FractalType::Mandelbulb, RenderMode::Volumetric,
                1280, 720, 128, v, true);
            add(b, FractalType::Mandelbulb, RenderMode::Volumetric,
                1280, 720, 128, v, false);
        }
    } else {
        std::fprintf(stderr, "Unknown scenario set '%s' — using 'quick'\n",
                     set.c_str());
        buildScenarios(out, "quick", wOverride, hOverride);
    }
}

// CPU is single-threaded scalar code; cap it to reasonable resolutions so
// the bench finishes in this lifetime. Volumetric isn't implemented on CPU.
bool cpuShouldRun(const Scenario &s) {
    if (s.mode == RenderMode::Volumetric)
        return false;
    return (long)s.width * s.height <= 640L * 360L;
}

void configureScenarioParams(const Scenario &s, RenderParams &p,
                              OrbitState &orb) {
    p = RenderParams{};
    p.fractalType = s.ft;
    p.renderMode  = s.mode;
    p.maxSteps    = s.maxSteps;
    p.renderScale = 1.0f; // we feed exact dimensions; bypass the *scale path
    p.vol.steps   = s.volSteps;
    p.vol.useSharedMem = s.volSmem;
    orb = OrbitState{};
    orb.dist = s.orbitDist;
    orb.apply(p);
}

double wallSec() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------
struct Result {
    std::string backend;
    Scenario scen;
    int warmup = 0;
    int frames = 0;
    Stats frameMs;   // wall-clock around render() — always populated
    Stats kernelMs;  // GPU cudaEvent kernel-only — only when hasKernel
    bool hasKernel = false;
};

// ---------------------------------------------------------------------------
// Runners
// ---------------------------------------------------------------------------
void runCpuScenario(const Scenario &s, int warmup, int frames, Result &r) {
    CPURenderer renderer;
    RenderParams p; OrbitState orb;
    configureScenarioParams(s, p, orb);

    constexpr float kAzPerFrame = 0.0174533f; // 1 deg/frame so rays vary
    for (int i = 0; i < warmup; ++i) {
        orb.azimuth = (float)i * kAzPerFrame; orb.apply(p);
        renderer.render(s.width, s.height, p);
    }
    std::vector<double> frameMs; frameMs.reserve(frames);
    for (int i = 0; i < frames; ++i) {
        orb.azimuth = (float)(warmup + i) * kAzPerFrame; orb.apply(p);
        double t0 = wallSec();
        renderer.render(s.width, s.height, p);
        double t1 = wallSec();
        frameMs.push_back((t1 - t0) * 1000.0);
    }
    r.backend  = "CPU";
    r.scen     = s;
    r.warmup   = warmup;
    r.frames   = frames;
    r.frameMs  = computeStats(frameMs);
    r.hasKernel = false;
}

#ifdef FRACTAL_USE_CUDA
void runGpuScenario(GPURenderer &renderer, GLuint tex, int &curW, int &curH,
                    const Scenario &s, int warmup, int frames, Result &r) {
    RenderParams p; OrbitState orb;
    configureScenarioParams(s, p, orb);

    if (curW != s.width || curH != s.height) {
        // Detach (so the CUDA resource is unregistered before we resize the
        // texture's backing storage), resize, then re-register.
        renderer.setOutputTexture(0, 0, 0);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, s.width, s.height, 0,
                     GL_RGBA, GL_FLOAT, nullptr);
        renderer.setOutputTexture(tex, s.width, s.height);
        curW = s.width;
        curH = s.height;
    }

    constexpr float kAzPerFrame = 0.0174533f;
    for (int i = 0; i < warmup; ++i) {
        orb.azimuth = (float)i * kAzPerFrame; orb.apply(p);
        renderer.render(s.width, s.height, p);
    }
    std::vector<double> frameMs;  frameMs.reserve(frames);
    std::vector<double> kernelMs; kernelMs.reserve(frames);
    for (int i = 0; i < frames; ++i) {
        orb.azimuth = (float)(warmup + i) * kAzPerFrame; orb.apply(p);
        double t0 = wallSec();
        renderer.render(s.width, s.height, p);
        double t1 = wallSec();
        frameMs.push_back((t1 - t0) * 1000.0);
        kernelMs.push_back(renderer.renderMs());
    }
    r.backend   = "GPU";
    r.scen      = s;
    r.warmup    = warmup;
    r.frames    = frames;
    r.frameMs   = computeStats(frameMs);
    r.kernelMs  = computeStats(kernelMs);
    r.hasKernel = true;
}
#endif

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------
void printResultsTable(const std::vector<Result> &rs) {
    std::printf(
        "\n%-7s %-6s %-11s %-11s %5s %5s %5s | %10s %10s %10s | "
        "%10s %10s %10s | %7s\n",
        "backend", "scen", "fractal", "mode",
        "W", "H", "steps",
        "frame.mean", "frame.med", "frame.p99",
        "kern.mean",  "kern.med",  "kern.p99",
        "fps");
    std::printf(
        "----------------------------------------------------------------"
        "-------------------------------------------------------------\n");
    for (const auto &r : rs) {
        double fps = r.frameMs.mean > 0 ? 1000.0 / r.frameMs.mean : 0.0;
        std::printf("%-7s %-6s %-11s %-11s %5d %5d %5d | %10.3f %10.3f %10.3f | ",
                    r.backend.c_str(), r.scen.name.c_str(),
                    fractalName(r.scen.ft), modeName(r.scen.mode),
                    r.scen.width, r.scen.height, r.scen.maxSteps,
                    r.frameMs.mean, r.frameMs.median, r.frameMs.p99);
        if (r.hasKernel)
            std::printf("%10.3f %10.3f %10.3f | %7.1f\n",
                        r.kernelMs.mean, r.kernelMs.median, r.kernelMs.p99,
                        fps);
        else
            std::printf("%10s %10s %10s | %7.1f\n", "-", "-", "-", fps);
    }
}

void appendCSV(const std::string &path, const std::string &tag,
               const std::vector<Result> &rs) {
    bool exists = std::ifstream(path).good();
    std::ofstream out(path, std::ios::app);
    if (!out) {
        std::fprintf(stderr, "Failed to open CSV %s\n", path.c_str());
        return;
    }
    if (!exists) {
        out << "timestamp,tag,backend,scenario,fractal,mode,width,height,"
               "max_steps,vol_steps,warmup,frames,"
               "frame_mean_ms,frame_median_ms,frame_stddev_ms,"
               "frame_min_ms,frame_max_ms,frame_p99_ms,"
               "kernel_mean_ms,kernel_median_ms,kernel_p99_ms,"
               "fps_mean\n";
    }
    std::time_t t = std::time(nullptr);
    std::tm tm_ = *std::localtime(&t);
    char ts[32];
    std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", &tm_);
    for (const auto &r : rs) {
        double fps = r.frameMs.mean > 0 ? 1000.0 / r.frameMs.mean : 0.0;
        out << ts << "," << tag << "," << r.backend << "," << r.scen.name
            << "," << fractalName(r.scen.ft) << "," << modeName(r.scen.mode)
            << "," << r.scen.width << "," << r.scen.height
            << "," << r.scen.maxSteps << "," << r.scen.volSteps
            << "," << r.warmup << "," << r.frames
            << "," << r.frameMs.mean   << "," << r.frameMs.median
            << "," << r.frameMs.stddev << "," << r.frameMs.min_
            << "," << r.frameMs.max_   << "," << r.frameMs.p99 << ",";
        if (r.hasKernel)
            out << r.kernelMs.mean << "," << r.kernelMs.median
                << "," << r.kernelMs.p99;
        else
            out << ",,";
        out << "," << fps << "\n";
    }
}

void usage(const char *prog) {
    std::printf(
        "Usage: %s [options]\n"
        "  --cpu                only run CPU scenarios\n"
        "  --gpu                only run GPU scenarios\n"
        "  --no-cpu / --no-gpu  exclude one backend\n"
        "  --csv PATH           append per-scenario rows to CSV\n"
        "  --tag STRING         label written to CSV (e.g. git short sha)\n"
        "  --scenarios SET      quick (default) | full | steps | vol | vol-compare\n"
        "  --warmup N           default 5\n"
        "  --frames N           default 50\n"
        "  --width W            override scenario render width\n"
        "  --height H           override scenario render height\n"
        "  -h / --help          this message\n",
        prog);
}

} // namespace

int main(int argc, char **argv) {
    bool useCpu = true;
    bool useGpu = true;
    std::string csvPath;
    std::string tag = "untagged";
    std::string scenarioSet = "quick";
    int warmup = 5;
    int frames = 50;
    int wOverride = 0, hOverride = 0;

    auto need = [&](int &i) -> const char * {
        if (i + 1 >= argc) {
            std::fprintf(stderr, "missing value for %s\n", argv[i]);
            std::exit(2);
        }
        return argv[++i];
    };

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--cpu")        { useCpu = true;  useGpu = false; }
        else if (a == "--gpu")        { useCpu = false; useGpu = true; }
        else if (a == "--no-cpu")     { useCpu = false; }
        else if (a == "--no-gpu")     { useGpu = false; }
        else if (a == "--csv")        { csvPath = need(i); }
        else if (a == "--tag")        { tag = need(i); }
        else if (a == "--scenarios")  { scenarioSet = need(i); }
        else if (a == "--warmup")     { warmup = std::atoi(need(i)); }
        else if (a == "--frames")     { frames = std::atoi(need(i)); }
        else if (a == "--width")      { wOverride = std::atoi(need(i)); }
        else if (a == "--height")     { hOverride = std::atoi(need(i)); }
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else {
            std::fprintf(stderr, "Unknown arg: %s\n", argv[i]);
            usage(argv[0]);
            return 2;
        }
    }

#ifndef FRACTAL_USE_CUDA
    if (useGpu)
        std::fprintf(stderr,
                     "(built without FRACTAL_USE_CUDA — GPU scenarios skipped)\n");
    useGpu = false;
#endif

    std::vector<Scenario> scenarios;
    buildScenarios(scenarios, scenarioSet, wOverride, hOverride);

    std::vector<Result> results;

    GLFWwindow *win = nullptr;
    GLuint tex = 0;
#ifdef FRACTAL_USE_CUDA
    if (useGpu) {
        if (const char *f = std::getenv("FRACTAL_FORCE_X11");
            f && *f && *f != '0')
            glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
        if (!glfwInit()) {
            std::fprintf(stderr, "GLFW init failed\n");
            return 1;
        }
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        win = glfwCreateWindow(64, 64, "FractalGPUBench", nullptr, nullptr);
        if (!win) {
            std::fprintf(stderr, "Hidden window failed\n");
            glfwTerminate();
            return 1;
        }
        glfwMakeContextCurrent(win);
        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
            std::fprintf(stderr, "GLAD init failed\n");
            return 1;
        }
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        gpu_print_diagnostics(16, 16, 24);
    }
#endif

    std::printf("\nRunning %zu scenario(s)  warmup=%d  frames=%d  "
                "tag='%s'  set='%s'\n",
                scenarios.size(), warmup, frames, tag.c_str(),
                scenarioSet.c_str());

    if (useCpu) {
        std::printf("\n[CPU]\n");
        for (const auto &s : scenarios) {
            if (!cpuShouldRun(s)) {
                std::printf("  skip CPU %-6s %-11s %-11s %dx%d (would take too long)\n",
                            s.name.c_str(), fractalName(s.ft),
                            modeName(s.mode), s.width, s.height);
                continue;
            }
            std::printf("  CPU  %-6s %-11s %-11s %dx%d ...",
                        s.name.c_str(), fractalName(s.ft),
                        modeName(s.mode), s.width, s.height);
            std::fflush(stdout);
            Result r;
            runCpuScenario(s, warmup, frames, r);
            std::printf(" %.2f ms (avg)\n", r.frameMs.mean);
            results.push_back(std::move(r));
        }
    }

#ifdef FRACTAL_USE_CUDA
    if (useGpu) {
        std::printf("\n[GPU]\n");
        GPURenderer gpu;
        int curW = -1, curH = -1;
        for (const auto &s : scenarios) {
            std::printf("  GPU  %-6s %-11s %-11s %dx%d ...",
                        s.name.c_str(), fractalName(s.ft),
                        modeName(s.mode), s.width, s.height);
            std::fflush(stdout);
            Result r;
            runGpuScenario(gpu, tex, curW, curH, s, warmup, frames, r);
            std::printf(" frame %.3f ms / kernel %.3f ms\n",
                        r.frameMs.mean, r.kernelMs.mean);
            results.push_back(std::move(r));
        }
        // Detach before destroying the texture / window.
        gpu.setOutputTexture(0, 0, 0);
    }
#endif

    printResultsTable(results);

    if (!csvPath.empty()) {
        appendCSV(csvPath, tag, results);
        std::printf("\nAppended %zu rows to %s (tag='%s')\n",
                    results.size(), csvPath.c_str(), tag.c_str());
    }

#ifdef FRACTAL_USE_CUDA
    if (win) {
        glDeleteTextures(1, &tex);
        glfwDestroyWindow(win);
        glfwTerminate();
    }
#endif
    return 0;
}

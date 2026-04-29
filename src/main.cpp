#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <iostream>

#include "config.hpp"
#include "recorder.hpp"
#include "renderer_cpu.hpp"
#include "shaders.hpp"
#include <ctime>
#include <vector>
#ifdef FRACTAL_USE_CUDA
#include "renderer_gpu.hpp"
#endif

// GL helpers
static GLuint compileShader(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetShaderInfoLog(s, 512, nullptr, log);
        std::cerr << "Shader error: " << log << "\n";
    }
    return s;
}

static GLuint createProgram(const char *vert, const char *frag) {
    GLuint vs = compileShader(GL_VERTEX_SHADER, vert);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, frag);
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    glDeleteShader(vs);
    glDeleteShader(fs);
    return p;
}

// orbit camera
//  Spherical
// coords (dist, azimuth, elevation) → Cartesian camera position
struct OrbitCamera {
    float dist = 3.0f;
    float azimuth = 0.0f;
    float elevation = 0.3f;
    glm::vec3 target = {0.0f, 0.0f, 0.0f};

    void apply(RenderParams &p) const {
        p.camera.target = target;
        p.camera.position =
            target + glm::vec3(dist * std::cos(elevation) * std::sin(azimuth),
                               dist * std::sin(elevation),
                               dist * std::cos(elevation) * std::cos(azimuth));
        p.camera.up = {0.0f, 1.0f, 0.0f};
    }
};

// app state
struct App {
    OrbitCamera orbit;
    RenderParams params;
    CPURenderer cpuRenderer;
#ifdef FRACTAL_USE_CUDA
    GPURenderer gpuRenderer;
    bool useGPU = true;
#else
    bool useGPU = false;
#endif
    GLuint tex = 0;
    int texW = 0;
    int texH = 0;
    bool dirty = true;
    bool dragging = false;
    double lastMX = 0.0;
    double lastMY = 0.0;

    // Recording. drawUI() flips the request flags; the main loop services
    // them so the recorder dimensions reflect the current framebuffer size.
    Recorder recorder;
    int recFps = 30;
    int recQuality = (int)Recorder::Quality::Medium;
    bool recRequestStart = false;
    bool recRequestStop = false;
    std::string recMessage;

    // Auto camera animation — drives azimuth (and optional elevation sway)
    // off real wall-clock dt so motion is smooth regardless of render rate.
    bool autoRotate = false;
    float autoSpinDeg = 30.0f;       // azimuth deg/sec
    float autoElevAmp = 0.0f;        // elevation sway amplitude in radians
    float autoElevPeriod = 8.0f;     // seconds for one full sway cycle
    float autoElevBase = 0.3f;       // baseline elevation, captured on toggle
    double autoTime = 0.0;           // accumulated time since toggle on
    std::chrono::steady_clock::time_point lastFrameTime;
    bool haveLastFrameTime = false;
};

static App g;

// GLFW callbacks
static void onMouseBtn(GLFWwindow *win, int btn, int action, int) {
    if (ImGui::GetIO().WantCaptureMouse)
        return;
    if (btn == GLFW_MOUSE_BUTTON_LEFT) {
        g.dragging = (action == GLFW_PRESS);
        if (g.dragging)
            glfwGetCursorPos(win, &g.lastMX, &g.lastMY);
    }
}

static void onCursorPos(GLFWwindow *, double x, double y) {
    if (!g.dragging)
        return;
    float dx = (float)(x - g.lastMX);
    float dy = (float)(y - g.lastMY);
    g.lastMX = x;
    g.lastMY = y;
    g.orbit.azimuth -= dx * 0.005f;
    g.orbit.elevation += dy * 0.005f;
    g.orbit.elevation = std::clamp(g.orbit.elevation, -1.4f, 1.4f);
    g.orbit.apply(g.params);
    g.dirty = true;
}

static void onScroll(GLFWwindow *, double, double dy) {
    if (ImGui::GetIO().WantCaptureMouse)
        return;
    g.orbit.dist *= std::pow(0.88f, (float)dy);
    g.orbit.dist = std::clamp(g.orbit.dist, 0.3f, 30.0f);
    g.orbit.apply(g.params);
    g.dirty = true;
}

// ImGui
static void applyStyle() {
    ImGuiStyle &s = ImGui::GetStyle();
    s.WindowRounding = 8.0f;
    s.FrameRounding = 4.0f;
    s.GrabRounding = 4.0f;
    s.ItemSpacing = {8.0f, 6.0f};
    s.FramePadding = {6.0f, 4.0f};
    s.WindowPadding = {12.0f, 12.0f};
    s.ScrollbarSize = 10.0f;

    ImVec4 *c = s.Colors;
    c[ImGuiCol_WindowBg] = {0.06f, 0.05f, 0.11f, 0.96f};
    c[ImGuiCol_ChildBg] = {0.08f, 0.07f, 0.14f, 1.00f};
    c[ImGuiCol_FrameBg] = {0.12f, 0.10f, 0.22f, 1.00f};
    c[ImGuiCol_FrameBgHovered] = {0.18f, 0.14f, 0.32f, 1.00f};
    c[ImGuiCol_FrameBgActive] = {0.24f, 0.18f, 0.44f, 1.00f};
    c[ImGuiCol_TitleBg] = {0.08f, 0.06f, 0.16f, 1.00f};
    c[ImGuiCol_TitleBgActive] = {0.14f, 0.10f, 0.28f, 1.00f};
    c[ImGuiCol_Header] = {0.20f, 0.14f, 0.40f, 1.00f};
    c[ImGuiCol_HeaderHovered] = {0.30f, 0.20f, 0.58f, 1.00f};
    c[ImGuiCol_HeaderActive] = {0.36f, 0.24f, 0.68f, 1.00f};
    c[ImGuiCol_SliderGrab] = {0.44f, 0.22f, 0.86f, 1.00f};
    c[ImGuiCol_SliderGrabActive] = {0.58f, 0.34f, 1.00f, 1.00f};
    c[ImGuiCol_Button] = {0.28f, 0.14f, 0.60f, 1.00f};
    c[ImGuiCol_ButtonHovered] = {0.40f, 0.22f, 0.80f, 1.00f};
    c[ImGuiCol_ButtonActive] = {0.52f, 0.30f, 0.94f, 1.00f};
    c[ImGuiCol_CheckMark] = {0.72f, 0.52f, 1.00f, 1.00f};
    c[ImGuiCol_Text] = {0.92f, 0.90f, 0.96f, 1.00f};
    c[ImGuiCol_TextDisabled] = {0.44f, 0.40f, 0.54f, 1.00f};
    c[ImGuiCol_Separator] = {0.22f, 0.16f, 0.40f, 1.00f};
    c[ImGuiCol_Tab] = {0.14f, 0.10f, 0.28f, 1.00f};
    c[ImGuiCol_TabHovered] = {0.30f, 0.20f, 0.56f, 1.00f};
    c[ImGuiCol_TabActive] = {0.24f, 0.16f, 0.48f, 1.00f};
    c[ImGuiCol_PopupBg] = {0.08f, 0.07f, 0.14f, 0.98f};
    c[ImGuiCol_ScrollbarBg] = {0.06f, 0.05f, 0.11f, 1.00f};
    c[ImGuiCol_ScrollbarGrab] = {0.28f, 0.14f, 0.58f, 1.00f};
}

// presets
static void presetMandelbulb(float power, int iter, float dist, glm::vec3 palA,
                             glm::vec3 palB) {
    g.params.fractalType = FractalType::Mandelbulb;
    g.params.mandelbulb.power = power;
    g.params.mandelbulb.iterations = iter;
    g.params.color.paletteA = palA;
    g.params.color.paletteB = palB;
    g.orbit.dist = dist;
    g.orbit.apply(g.params);
    g.dirty = true;
}

static void presetMandelbox(float scale, int iter, float dist, glm::vec3 palA,
                            glm::vec3 palB) {
    g.params.fractalType = FractalType::Mandelbox;
    g.params.mandelbox.scale = scale;
    g.params.mandelbox.iterations = iter;
    g.params.color.paletteA = palA;
    g.params.color.paletteB = palB;
    g.orbit.dist = dist;
    g.orbit.apply(g.params);
    g.dirty = true;
}

static void presetJulia(glm::vec4 c, int iter, float dist, glm::vec3 palA,
                        glm::vec3 palB) {
    g.params.fractalType = FractalType::Julia;
    g.params.julia.c = c;
    g.params.julia.iterations = iter;
    g.params.color.paletteA = palA;
    g.params.color.paletteB = palB;
    g.orbit.dist = dist;
    g.orbit.apply(g.params);
    g.dirty = true;
}

// UI
static void drawUI() {
    ImGui::SetNextWindowPos({10.0f, 10.0f}, ImGuiCond_Always);
    ImGui::SetNextWindowSize({290.0f, 0.0f}, ImGuiCond_Always);
    ImGui::Begin("FractalGPU", nullptr,
                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoCollapse |
                     ImGuiWindowFlags_AlwaysAutoResize);

    bool changed = false;

    // fractal type
    ImGui::SeparatorText("Fractal");
    const char *types[] = {"Mandelbulb", "Mandelbox", "Julia (quaternion)"};
    int ft = (int)g.params.fractalType;
    if (ImGui::Combo("Type", &ft, types, 3)) {
        g.params.fractalType = (FractalType)ft;
        changed = true;
    }

    ImGui::Spacing();
    ImGui::SeparatorText("Presets");

    // fractal-specific presets
    if (g.params.fractalType == FractalType::Mandelbulb) {
        if (ImGui::Button("Classic p8")) {
            presetMandelbulb(8, 10, 3.0f, {0.2f, 0.5f, 1.0f},
                             {1.0f, 0.3f, 0.1f});
        }
        ImGui::SameLine();
        if (ImGui::Button("Alien p6")) {
            presetMandelbulb(6, 10, 3.0f, {0.0f, 0.8f, 0.4f},
                             {0.6f, 0.1f, 0.8f});
        }
        if (ImGui::Button("Spiky p12")) {
            presetMandelbulb(12, 12, 3.0f, {1.0f, 0.6f, 0.0f},
                             {0.1f, 0.1f, 0.5f});
        }
        ImGui::SameLine();
        if (ImGui::Button("Soft p3")) {
            presetMandelbulb(3, 15, 3.0f, {0.9f, 0.7f, 0.5f},
                             {0.3f, 0.2f, 0.7f});
        }
    } else if (g.params.fractalType == FractalType::Mandelbox) {
        if (ImGui::Button("Standard")) {
            presetMandelbox(2.0f, 15, 5.0f, {0.2f, 0.4f, 0.9f},
                            {0.8f, 0.5f, 0.1f});
        }
        ImGui::SameLine();
        if (ImGui::Button("Inverted")) {
            presetMandelbox(-2.0f, 12, 5.0f, {0.8f, 0.2f, 0.5f},
                            {0.2f, 0.7f, 0.8f});
        }
        ImGui::SameLine();
        if (ImGui::Button("Soft")) {
            presetMandelbox(1.5f, 15, 5.0f, {0.9f, 0.8f, 0.5f},
                            {0.3f, 0.4f, 0.8f});
        }
    } else {
        if (ImGui::Button("Swirl")) {
            presetJulia({-0.2f, 0.6f, 0.2f, 0.0f}, 15, 2.0f, {0.2f, 0.5f, 1.0f},
                        {1.0f, 0.4f, 0.2f});
        }
        ImGui::SameLine();
        if (ImGui::Button("Coral")) {
            presetJulia({-0.125f, -0.256f, 0.847f, 0.089f}, 15, 2.0f,
                        {1.0f, 0.5f, 0.3f}, {0.3f, 0.2f, 0.8f});
        }
        if (ImGui::Button("Crystal")) {
            presetJulia({0.355f, 0.355f, 0.0f, 0.0f}, 15, 2.0f,
                        {0.4f, 0.9f, 1.0f}, {1.0f, 0.8f, 0.2f});
        }
        ImGui::SameLine();
        if (ImGui::Button("Dragon")) {
            presetJulia({-0.4f, 0.6f, 0.0f, 0.0f}, 15, 2.0f, {1.0f, 0.2f, 0.3f},
                        {0.2f, 0.6f, 0.4f});
        }
    }

    ImGui::Spacing();
    ImGui::SeparatorText("Parameters");

    // fractal-specific sliders
    if (g.params.fractalType == FractalType::Mandelbulb) {
        auto &mb = g.params.mandelbulb;
        if (ImGui::SliderFloat("Power", &mb.power, 1.0f, 20.0f))
            changed = true;
        if (ImGui::SliderInt("Iterations", &mb.iterations, 1, 24))
            changed = true;
        if (ImGui::SliderFloat("Bailout", &mb.bailout, 1.0f, 8.0f))
            changed = true;
    } else if (g.params.fractalType == FractalType::Mandelbox) {
        auto &bx = g.params.mandelbox;
        if (ImGui::SliderFloat("Scale", &bx.scale, -3.0f, 3.0f))
            changed = true;
        if (ImGui::SliderInt("Iterations", &bx.iterations, 1, 24))
            changed = true;
        if (ImGui::SliderFloat("Fold limit", &bx.foldLimit, 0.5f, 2.0f))
            changed = true;
        if (ImGui::SliderFloat("Min radius", &bx.minRadius, 0.1f, 1.0f))
            changed = true;
        if (ImGui::SliderFloat("Fixed radius", &bx.fixedRadius, 0.5f, 2.0f))
            changed = true;
    } else {
        auto &jl = g.params.julia;
        if (ImGui::SliderFloat("c.x", &jl.c.x, -1.0f, 1.0f))
            changed = true;
        if (ImGui::SliderFloat("c.y", &jl.c.y, -1.0f, 1.0f))
            changed = true;
        if (ImGui::SliderFloat("c.z", &jl.c.z, -1.0f, 1.0f))
            changed = true;
        if (ImGui::SliderFloat("c.w", &jl.c.w, -1.0f, 1.0f))
            changed = true;
        if (ImGui::SliderInt("Iterations", &jl.iterations, 1, 24))
            changed = true;
    }

    ImGui::Spacing();
    ImGui::SeparatorText("Render Mode");
    const char *modes[] = {"Surface (ray march)", "Volumetric"};
    int rm = (int)g.params.renderMode;
    if (ImGui::Combo("Mode", &rm, modes, 2)) {
        g.params.renderMode = (RenderMode)rm;
        changed = true;
    }
    if (g.params.renderMode == RenderMode::Volumetric) {
        auto &v = g.params.vol;
        if (ImGui::SliderInt("Vol steps", &v.steps, 4, 32))
            changed = true;
        if (ImGui::SliderFloat("Density falloff", &v.densityFalloff, 0.5f, 32.0f))
            changed = true;
        if (ImGui::SliderFloat("Absorption", &v.absorption, 0.1f, 20.0f))
            changed = true;
        if (ImGui::SliderFloat("Emission", &v.emission, 0.0f, 5.0f))
            changed = true;
        if (ImGui::SliderFloat("Trap weight", &v.trapWeight, 0.0f, 8.0f))
            changed = true;
        if (ImGui::SliderFloat("Bound radius", &v.bound, 0.5f, 8.0f))
            changed = true;
    }

    ImGui::Spacing();
    ImGui::SeparatorText("Render Quality");
    if (ImGui::SliderFloat("Resolution", &g.params.renderScale, 0.1f, 1.0f))
        changed = true;
    if (ImGui::SliderInt("Max steps", &g.params.maxSteps, 16, 512))
        changed = true;
    if (ImGui::SliderFloat("Epsilon", &g.params.epsilon, 0.0001f, 0.005f))
        changed = true;

    // lighting
    if (ImGui::CollapsingHeader("Lighting")) {
        auto &l = g.params.light;
        if (ImGui::SliderFloat("Light X", &l.direction.x, -1.0f, 1.0f))
            changed = true;
        if (ImGui::SliderFloat("Light Y", &l.direction.y, -1.0f, 1.0f))
            changed = true;
        if (ImGui::SliderFloat("Light Z", &l.direction.z, -1.0f, 1.0f))
            changed = true;
        if (ImGui::SliderFloat("Ambient", &l.ambient, 0.0f, 0.5f))
            changed = true;
        if (ImGui::SliderFloat("Diffuse", &l.diffuse, 0.0f, 2.0f))
            changed = true;
        if (ImGui::SliderFloat("Specular", &l.specular, 0.0f, 2.0f))
            changed = true;
        if (ImGui::SliderFloat("Shininess", &l.shininess, 1.0f, 128.0f))
            changed = true;
        if (ImGui::Checkbox("Soft shadows", &l.softShadows))
            changed = true;
        if (l.softShadows) {
            if (ImGui::SliderFloat("Shadow K", &l.shadowK, 1.0f, 32.0f))
                changed = true;
        }
        if (ImGui::Checkbox("Ambient occlusion", &l.aoEnabled))
            changed = true;
        if (l.aoEnabled) {
            if (ImGui::SliderInt("AO steps", &l.aoSteps, 1, 10))
                changed = true;
            if (ImGui::SliderFloat("AO step size", &l.aoStepSize, 0.01f, 0.2f))
                changed = true;
        }
    }

    // color
    if (ImGui::CollapsingHeader("Color")) {
        auto &col = g.params.color;
        if (ImGui::ColorEdit3("Palette A", glm::value_ptr(col.paletteA)))
            changed = true;
        if (ImGui::ColorEdit3("Palette B", glm::value_ptr(col.paletteB)))
            changed = true;
        if (ImGui::ColorEdit3("BG top", glm::value_ptr(col.bgTop)))
            changed = true;
        if (ImGui::ColorEdit3("BG bottom", glm::value_ptr(col.bgBottom)))
            changed = true;
        if (ImGui::Checkbox("Orbit trap color", &col.orbitTrap))
            changed = true;
        if (col.orbitTrap) {
            if (ImGui::SliderFloat("Trap scale", &col.trapScale, 0.1f, 5.0f))
                changed = true;
        }
        if (ImGui::SliderFloat("Glow", &col.glowStr, 0.0f, 1.0f))
            changed = true;
    }

    // camera
    if (ImGui::CollapsingHeader("Camera")) {
        bool cc = false;
        if (ImGui::SliderFloat("Distance", &g.orbit.dist, 0.3f, 30.0f))
            cc = true;
        if (ImGui::SliderFloat("Azimuth", &g.orbit.azimuth, -3.14f, 3.14f))
            cc = true;
        if (ImGui::SliderFloat("Elevation", &g.orbit.elevation, -1.4f, 1.4f))
            cc = true;
        if (ImGui::SliderFloat("FOV", &g.params.camera.fov, 20.0f, 120.0f))
            cc = true;
        if (cc) {
            g.orbit.apply(g.params);
            changed = true;
        }
        ImGui::TextDisabled("pos %.2f %.2f %.2f", g.params.camera.position.x,
                            g.params.camera.position.y,
                            g.params.camera.position.z);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

#ifdef FRACTAL_USE_CUDA
    ImGui::SeparatorText("Backend");
    bool gpuToggle = g.useGPU;
    if (ImGui::RadioButton("CPU", !gpuToggle)) {
        g.useGPU = false;
        g.dirty = true;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("GPU (CUDA)", gpuToggle)) {
        g.useGPU = true;
        g.dirty = true;
    }
    ImGui::Spacing();
#endif

    // Auto camera animation — designed to feed clean, time-paced motion
    // into the recorder so videos don't look like manual drag jitter.
    ImGui::SeparatorText("Animation");
    if (ImGui::Checkbox("Auto rotate", &g.autoRotate)) {
        if (g.autoRotate) {
            // Snapshot user's elevation as the sway baseline and reset
            // phase so the sin term starts at zero (no jump).
            g.autoElevBase = g.orbit.elevation;
            g.autoTime = 0.0;
        }
    }
    if (g.autoRotate) {
        ImGui::SliderFloat("Spin speed", &g.autoSpinDeg, -180.0f, 180.0f,
                           "%.0f deg/s");
        ImGui::SliderFloat("Elev sway", &g.autoElevAmp, 0.0f, 0.8f);
        if (g.autoElevAmp > 0.001f)
            ImGui::SliderFloat("Sway period", &g.autoElevPeriod, 1.0f, 30.0f,
                               "%.1f s");
    }
    ImGui::Spacing();

    // Recording.
    ImGui::SeparatorText("Recording");
    if (g.recorder.isActive()) {
        // Visual cue: red-tinted button while recording.
        ImGui::PushStyleColor(ImGuiCol_Button,        {0.62f, 0.16f, 0.20f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0.78f, 0.22f, 0.26f, 1.0f});
        ImGui::PushStyleColor(ImGuiCol_ButtonActive,  {0.92f, 0.30f, 0.34f, 1.0f});
        // Stable label/ID — putting live stats in the label changes the
        // ImGui ID every frame and breaks press/release click detection.
        if (ImGui::Button("Stop recording", {-1.0f, 30.0f}))
            g.recRequestStop = true;
        ImGui::PopStyleColor(3);
        ImGui::TextDisabled("%.1fs  %d frames  %dx%d @ %d fps",
                            g.recorder.elapsedSec(), g.recorder.frames(),
                            g.recorder.width(), g.recorder.height(),
                            g.recorder.fps());
    } else {
        const char *qNames[] = {"Low", "Medium", "High", "Very High"};
        ImGui::Combo("Quality", &g.recQuality, qNames, 4);
        ImGui::SliderInt("FPS", &g.recFps, 15, 60);
        if (ImGui::Button("Record", {-1.0f, 30.0f}))
            g.recRequestStart = true;
    }
    if (!g.recMessage.empty())
        ImGui::TextDisabled("%s", g.recMessage.c_str());
    ImGui::Spacing();

    // render button + stats
    if (ImGui::Button("Render now", {-1.0f, 30.0f}))
        g.dirty = true;

    ImGui::Spacing();
    double ms = g.useGPU
#ifdef FRACTAL_USE_CUDA
                    ? g.gpuRenderer.renderMs()
#else
                    ? 0.0
#endif
                    : g.cpuRenderer.renderMs();
    int rw = g.useGPU
#ifdef FRACTAL_USE_CUDA
                 ? g.gpuRenderer.renderWidth()
#else
                 ? 0
#endif
                 : g.cpuRenderer.renderWidth();
    int rh = g.useGPU
#ifdef FRACTAL_USE_CUDA
                 ? g.gpuRenderer.renderHeight()
#else
                 ? 0
#endif
                 : g.cpuRenderer.renderHeight();
    ImGui::TextDisabled("Render time : %.1f ms", ms);
    ImGui::TextDisabled("Resolution  : %d x %d", rw, rh);
    ImGui::TextDisabled("Left-drag: orbit   Scroll: zoom");

    ImGui::End();

    if (changed)
        g.dirty = true;
}

//  main
int main() {
    // Opt-in escape hatch: on Wayland with hybrid graphics (e.g. AMD iGPU
    // driving the display + NVIDIA dGPU for CUDA), the EGL loader binds the
    // GL context to the iGPU and CUDA-GL interop refuses cross-vendor
    // registration. Forcing GLFW onto X11/GLX (via XWayland) lets the
    // `__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia` env pair
    // route the GL context to NVIDIA so interop works. Default behaviour is
    // unchanged.
    if (const char *force = std::getenv("FRACTAL_FORCE_X11");
        force && force[0] && force[0] != '0') {
        glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
    }

    if (!glfwInit()) {
        std::cerr << "GLFW init failed\n";
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow *win =
        glfwCreateWindow(1280, 720, "FractalGPU", nullptr, nullptr);
    if (!win) {
        std::cerr << "Window failed\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1); // vsync

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "GLAD init failed\n";
        glfwTerminate();
        return -1;
    }

    glfwSetMouseButtonCallback(win, onMouseBtn);
    glfwSetCursorPosCallback(win, onCursorPos);
    glfwSetScrollCallback(win, onScroll);

    // ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::GetIO().IniFilename = nullptr;
    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    applyStyle();

    // fullscreen quad
    GLuint prog = createProgram(QUAD_VERT, QUAD_FRAG);
    float verts[] = {-1, -1, 0, 0, 1, -1, 1, 0, 1,  1, 1, 1,
                     -1, -1, 0, 0, 1, 1,  1, 1, -1, 1, 0, 1};
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                          (void *)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                          (void *)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);

    // HDR target texture, used by both CPU (uploads via glTexSubImage2D) and
    // GPU (writes directly via CUDA-GL interop, zero copies). RGBA32F because
    // CUDA surface objects don't support 3-channel float formats.
    glGenTextures(1, &g.tex);
    glBindTexture(GL_TEXTURE_2D, g.tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Lazily (re)allocate the GL texture whenever the requested render size
    // changes, then re-register it with CUDA so the GPU renderer's mapped
    // resource points at the right storage.
    auto ensureTexture = [](int w, int h) {
        if (w == g.texW && h == g.texH)
            return;
#ifdef FRACTAL_USE_CUDA
        // Detach the CUDA resource BEFORE reallocating the GL storage.
        // glTexImage2D frees the old backing memory; if the CUDA resource is
        // still registered against it, the next cudaGraphicsMapResources
        // dereferences a dangling pointer and the error surfaces (often
        // asynchronously) as "illegal memory access" on the next CUDA call.
        g.gpuRenderer.setOutputTexture(0, 0, 0);
#endif
        glBindTexture(GL_TEXTURE_2D, g.tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT,
                     nullptr);
        g.texW = w;
        g.texH = h;
#ifdef FRACTAL_USE_CUDA
        g.gpuRenderer.setOutputTexture(g.tex, w, h);
#endif
    };

    g.orbit.apply(g.params);

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();

        int winW, winH;
        glfwGetFramebufferSize(win, &winW, &winH);
        glViewport(0, 0, winW, winH);

        // Wall-clock dt for smooth animation (independent of render cost).
        auto nowT = std::chrono::steady_clock::now();
        double dt = 0.0;
        if (g.haveLastFrameTime)
            dt = std::chrono::duration<double>(nowT - g.lastFrameTime).count();
        g.lastFrameTime = nowT;
        g.haveLastFrameTime = true;
        // Clamp pathological dt (e.g. user dragged the title bar): keeps the
        // camera from teleporting after a stall.
        if (dt > 0.25)
            dt = 0.25;

        // Pace the animation off the recording's video clock when active —
        // each captured frame represents exactly 1/recFps of camera motion,
        // independent of how long the render+readback actually took. This
        // eliminates the freeze-then-jump skipping that wall-clock dt produces
        // when the loop occasionally stalls past vsync.
        bool willCapture = g.recorder.isActive() && g.recorder.dueForFrame();
        double animDt;
        if (g.recorder.isActive())
            animDt = willCapture ? (1.0 / (double)g.recorder.fps()) : 0.0;
        else
            animDt = dt;

        if (g.autoRotate && animDt > 0.0) {
            g.autoTime += animDt;
            const float twoPi = 6.2831853f;
            g.orbit.azimuth +=
                (g.autoSpinDeg * 3.14159265f / 180.0f) * (float)animDt;
            if (g.orbit.azimuth > 3.14159265f)
                g.orbit.azimuth -= twoPi;
            if (g.orbit.azimuth < -3.14159265f)
                g.orbit.azimuth += twoPi;
            if (g.autoElevAmp > 0.001f) {
                float w = twoPi / std::max(g.autoElevPeriod, 0.1f);
                g.orbit.elevation =
                    g.autoElevBase + g.autoElevAmp * std::sin(w * (float)g.autoTime);
                g.orbit.elevation = std::clamp(g.orbit.elevation, -1.4f, 1.4f);
            }
            g.orbit.apply(g.params);
            g.dirty = true;
        }

        // Render → texture. GPU path writes directly into the GL texture's
        // GPU memory via CUDA-GL interop (zero-copy). CPU path produces a
        // host buffer and uploads via glTexSubImage2D (no per-frame realloc).
        if (g.dirty) {
            int rw = std::max(1, (int)(winW * g.params.renderScale));
            int rh = std::max(1, (int)(winH * g.params.renderScale));
            ensureTexture(rw, rh);
#ifdef FRACTAL_USE_CUDA
            if (g.useGPU) {
                g.gpuRenderer.render(winW, winH, g.params);
                // Fallback path (no CUDA-GL interop): upload the host mirror.
                if (g.gpuRenderer.needsHostUpload()) {
                    glBindTexture(GL_TEXTURE_2D, g.tex);
                    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                                    g.gpuRenderer.renderWidth(),
                                    g.gpuRenderer.renderHeight(),
                                    GL_RGBA, GL_FLOAT,
                                    g.gpuRenderer.pixels().data());
                }
            } else
#endif
            {
                g.cpuRenderer.render(winW, winH, g.params);
                glBindTexture(GL_TEXTURE_2D, g.tex);
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                                g.cpuRenderer.renderWidth(),
                                g.cpuRenderer.renderHeight(),
                                GL_RGB, GL_FLOAT,
                                g.cpuRenderer.pixels().data());
            }
            g.dirty = false;
        }

        // draw quad (tone-maps and gamma-corrects via QUAD_FRAG)
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(prog);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, g.tex);
        glUniform1i(glGetUniformLocation(prog, "tex"), 0);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);

        // Capture for recording: read back the fractal scene before ImGui
        // overlays the UI. Gated by willCapture so we only do the expensive
        // glReadPixels + pipe write at the recording's frame cadence (no
        // duplicate frames, no wall-clock catch-up bursts).
        if (willCapture) {
            static std::vector<unsigned char> readback;
            int rw = g.recorder.width();
            int rh = g.recorder.height();
            readback.resize((size_t)rw * (size_t)rh * 3);
            glPixelStorei(GL_PACK_ALIGNMENT, 1);
            glReadPixels(0, 0, rw, rh, GL_RGB, GL_UNSIGNED_BYTE,
                         readback.data());
            g.recorder.writeFrame(readback.data());
        }

        // ImGui on top
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        drawUI();
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Service recording start/stop requested from drawUI() this frame.
        if (g.recRequestStart) {
            g.recRequestStart = false;
            std::time_t t = std::time(nullptr);
            std::tm tm_ = *std::localtime(&t);
            char fname[128];
            std::strftime(fname, sizeof(fname),
                          "fractal_%Y%m%d_%H%M%S.mp4", &tm_);
            if (g.recorder.start(winW, winH, g.recFps,
                                  (Recorder::Quality)g.recQuality, fname))
                g.recMessage = std::string("Recording -> ") + fname;
            else
                g.recMessage = std::string("Failed: ") + g.recorder.error();
        }
        if (g.recRequestStop) {
            g.recRequestStop = false;
            int frames = g.recorder.frames();
            std::string p = g.recorder.path();
            g.recorder.stop();
            char msg[256];
            snprintf(msg, sizeof(msg), "Saved %s (%d frames)", p.c_str(),
                     frames);
            g.recMessage = msg;
        }

        glfwSwapBuffers(win);
    }
    // Make sure the pipe is closed if the user quit mid-recording.
    g.recorder.stop();

    glDeleteTextures(1, &g.tex);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(prog);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
    return 0;
}

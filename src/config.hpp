#pragma once
#include <glm/glm.hpp>

enum class FractalType : int { Mandelbulb = 0, Mandelbox = 1, Julia = 2 };

struct MandelbulbParams {
    float power = 8.0f;
    int iterations = 10;
    float bailout = 2.0f;
};

struct MandelboxParams {
    float scale = 2.0f;
    int iterations = 15;
    float foldLimit = 1.0f;
    float minRadius = 0.5f;
    float fixedRadius = 1.0f;
};

struct JuliaParams {
    glm::vec4 c = {-0.2f, 0.6f, 0.2f, 0.0f};
    int iterations = 15;
};

struct CameraParams {
    glm::vec3 position = {0.0f, 0.0f, 3.0f};
    glm::vec3 target = {0.0f, 0.0f, 0.0f};
    glm::vec3 up = {0.0f, 1.0f, 0.0f};
    float fov = 60.0f;
};

struct LightParams {
    glm::vec3 direction = {0.371f, 0.743f, 0.557f}; // pre-normalized (1,2,1.5)
    glm::vec3 color = {1.0f, 0.95f, 0.85f};
    float ambient = 0.05f;
    float diffuse = 1.0f;
    float specular = 0.5f;
    float shininess = 32.0f;
    bool softShadows = true;
    float shadowK = 8.0f;
    bool aoEnabled = true;
    int aoSteps = 5;
    float aoStepSize = 0.05f;
};

struct ColorParams {
    glm::vec3 paletteA = {0.3f, 0.5f, 1.0f};
    glm::vec3 paletteB = {1.0f, 0.3f, 0.1f};
    glm::vec3 bgTop = {0.01f, 0.01f, 0.03f};
    glm::vec3 bgBottom = {0.04f, 0.04f, 0.08f};
    bool orbitTrap = true;
    float trapScale = 1.0f;
    float glowStr = 0.3f;
};

struct RenderParams {
    FractalType fractalType = FractalType::Mandelbulb;
    MandelbulbParams mandelbulb;
    MandelboxParams mandelbox;
    JuliaParams julia;
    CameraParams camera;
    LightParams light;
    ColorParams color;
    int maxSteps = 128;
    float maxDist = 20.0f;
    float epsilon = 0.001f;
    float renderScale = 0.5f;
};

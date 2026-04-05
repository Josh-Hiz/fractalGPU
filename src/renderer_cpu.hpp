#pragma once
#include "camera.hpp"
#include "config.hpp"
#include <glm/glm.hpp>
#include <vector>

class CPURenderer {
  public:
    // Render into internal pixel buffer at (windowW * renderScale) resolution
    void render(int windowW, int windowH, const RenderParams &params);

    const std::vector<float> &pixels() const { return m_pixels; }
    int renderWidth() const { return m_width; }
    int renderHeight() const { return m_height; }
    double renderMs() const { return m_ms; }

  private:
    std::vector<float> m_pixels; // packed RGB floats (HDR)
    int m_width = 0;
    int m_height = 0;
    double m_ms = 0.0;

    glm::vec3 shadePixel(int px, int py, const RenderParams &p);
    float march(const Ray &ray, const RenderParams &p, float &trap,
                float &minD);
    glm::vec3 calcNormal(const glm::vec3 &pos, const RenderParams &p);
    float softShadow(const glm::vec3 &pos, const glm::vec3 &ldir,
                     const RenderParams &p);
    float ambientOcc(const glm::vec3 &pos, const glm::vec3 &nor,
                     const RenderParams &p);
    glm::vec3 cosPalette(float t, const glm::vec3 &a, const glm::vec3 &b);
    glm::vec3 background(const glm::vec3 &dir, float minD,
                         const RenderParams &p);
};

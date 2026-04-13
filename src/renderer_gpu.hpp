#pragma once
#include "config.hpp"
#include <vector>

// GPU (CUDA) renderer, same interface as CPURenderer.
// Only available when compiled with FRACTAL_USE_CUDA defined.
class GPURenderer {
  public:
    ~GPURenderer();

    // Render into an internal pixel buffer at (windowW * renderScale) resolution.
    void render(int windowW, int windowH, const RenderParams &params);

    const std::vector<float> &pixels() const { return m_pixels; }
    int renderWidth() const { return m_width; }
    int renderHeight() const { return m_height; }
    double renderMs() const { return m_ms; }

  private:
    std::vector<float> m_pixels; // packed RGB floats (HDR), host copy
    float *d_pixels = nullptr;   // device buffer
    size_t d_pixels_bytes = 0;   // current allocation size in bytes
    int m_width = 0;
    int m_height = 0;
    double m_ms = 0.0;
};

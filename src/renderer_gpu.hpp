#pragma once
#include "config.hpp"
#include <vector>

// GPU (CUDA) renderer with two output paths:
//
//   * Fast path — CUDA-OpenGL interop: the kernel writes pixels directly into
//     a registered GL texture's GPU memory, eliminating the device→host→device
//     roundtrip entirely.
//
//   * Fallback path — device buffer + host upload: when interop registration
//     fails (most commonly hybrid-graphics laptops where GL runs on the iGPU
//     and CUDA on the dGPU), the kernel writes into a CUDA-allocated device
//     buffer; we cudaMemcpy it to a host vector and the caller uploads via
//     glTexSubImage2D.
//
// Only available when compiled with FRACTAL_USE_CUDA defined.

// Forward-declare opaque CUDA handle so callers don't need the CUDA headers.
struct cudaGraphicsResource;

class GPURenderer {
  public:
    ~GPURenderer();

    // Register a GL texture (must be GL_RGBA32F, surface load/store capable)
    // as the kernel's render target. Pass 0 to detach. Re-call when the
    // texture is reallocated (e.g. window/resolution change). If interop
    // registration fails, transparently switches to the host-upload fallback.
    void setOutputTexture(unsigned int glTex, int width, int height);

    // Render into the registered target.
    void render(int windowW, int windowH, const RenderParams &params);

    int renderWidth() const { return m_width; }
    int renderHeight() const { return m_height; }
    double renderMs() const { return m_ms; }

    // True when the renderer is in fallback mode and the caller must upload
    // pixels() to the GL texture itself (RGBA float layout, width*height*4).
    bool needsHostUpload() const { return !m_useInterop; }
    const std::vector<float> &pixels() const { return m_pixels; }

  private:
    cudaGraphicsResource *m_glRes = nullptr; // CUDA-mapped handle to m_glTex
    unsigned int m_glTex = 0;
    int m_width = 0;
    int m_height = 0;
    double m_ms = 0.0;

    bool m_useInterop = false;

    // Fallback storage (only populated when m_useInterop == false).
    void *m_dPixels = nullptr;       // device buffer of float4 pixels
    size_t m_dPixelsBytes = 0;
    std::vector<float> m_pixels;     // host mirror, RGBA32F, length w*h*4
};

// Print device properties, kernel attributes (regs/thread, smem), and
// theoretical occupancy for the bench harness. Implemented in renderer_gpu.cu
// because the kernels have internal linkage there.
void gpu_print_diagnostics(int blockX, int blockY, int volSteps);

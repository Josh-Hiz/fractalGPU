// GPU fractal renderer — CUDA implementation.
//
// All SDF, marching, and shading logic is reimplemented here as __device__
// functions so they run entirely on the GPU.  The public render() call launches
// a 2-D kernel (one thread per pixel), then copies the HDR float buffer back
// to host for the existing OpenGL upload path.

// The system GLM only knows CUDA up to 8.0.  When GLM_FORCE_CUDA is defined
// GLM skips including cuda.h, so CUDA_VERSION is never set and its version
// check fires.  Synthesise it from nvcc's own macros before GLM sees it.
#ifdef __CUDACC__
#  define CUDA_VERSION (__CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10)
#endif
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <chrono>
#include <cstdio>
#include <vector>

#include "config.hpp"
#include "renderer_gpu.hpp"

// ---------------------------------------------------------------------------
// CUDA error helpers
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d  %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                  \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// GLM device helpers, scalar math replacements for std:: that work on device
// ---------------------------------------------------------------------------

__device__ static inline float d_clamp(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}

__device__ static inline glm::vec3 d_clampv(glm::vec3 v, float lo, float hi) {
    return glm::vec3(d_clamp(v.x, lo, hi), d_clamp(v.y, lo, hi),
                     d_clamp(v.z, lo, hi));
}

__device__ static inline float d_mix(float a, float b, float t) {
    return a + t * (b - a);
}

__device__ static inline glm::vec3 d_mixv(glm::vec3 a, glm::vec3 b, float t) {
    return glm::vec3(d_mix(a.x, b.x, t), d_mix(a.y, b.y, t),
                     d_mix(a.z, b.z, t));
}

// ---------------------------------------------------------------------------
// Device SDF functions
// (same algorithms as sdf.hpp but using device safe math)
// ---------------------------------------------------------------------------

__device__ static float sdf_mandelbulb(glm::vec3 pos, const MandelbulbParams &p,
                                        float &trap) {
    glm::vec3 z = pos;
    float dr = 1.0f;
    float r = 0.0f;
    trap = 1e10f;

    for (int i = 0; i < p.iterations; ++i) {
        r = glm::length(z);
        if (r > p.bailout)
            break;
        if (r < 1e-8f) {
            trap = 0.0f;
            return 0.0f;
        }

        float theta = acosf(d_clamp(z.z / r, -1.0f, 1.0f));
        float phi = atan2f(z.y, z.x);
        dr = powf(r, p.power - 1.0f) * p.power * dr + 1.0f;

        float zr = powf(r, p.power);
        theta *= p.power;
        phi *= p.power;

        z = zr * glm::vec3(__sinf(theta) * __cosf(phi),
                            __sinf(theta) * __sinf(phi), __cosf(theta));
        z += pos;
        trap = fminf(trap, r);
    }

    if (r < 1e-8f || dr < 1e-8f)
        return 0.0f;
    return 0.5f * logf(r) * r / dr;
}

__device__ static float sdf_mandelbox(glm::vec3 pos, const MandelboxParams &p,
                                       float &trap) {
    glm::vec3 z = pos;
    float dr = 1.0f;
    trap = 1e10f;

    const float minR2 = p.minRadius * p.minRadius;
    const float fixedR2 = p.fixedRadius * p.fixedRadius;

    for (int i = 0; i < p.iterations; ++i) {
        // box fold
        z = d_clampv(z, -p.foldLimit, p.foldLimit) * 2.0f - z;

        // sphere fold
        float r2 = glm::dot(z, z);
        trap = fminf(trap, r2);

        if (r2 < minR2) {
            float k = fixedR2 / minR2;
            z *= k;
            dr *= k;
        } else if (r2 < fixedR2) {
            float k = fixedR2 / r2;
            z *= k;
            dr *= k;
        }

        z = z * p.scale + pos;
        dr = dr * fabsf(p.scale) + 1.0f;
    }

    return glm::length(z) / fabsf(dr);
}

__device__ static float sdf_julia(glm::vec3 pos, const JuliaParams &p,
                                   float &trap) {
    glm::vec4 z = glm::vec4(pos, 0.0f);
    float dz = 1.0f;
    trap = 1e10f;

    for (int i = 0; i < p.iterations; ++i) {
        float r2 = glm::dot(z, z);
        if (r2 < 1e-10f) {
            trap = 0.0f;
            return 0.0f;
        }

        trap = fminf(trap, r2);

        if (r2 > 4.0f) {
            float r = sqrtf(r2);
            if (dz < 1e-10f)
                return 0.0f;
            return 0.5f * r * logf(r) / dz;
        }

        dz *= 2.0f * sqrtf(r2);

        z = glm::vec4(z.x * z.x - z.y * z.y - z.z * z.z - z.w * z.w,
                      2.0f * z.x * z.y, 2.0f * z.x * z.z,
                      2.0f * z.x * z.w) +
            p.c;
    }

    return 0.0f;
}

__device__ static float sdf_evaluate(const glm::vec3 &pos,
                                      const RenderParams &p, float &trap) {
    switch (p.fractalType) {
    case FractalType::Mandelbulb:
        return sdf_mandelbulb(pos, p.mandelbulb, trap);
    case FractalType::Mandelbox:
        return sdf_mandelbox(pos, p.mandelbox, trap);
    case FractalType::Julia:
        return sdf_julia(pos, p.julia, trap);
    }
    return 1.0f;
}

// ---------------------------------------------------------------------------
// Device ray helpers
// ---------------------------------------------------------------------------

struct DevRay {
    glm::vec3 origin;
    glm::vec3 dir;
};

__device__ static DevRay makeRay(int px, int py, int width, int height,
                                  const CameraParams &cam) {
    float aspect = (float)width / (float)height;
    float tanHalfFov = tanf(cam.fov * glm::pi<float>() / 360.0f);

    glm::vec3 fwd = glm::normalize(cam.target - cam.position);
    glm::vec3 right = glm::normalize(glm::cross(fwd, cam.up));
    glm::vec3 up = glm::cross(right, fwd);

    float u = (2.0f * ((float)px + 0.5f) / (float)width - 1.0f) * aspect * tanHalfFov;
    float v = (1.0f - 2.0f * ((float)py + 0.5f) / (float)height) * tanHalfFov;

    return {cam.position, glm::normalize(fwd + u * right + v * up)};
}

// ---------------------------------------------------------------------------
// Device ray marching + shading
// ---------------------------------------------------------------------------

// Returns hit distance t, or -1 on miss.
__device__ static float march(const DevRay &ray, const RenderParams &p,
                               float &trap, float &minD) {
    float t = 0.001f;
    trap = 1e10f;
    minD = 1e10f;

    for (int i = 0; i < p.maxSteps; ++i) {
        glm::vec3 pos = ray.origin + ray.dir * t;
        float localTrap;
        float d = sdf_evaluate(pos, p, localTrap);

        trap = fminf(trap, localTrap);
        minD = fminf(minD, d);

        if (d < p.epsilon)
            return t;
        if (t > p.maxDist)
            break;
        t += fmaxf(d, p.epsilon * 0.5f);
    }
    return -1.0f;
}

// Central difference numerical gradient of the SDF.
__device__ static glm::vec3 calcNormal(const glm::vec3 &pos,
                                        const RenderParams &p) {
    float e = p.epsilon * 2.0f;
    float dummy;
    return glm::normalize(glm::vec3(
        sdf_evaluate(pos + glm::vec3(e, 0, 0), p, dummy) -
            sdf_evaluate(pos - glm::vec3(e, 0, 0), p, dummy),
        sdf_evaluate(pos + glm::vec3(0, e, 0), p, dummy) -
            sdf_evaluate(pos - glm::vec3(0, e, 0), p, dummy),
        sdf_evaluate(pos + glm::vec3(0, 0, e), p, dummy) -
            sdf_evaluate(pos - glm::vec3(0, 0, e), p, dummy)));
}

// Penumbra soft shadow.
__device__ static float softShadow(const glm::vec3 &pos, const glm::vec3 &ldir,
                                    const RenderParams &p) {
    if (!p.light.softShadows) {
        float dummy;
        DevRay sr{pos + ldir * 0.02f, ldir};
        return (march(sr, p, dummy, dummy) > 0.0f) ? 0.0f : 1.0f;
    }
    float res = 1.0f;
    float t = 0.02f;
    for (int i = 0; i < 32; ++i) {
        float dummy;
        float d = sdf_evaluate(pos + ldir * t, p, dummy);
        if (d < p.epsilon)
            return 0.0f;
        res = fminf(res, p.light.shadowK * d / t);
        t += d_clamp(d, 0.01f, 0.2f);
        if (t > 8.0f)
            break;
    }
    return d_clamp(res, 0.0f, 1.0f);
}

// Step based ambient occlusion along the surface normal.
__device__ static float ambientOcc(const glm::vec3 &pos, const glm::vec3 &nor,
                                    const RenderParams &p) {
    if (!p.light.aoEnabled)
        return 1.0f;
    float occ = 0.0f;
    float scale = 1.0f;
    for (int i = 1; i <= p.light.aoSteps; ++i) {
        float d = (float)i * p.light.aoStepSize;
        float dummy;
        float dist = sdf_evaluate(pos + nor * d, p, dummy);
        occ += (d - dist) * scale;
        scale *= 0.5f;
    }
    return d_clamp(1.0f - 2.0f * occ, 0.0f, 1.0f);
}

// Cosine palette (Inigo Quilez).
__device__ static glm::vec3 cosPalette(float t, const glm::vec3 &a,
                                        const glm::vec3 &b) {
    float f = 0.5f + 0.5f * __cosf(glm::pi<float>() * d_clamp(t, 0.0f, 1.0f));
    return d_mixv(b, a, f);
}

// Background: vertical gradient + glow.
__device__ static glm::vec3 background(const glm::vec3 &dir, float minD,
                                        const RenderParams &p) {
    float fade = 0.5f * (dir.y + 1.0f);
    glm::vec3 bg = d_mixv(p.color.bgBottom, p.color.bgTop, fade);
    float glow = __expf(-18.0f * minD) * p.color.glowStr;
    bg += d_mixv(p.color.paletteA, p.color.paletteB, 0.5f) * glow;
    return bg;
}

// Full per pixel shading.
__device__ static glm::vec3 shadePixel(int px, int py, int width, int height,
                                        const RenderParams &p) {
    DevRay ray = makeRay(px, py, width, height, p.camera);
    float trap, minD;
    float t = march(ray, p, trap, minD);

    if (t < 0.0f)
        return background(ray.dir, minD, p);

    glm::vec3 pos = ray.origin + ray.dir * t;
    glm::vec3 nor = calcNormal(pos, p);
    glm::vec3 viewDir = -ray.dir;
    glm::vec3 ldir = glm::normalize(p.light.direction);

    float diff = fmaxf(glm::dot(nor, ldir), 0.0f);
    glm::vec3 halfVec = glm::normalize(ldir + viewDir);
    float spec = powf(fmaxf(glm::dot(nor, halfVec), 0.0f), p.light.shininess);

    float shadow = softShadow(pos + nor * p.epsilon * 3.0f, ldir, p);
    float ao = ambientOcc(pos, nor, p);

    float trapT = p.color.orbitTrap
                      ? d_clamp(trap * p.color.trapScale, 0.0f, 1.0f)
                      : 0.5f;
    glm::vec3 surfCol = cosPalette(trapT, p.color.paletteA, p.color.paletteB);

    glm::vec3 ambientCol = p.light.ambient * surfCol * ao;
    glm::vec3 diffuseCol = p.light.diffuse * diff * shadow * p.light.color * surfCol;
    glm::vec3 specularCol = p.light.specular * spec * shadow * p.light.color;

    glm::vec3 col = ambientCol + diffuseCol + specularCol;

    // subtle distance fog
    col = d_mixv(p.color.bgTop, col, __expf(-t * 0.04f));

    return col;
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

__global__ static void renderKernel(float *pixels, int width, int height,
                                     RenderParams params) {
    int px = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int py = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (px >= width || py >= height)
        return;

    glm::vec3 col = shadePixel(px, py, width, height, params);

    int idx = (py * width + px) * 3;
    pixels[idx + 0] = col.r;
    pixels[idx + 1] = col.g;
    pixels[idx + 2] = col.b;
}

// ---------------------------------------------------------------------------
// GPURenderer implementation
// ---------------------------------------------------------------------------

GPURenderer::~GPURenderer() {
    if (d_pixels)
        CUDA_CHECK(cudaFree(d_pixels));
}

void GPURenderer::render(int winW, int winH, const RenderParams &params) {
    m_width = glm::max(1, (int)(winW * params.renderScale));
    m_height = glm::max(1, (int)(winH * params.renderScale));

    size_t needed = (size_t)m_width * m_height * 3 * sizeof(float);
    if (needed > d_pixels_bytes) {
        if (d_pixels)
            CUDA_CHECK(cudaFree(d_pixels));
        CUDA_CHECK(cudaMalloc(&d_pixels, needed));
        d_pixels_bytes = needed;
    }

    // CUDA events for accurate GPU timing
    cudaEvent_t evStart, evStop;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));

    dim3 block(16, 16);
    dim3 grid((m_width + block.x - 1) / block.x,
              (m_height + block.y - 1) / block.y);

    CUDA_CHECK(cudaEventRecord(evStart));
    renderKernel<<<grid, block>>>(d_pixels, m_width, m_height, params);
    CUDA_CHECK(cudaEventRecord(evStop));
    CUDA_CHECK(cudaEventSynchronize(evStop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, evStart, evStop));
    m_ms = (double)ms;

    CUDA_CHECK(cudaEventDestroy(evStart));
    CUDA_CHECK(cudaEventDestroy(evStop));

    m_pixels.resize(m_width * m_height * 3);
    CUDA_CHECK(cudaMemcpy(m_pixels.data(), d_pixels, needed,
                          cudaMemcpyDeviceToHost));
}

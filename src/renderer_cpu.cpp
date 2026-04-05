#include "renderer_cpu.hpp"
#include "sdf.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <glm/gtc/constants.hpp>

// Cosine palette (Inigo Quilez): smooth gradient between two colors driven by t
glm::vec3 CPURenderer::cosPalette(float t, const glm::vec3 &a,
                                  const glm::vec3 &b) {
    float f =
        0.5f + 0.5f * std::cos(glm::pi<float>() * std::clamp(t, 0.0f, 1.0f));
    return glm::mix(b, a, f);
}

// Background: vertical gradient + glow emitted toward nearby fractal surface
glm::vec3 CPURenderer::background(const glm::vec3 &dir, float minD,
                                  const RenderParams &p) {
    float fade = 0.5f * (dir.y + 1.0f);
    glm::vec3 bg = glm::mix(p.color.bgBottom, p.color.bgTop, fade);
    float glow = std::exp(-18.0f * minD) * p.color.glowStr;
    bg += glm::mix(p.color.paletteA, p.color.paletteB, 0.5f) * glow;
    return bg;
}

// ray marching

// Returns hit distance t, or -1 if no hit.
// trap: orbit trap value for coloring; minD: minimum SDF value seen (drives
// glow)
float CPURenderer::march(const Ray &ray, const RenderParams &p, float &trap,
                         float &minD) {
    float t = 0.001f;
    trap = 1e10f;
    minD = 1e10f;

    for (int i = 0; i < p.maxSteps; ++i) {
        glm::vec3 pos = ray.origin + ray.dir * t;
        float localTrap;
        float d = SDF::evaluate(pos, p, localTrap);

        trap = std::min(trap, localTrap);
        minD = std::min(minD, d);

        if (d < p.epsilon)
            return t; // hit
        if (t > p.maxDist)
            break; // escaped
        t += std::max(
            d, p.epsilon * 0.5f); // min step avoids infinite loop near interior
    }
    return -1.0f; // miss
}

// Central-difference numerical gradient of the SDF
glm::vec3 CPURenderer::calcNormal(const glm::vec3 &pos, const RenderParams &p) {
    float e = p.epsilon * 2.0f;
    float dummy;
    return glm::normalize(
        glm::vec3(SDF::evaluate(pos + glm::vec3(e, 0, 0), p, dummy) -
                      SDF::evaluate(pos - glm::vec3(e, 0, 0), p, dummy),
                  SDF::evaluate(pos + glm::vec3(0, e, 0), p, dummy) -
                      SDF::evaluate(pos - glm::vec3(0, e, 0), p, dummy),
                  SDF::evaluate(pos + glm::vec3(0, 0, e), p, dummy) -
                      SDF::evaluate(pos - glm::vec3(0, 0, e), p, dummy)));
}

// Penumbra soft shadow: march toward light, accumulate proximity to surfaces
float CPURenderer::softShadow(const glm::vec3 &pos, const glm::vec3 &ldir,
                              const RenderParams &p) {
    if (!p.light.softShadows) {
        // hard shadow: check for any occluder
        float dummy;
        Ray sr{pos + ldir * 0.02f, ldir};
        return (march(sr, p, dummy, dummy) > 0.0f) ? 0.0f : 1.0f;
    }
    float res = 1.0f;
    float t = 0.02f;
    for (int i = 0; i < 32; ++i) {
        float dummy;
        float d = SDF::evaluate(pos + ldir * t, p, dummy);
        if (d < p.epsilon)
            return 0.0f;
        res = std::min(res, p.light.shadowK * d / t);
        t += std::clamp(d, 0.01f, 0.2f);
        if (t > 8.0f)
            break;
    }
    return std::clamp(res, 0.0f, 1.0f);
}

// Step-based ambient occlusion along the surface normal
float CPURenderer::ambientOcc(const glm::vec3 &pos, const glm::vec3 &nor,
                              const RenderParams &p) {
    if (!p.light.aoEnabled)
        return 1.0f;
    float occ = 0.0f;
    float scale = 1.0f;
    for (int i = 1; i <= p.light.aoSteps; ++i) {
        float d = i * p.light.aoStepSize;
        float dummy;
        float dist = SDF::evaluate(pos + nor * d, p, dummy);
        occ += (d - dist) * scale;
        scale *= 0.5f;
    }
    return std::clamp(1.0f - 2.0f * occ, 0.0f, 1.0f);
}

// shading
glm::vec3 CPURenderer::shadePixel(int px, int py, const RenderParams &p) {
    Ray ray = makeRay(px, py, m_width, m_height, p.camera);
    float trap, minD;
    float t = march(ray, p, trap, minD);

    if (t < 0.0f) {
        return background(ray.dir, minD, p);
    }

    glm::vec3 pos = ray.origin + ray.dir * t;
    glm::vec3 nor = calcNormal(pos, p);
    glm::vec3 viewDir = -ray.dir;
    glm::vec3 ldir = glm::normalize(p.light.direction);

    // lighting terms
    float diff = std::max(glm::dot(nor, ldir), 0.0f);
    glm::vec3 halfVec = glm::normalize(ldir + viewDir);
    float spec =
        std::pow(std::max(glm::dot(nor, halfVec), 0.0f), p.light.shininess);

    float shadow = softShadow(pos + nor * p.epsilon * 3.0f, ldir, p);
    float ao = ambientOcc(pos, nor, p);

    // orbit trap drives surface color through the cosine palette
    float trapT = p.color.orbitTrap
                      ? std::clamp(trap * p.color.trapScale, 0.0f, 1.0f)
                      : 0.5f;
    glm::vec3 surfCol = cosPalette(trapT, p.color.paletteA, p.color.paletteB);

    glm::vec3 ambient = p.light.ambient * surfCol * ao;
    glm::vec3 diffuse =
        p.light.diffuse * diff * shadow * p.light.color * surfCol;
    glm::vec3 specular = p.light.specular * spec * shadow * p.light.color;

    glm::vec3 col = ambient + diffuse + specular;

    // subtle distance fog — thins out detail on far surfaces
    col = glm::mix(p.color.bgTop, col, std::exp(-t * 0.04f));

    return col;
}

// public render
void CPURenderer::render(int winW, int winH, const RenderParams &params) {
    m_width = std::max(1, (int)(winW * params.renderScale));
    m_height = std::max(1, (int)(winH * params.renderScale));
    m_pixels.resize(m_width * m_height * 3);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int y = 0; y < m_height; ++y) {
        for (int x = 0; x < m_width; ++x) {
            glm::vec3 col = shadePixel(x, y, params);
            int idx = (y * m_width + x) * 3;
            m_pixels[idx] = col.r;
            m_pixels[idx + 1] = col.g;
            m_pixels[idx + 2] = col.b;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    m_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
}

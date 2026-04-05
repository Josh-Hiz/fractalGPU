#pragma once
#include "config.hpp"
#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>
#include <limits>

namespace SDF {

// Mandelbulb
inline float mandelbulb(glm::vec3 pos, const MandelbulbParams &p, float &trap) {
    glm::vec3 z = pos;
    float dr = 1.0f;
    float r = 0.0f;
    trap = std::numeric_limits<float>::max();

    for (int i = 0; i < p.iterations; ++i) {
        r = glm::length(z);
        if (r > p.bailout)
            break;
        if (r < 1e-8f) {
            trap = 0.0f;
            return 0.0f;
        }

        float theta = std::acos(std::clamp(z.z / r, -1.0f, 1.0f));
        float phi = std::atan2(z.y, z.x);
        dr = std::pow(r, p.power - 1.0f) * p.power * dr + 1.0f;

        float zr = std::pow(r, p.power);
        theta *= p.power;
        phi *= p.power;

        z = zr * glm::vec3(std::sin(theta) * std::cos(phi),
                           std::sin(theta) * std::sin(phi), std::cos(theta));
        z += pos;
        trap = std::min(trap, r);
    }

    if (r < 1e-8f || dr < 1e-8f)
        return 0.0f;
    return 0.5f * std::log(r) * r / dr;
}

// Mandelbox
inline float mandelbox(glm::vec3 pos, const MandelboxParams &p, float &trap) {
    glm::vec3 z = pos;
    float dr = 1.0f;
    trap = std::numeric_limits<float>::max();

    const float minR2 = p.minRadius * p.minRadius;
    const float fixedR2 = p.fixedRadius * p.fixedRadius;

    for (int i = 0; i < p.iterations; ++i) {
        // box fold
        z = glm::clamp(z, -p.foldLimit, p.foldLimit) * 2.0f - z;

        // sphere fold
        float r2 = glm::dot(z, z);
        trap = std::min(trap, r2);

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
        dr = dr * std::abs(p.scale) + 1.0f;
    }

    return glm::length(z) / std::abs(dr);
}

// Quaternion Julia set
inline float julia(glm::vec3 pos, const JuliaParams &p, float &trap) {
    glm::vec4 z = glm::vec4(pos, 0.0f);
    float dz = 1.0f;
    trap = std::numeric_limits<float>::max();

    for (int i = 0; i < p.iterations; ++i) {
        float r2 = glm::dot(z, z);
        if (r2 < 1e-10f) {
            trap = 0.0f;
            return 0.0f;
        }

        trap = std::min(trap, r2);

        if (r2 > 4.0f) {
            float r = std::sqrt(r2);
            if (dz < 1e-10f)
                return 0.0f;
            return 0.5f * r * std::log(r) / dz;
        }

        // |dz| *= 2 * |z| (derivative of z^2 is 2z)
        dz *= 2.0f * std::sqrt(r2);

        // quaternion square: z = z^2 + c
        z = glm::vec4(z.x * z.x - z.y * z.y - z.z * z.z - z.w * z.w,
                      2.0f * z.x * z.y, 2.0f * z.x * z.z, 2.0f * z.x * z.w) +
            p.c;
    }

    return 0.0f; // inside the set
}

// Dispatch
inline float evaluate(const glm::vec3 &pos, const RenderParams &p,
                      float &trap) {
    switch (p.fractalType) {
    case FractalType::Mandelbulb:
        return mandelbulb(pos, p.mandelbulb, trap);
    case FractalType::Mandelbox:
        return mandelbox(pos, p.mandelbox, trap);
    case FractalType::Julia:
        return julia(pos, p.julia, trap);
    }
    return 1.0f;
}

} // namespace SDF

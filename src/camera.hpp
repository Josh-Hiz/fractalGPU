#pragma once
#include <glm/glm.hpp>
#include <cmath>
#include "config.hpp"

struct Ray {
    glm::vec3 origin;
    glm::vec3 dir;
};

inline Ray makeRay(int px, int py, int width, int height, const CameraParams& cam) {
    float aspect      = (float)width / (float)height;
    float tanHalfFov  = std::tan(cam.fov * 3.14159265f / 360.0f); // fov/2 in radians

    glm::vec3 fwd   = glm::normalize(cam.target - cam.position);
    glm::vec3 right = glm::normalize(glm::cross(fwd, cam.up));
    glm::vec3 up    = glm::cross(right, fwd);

    // NDC: x in [-1,1] (scaled by aspect), y in [-1,1]
    float u = (2.0f * (px + 0.5f) / width  - 1.0f) * aspect * tanHalfFov;
    float v = (1.0f - 2.0f * (py + 0.5f) / height) * tanHalfFov;

    return { cam.position, glm::normalize(fwd + u * right + v * up) };
}

#pragma once

// Fullscreen quad — just passes UV through
static const char* QUAD_VERT = R"glsl(
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;
out vec2 uv;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    uv = aUV;
}
)glsl";

// Displays the CPU-rendered texture with ACES tone mapping + gamma correction
static const char* QUAD_FRAG = R"glsl(
#version 330 core
in vec2 uv;
out vec4 FragColor;
uniform sampler2D tex;

// ACES filmic tone mapping (approximation by Krzysztof Narkowicz)
vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

void main() {
    vec3 col = texture(tex, uv).rgb;
    col = aces(col);
    col = pow(col, vec3(1.0 / 2.2)); // gamma correction
    FragColor = vec4(col, 1.0);
}
)glsl";

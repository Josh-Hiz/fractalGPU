cmake_minimum_required(VERSION 3.22.0)

project(FractalGPU VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 20)

# Building static library for imgui
add_library(imgui STATIC
    external/imgui/imgui_demo.cpp
    external/imgui/imgui_draw.cpp
    external/imgui/imgui_tables.cpp
    external/imgui/imgui_widgets.cpp
    external/imgui/imgui.cpp
    external/imgui/backends/imgui_impl_glfw.cpp
    external/imgui/backends/imgui_impl_opengl3.cpp
)

# imgui needs imgui and imgui backend
target_include_directories(imgui PUBLIC
    external/imgui
    external/imgui/backends
    external/glad/include
)

target_compile_definitions(imgui PUBLIC
    IMGUI_IMPL_OPENGL_LOADER_GLAD  # tell ImGui's backend to use GLAD explicitly
)

# target glfw for imgui
find_package(glfw3 REQUIRED)
target_link_libraries(imgui PUBLIC glfw)

# Source files
add_executable(FractalGPU 
    src/main.cpp
    external/glad/src/glad.c # glad C file   
)

# OpenGL shared library
set(OpenGL_GL_PREFERENCE "GLVND")
find_package(OpenGL REQUIRED)

target_include_directories(FractalGPU PRIVATE
    external/glad/include
)

target_link_libraries(FractalGPU PRIVATE
    OpenGL::GL
    glfw
    imgui
    ${CMAKE_DL_LIBS}
)

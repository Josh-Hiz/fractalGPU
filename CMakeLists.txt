cmake_minimum_required(VERSION 3.10.0)

project(FractalGPU VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 20)

add_executable(FractalGPU 
    src/main.cpp
    external/glad/src/glad.c # glad C file   
)

# OpenGL shared library
set(OpenGL_GL_PREFERENCE "GLVND")
find_package(OpenGL REQUIRED)
target_link_libraries(FractalGPU PRIVATE OpenGL::GL)

# GLFW shared library
find_package(glfw3 REQUIRED)
target_link_libraries(FractalGPU PRIVATE glfw)

# GLAD headers
target_include_directories(FractalGPU PRIVATE external/glad/include)

cmake_minimum_required(VERSION 3.18.0)
project(FractalGPU VERSION 1.0.0)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(imgui STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/imgui.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/imgui_demo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/imgui_draw.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/imgui_tables.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/imgui_widgets.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/backends/imgui_impl_glfw.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/backends
    ${CMAKE_CURRENT_SOURCE_DIR}/external/glad/include
)
target_compile_definitions(imgui PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLAD)
find_package(glfw3 REQUIRED)
target_link_libraries(imgui PUBLIC glfw)

add_executable(FractalGPU
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer_cpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/external/glad/src/glad.c
)

set(OpenGL_GL_PREFERENCE "GLVND")
find_package(OpenGL REQUIRED)

target_include_directories(FractalGPU PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/external/glad/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(FractalGPU PRIVATE
    OpenGL::GL
    glfw
    imgui
    ${CMAKE_DL_LIBS}
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(FractalGPU PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>
    )
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(FractalGPU PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
    )
endif()

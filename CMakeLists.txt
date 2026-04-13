cmake_minimum_required(VERSION 3.18.0)

option(FRACTAL_ENABLE_CUDA "Build with CUDA GPU rendering support" OFF)

# Enable CUDA language before project() so the compiler is found early
if(FRACTAL_ENABLE_CUDA)
    project(FractalGPU VERSION 1.0.0 LANGUAGES C CXX CUDA)
else()
    project(FractalGPU VERSION 1.0.0 LANGUAGES C CXX)
endif()

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
    # glad lives here so it isn't dropped by CMake's CUDA separable-link rewrite
    ${CMAKE_CURRENT_SOURCE_DIR}/external/glad/src/glad.c
)
target_include_directories(imgui PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/backends
    ${CMAKE_CURRENT_SOURCE_DIR}/external/glad/include
)
target_compile_definitions(imgui PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLAD)
find_package(glfw3 REQUIRED)
target_link_libraries(imgui PUBLIC glfw)

set(FRACTAL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer_cpu.cpp
)

if(FRACTAL_ENABLE_CUDA)
    list(APPEND FRACTAL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer_gpu.cu)
endif()

add_executable(FractalGPU ${FRACTAL_SOURCES})

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

if(FRACTAL_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    target_compile_definitions(FractalGPU PRIVATE FRACTAL_USE_CUDA)
    target_link_libraries(FractalGPU PRIVATE CUDA::cudart)

    set_target_properties(FractalGPU PROPERTIES
        CUDA_ARCHITECTURES native   # auto-detect installed GPU; override with -DCMAKE_CUDA_ARCHITECTURES=89 etc.
    )

    target_compile_options(FractalGPU PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
    )
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(FractalGPU PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>
    )
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(FractalGPU PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
    )
endif()

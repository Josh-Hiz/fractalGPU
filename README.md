# Fractal GPU

A complex parallelized 3D fractal renderer and engine implemented in C++ and CUDA.

Created by Joshua Hizgiaev and Marcos Traverso

## Building

Run the following to install GLFW3, OpenGL with Mesa (open-source implementation for Linux), the math library for OpenGL.

```sh
sudo apt instal libgl1-mesa-dev libglu1-mesa-dev libglfw3-dev libglm-dev mesa-utils
```

These libraries will be statically linked within CMake besides GLM, which is just headers.

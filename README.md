# Fractal GPU

A complex parallelized 3D fractal renderer and engine implemented in C++ and CUDA.

Created by Joshua Hizgiaev and Marcos Traverso

## Building

Run the following to install GLFW3, OpenGL with Mesa (open-source implementation for Linux), and the math library for OpenGL.

```sh
sudo apt instal libgl1-mesa-dev libglu1-mesa-dev libglfw3-dev libglm-dev mesa-utils
```

These libraries will be dynamically linked within CMake besides GLM, which is just headers.

On VSCode it should automatically build, however to build manually:

```sh
cmake -B build
make
./FractalGPU
```

If you want to use ninja generator (reccommended):

```sh
sudo apt install ninja-build
cmake -B build -G Ninja
ninja
./FractalGPU
```

## CPU Results (Graphically)

Here are the results of our CPU ray-marcher, real-time statistics and performance will be provided in a comprehensive report once CUDA and statistics gather functionality are fully implemented.

### Mandelbulb Render

![bulb](./assets/cpu_mandelbulb.png)

### Mandelbox Render

![bulb](./assets/cpu_mandelbox.png)

### Julia Set Quaternion Render

![bulb](./assets/cpu_julia.png)

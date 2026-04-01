#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <iostream>

int main() {
    GLFWwindow *window;

    if (!glfwInit()) {
        std::cout << "GLFW is not init\n";
        return -1;
    }
    window = glfwCreateWindow(1024, 720, "FractalGPU", NULL, NULL);
    if (!window) {
        std::cout << "Window creation failed\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        glfwTerminate();
        return -1;
    }

    glClearColor(0.25f, 0.5f, 0.75f, 1.0);
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        glClear(GL_COLOR_BUFFER_BIT);
        glfwSwapBuffers(window);
    }
    glfwTerminate();
    return 0;
}
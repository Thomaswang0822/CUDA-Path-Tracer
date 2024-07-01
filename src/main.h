#pragma once

#include "glslUtility.hpp"
#include "scene.h"
#include <cuda_gl_interop.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

using namespace std;

//-------------------------------
//----------PATH TRACER----------
//-------------------------------

extern Scene* scene;
extern unsigned int iteration;

extern int width;
extern int height;

void runCuda();
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
void mouseScrollCallback(GLFWwindow* window, double offsetX, double offsetY);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);

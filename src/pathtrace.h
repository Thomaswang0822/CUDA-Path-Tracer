#pragma once

#include <device_launch_parameters.h>
#include <vector>
#include "scene.h"
#include "common.h"

void pathTraceInit(Scene *scene);
void pathTraceFree();
void pathTrace(uchar4* pbo, glm::vec3* devImage, Scene* hstScene);

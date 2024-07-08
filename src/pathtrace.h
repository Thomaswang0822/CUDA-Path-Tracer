#pragma once

#include <device_launch_parameters.h>
#include <vector>
#include "scene.h"
#include "common.h"

__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
    glm::vec3* Image, int toneMapping
);

void pathTraceInit(Scene *scene);
void pathTraceFree();
void pathTrace(uchar4* pbo, glm::vec3* devImage, Scene* hstScene);

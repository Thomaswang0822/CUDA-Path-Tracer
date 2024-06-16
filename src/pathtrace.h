#pragma once

#include <vector>
#include <device_launch_parameters.h>  // "let VS know" blockIdx etc.
#include "scene.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);

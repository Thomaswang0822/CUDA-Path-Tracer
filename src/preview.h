#pragma once

#include "utilities.h"
#include <GL/gl.h>
#include <string>

extern GLuint pbo;

std::string currentTimeString();
bool init();
void mainLoop();

bool MouseOverImGuiWindow();
void InitImguiData(GuiDataContainer* guiData);
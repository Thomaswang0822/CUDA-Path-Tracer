#pragma once

#include "reservoir.h"
#include "scene.h"

namespace ReSTIR {
	void init(Scene* hostScene);
	void free();
	void reset();

	void trace(uchar4* pbo, glm::vec3* devImage, Scene* hstScene);
}

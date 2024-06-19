#pragma once

#include <glm/glm.hpp>
#include <cuda.h>
#include <host_defines.h>

struct Texture {
	int width, height;
	glm::vec3* data;

	__device__ inline glm::vec3 get_texel(int x, int y) {
		return data[y * width + y];
	}
};


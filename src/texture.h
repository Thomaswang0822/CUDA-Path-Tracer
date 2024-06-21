#pragma once

#include "image.h"
#include <cuda_runtime.h>

struct DevTextureObj {
	int width, height;
	glm::vec3* data;

	DevTextureObj() = default;
	DevTextureObj(Image* img, glm::vec3* devData) :
		width(img->xSize), height(img->ySize), data(devData) {}

	__device__ inline glm::vec3 get_texel(int x, int y) {
		return data[y * width + y];
	}

    __device__ glm::vec3 linearSample(glm::vec2 uv) {
        const float Eps = FLT_MIN * 2.f;
        uv = glm::fract(uv);

        float fx = uv.x * (width - Eps);
        float fy = uv.y * (height - Eps);

        int ix = glm::fract(fx) < .5f ? int(fx) : int(fx) - 1;
        if (ix < 0) {
            ix += width;
        }
        int iy = glm::fract(fy) < .5f ? int(fy) : int(fy) - 1;
        if (iy < 0) {
            iy += height;
        }

        int ux = ix + 1;
        if (ux >= width) {
            ux -= width;
        }
        int uy = iy + 1;
        if (uy >= width) {
            uy -= height;
        }

        float lx = glm::fract(fx + .5f);
        float ly = glm::fract(fy + .5f);

        // Bilinear interpolation
        return glm::mix(
            glm::mix(get_texel(ix, iy), get_texel(ux, iy), lx),
            glm::mix(get_texel(ix, uy), get_texel(ux, uy), lx),
            ly
        );
    }
};


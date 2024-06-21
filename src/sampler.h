#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <thrust/random.h>


/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

__host__ __device__ static
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

struct Sampler {
    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<float> u01;

    // constructor
    __host__ __device__
    Sampler(int iter = 0, int threadIdx = 0, int recurDepth = 0) {
        rng = makeSeededRandomEngine(iter, threadIdx, recurDepth);
        u01 = thrust::uniform_real_distribution<float>(0, 1);
    }

    __host__ __device__ inline float sample1D() {
        return u01(rng);
    }

    __host__ __device__ inline glm::vec2 sample2D() {
        return glm::vec2(u01(rng), u01(rng));
    }

    __host__ __device__ inline glm::vec3 sample3D() {
        return glm::vec3(u01(rng), u01(rng), u01(rng));
    }
};
#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>

#define DEBUG_RED glm::vec3(1.f, 0.f, 0.f)
#define DEBUG_GREEN glm::vec3(0.f, 1.f, 0.f)
#define DEBUG_BLUE glm::vec3(0.f, 0.f, 1.f)

#define BVH_DEBUG_VISUALIZATION false

__device__
static inline bool badVec(glm::vec3& v) {
    float len = glm::length(v);
    return len <= 1e-6f || len >= 1e6f;
}

__device__
static inline bool notUnitLength(glm::vec3& v) {
    float len = glm::length(v);
    return abs(len - 1.f) > 1e-3f;
}

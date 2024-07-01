#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>

#define DEBUG_RED glm::vec3(1.f, 0.f, 0.f)
#define DEBUG_GREEN glm::vec3(0.f, 1.f, 0.f)
#define DEBUG_BLUE glm::vec3(0.f, 0.f, 1.f)

#define BVH_DEBUG_VISUALIZATION false

namespace Debug {
    __device__
        static inline bool badVec(glm::vec3 v) {
        float len = glm::length(v);
        return len <= 1e-6f || len >= 1e6f;
    }

    __device__
        static inline bool isNanInf(glm::vec3 r) {
        return isnan(r.x) || isnan(r.y) || isnan(r.z) ||
            isinf(r.x) || isinf(r.y) || isinf(r.z);
    }

    __device__
        static inline bool notUnitLength(glm::vec3 v) {
        float len = glm::length(v);
        return abs(len - 1.f) > 1e-6f;
    }
}

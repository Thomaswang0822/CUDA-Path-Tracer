#pragma once

#include "glm/glm.hpp"

struct AABB {
    glm::vec3 minPos;
    glm::vec3 maxPos;
};

struct BVH_Node {
    AABB box;
    int geomIdx;
    int size;
};
#pragma once

#include "glm/glm.hpp"

struct AABB {
    glm::vec3 minPos;
    glm::vec3 maxPos;

    /** Constructors */
    AABB() : minPos(FLT_MAX), maxPos(-FLT_MAX) {};
    AABB(glm::vec3 pmin, glm::vec3 pmax) : minPos(pmin), maxPos(pmax) {};
    // box a 3 positions, likely a triangle
    AABB(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2) :
        minPos(glm::min(p0, glm::min(p1, p2))), maxPos(glm::max(p0, glm::max(p1, p2))) {};
    // combine 2 AABB
    AABB(const AABB& boxA, const AABB& boxB) :
        minPos(glm::min(boxA.minPos, boxB.minPos)), maxPos(glm::min(boxA.maxPos, boxB.maxPos)) {};

    /** Member functions */
    inline glm::vec3 center() const {
        return 0.5f * (minPos + maxPos);
    }

    inline float surfaceArea() const {
        glm::vec3 size3D = maxPos - minPos;
        return 2.0f * (size3D.x * size3D.y + size3D.x * size3D.z + size3D.y * size3D.z);
    }

    inline size_t longestAxis() const {
        glm::vec3 size3D = maxPos - minPos;
        if (size3D.x < size3D.y) {  // compare Y and Z
            return (size3D.y < size3D.z) ? 2 : 1;
        }
        else { // compare X and Z
            return (size3D.x < size3D.z) ? 2 : 0;
        }
    }
};

struct BVH_Node {
    AABB box;
    int geomIdx;
    int size;
};
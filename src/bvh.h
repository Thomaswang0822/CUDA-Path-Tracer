#pragma once

#include <vector>
#include <sstream>
#include <iomanip>
#include "glm/glm.hpp"
#include "camera.h"

#define NullPrimitive -1

struct AABB {
    glm::vec3 minPos;
    glm::vec3 maxPos;

    /** Constructors */
    AABB() : minPos(FLT_MAX), maxPos(-FLT_MAX) {};
    AABB(glm::vec3 pmin, glm::vec3 pmax) : minPos(pmin), maxPos(pmax) {};
    // box 3 positions, likely a triangle
    AABB(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2) :
        minPos(glm::min(p0, glm::min(p1, p2))), maxPos(glm::max(p0, glm::max(p1, p2))) {};
    // combine 2 AABB
    AABB(const AABB& boxA, const AABB& boxB) :
        minPos(glm::min(boxA.minPos, boxB.minPos)), maxPos(glm::min(boxA.maxPos, boxB.maxPos)) {};
    /** 2 more ways to build new AABB */
    AABB operator() (glm::vec3 p) {
        return { glm::min(minPos, p), glm::max(maxPos, p) };
    }
    AABB operator() (const AABB& rhs) {
        return { glm::min(minPos, rhs.minPos), glm::max(maxPos, rhs.maxPos) };
    }

    /** Helpers */
    std::string toString() const {
        std::stringstream ss;
        ss << "[AABB " << "pMin = " << utilityCore::vec3ToString(minPos);
        ss << ", pMax = " << utilityCore::vec3ToString(maxPos);
        ss << ", center = " << utilityCore::vec3ToString(this->center()) << "]";
        return ss.str();
    }

    /** Member functions */
    __host__ __device__ inline glm::vec3 center() const {
        return 0.5f * (minPos + maxPos);
    }

    __host__ __device__ inline float surfaceArea() const {
        glm::vec3 size3D = maxPos - minPos;
        return 2.0f * (size3D.x * size3D.y + size3D.x * size3D.z + size3D.y * size3D.z);
    }

    __host__ __device__ inline size_t longestAxis() const {
        glm::vec3 size3D = maxPos - minPos;
        if (size3D.x < size3D.y) {  // compare Y and Z
            return (size3D.y < size3D.z) ? 2 : 1;
        }
        else { // compare X and Z
            return (size3D.x < size3D.z) ? 2 : 0;
        }
    }

    __host__ __device__ bool intersect(const Ray& ray, float& dist) {
        glm::vec3 ori = ray.origin;
        glm::vec3 dir = ray.direction;

        glm::vec3 t1 = (minPos - ori) / dir;
        glm::vec3 t2 = (maxPos - ori) / dir;

        glm::vec3 ta = glm::min(t1, t2);
        glm::vec3 tb = glm::max(t1, t2);

        float tMin = -FLT_MAX, tMax = FLT_MAX;

        for (int i = 0; i < 3; i++) {
            if (glm::abs(dir[i]) > EPSILON) {
                if (tb[i] >= 0.f && tb[i] >= ta[i]) {
                    tMin = glm::max(tMin, ta[i]);
                    tMax = glm::min(tMax, tb[i]);
                }
            }
        }
        dist = tMin;

        if (tMax >= 0.f && tMax >= tMin - EPSILON) {
            glm::vec3 mid = ray.getPoint((tMin + tMax) * .5f);

            for (int i = 0; i < 3; i++) {
                if (mid[i] <= minPos[i] - EPSILON || mid[i] >= maxPos[i] + EPSILON) {
                    return false;
                }
            }
            return true;
        }
        return false;
    }
};

/**
 * Multi-Threaded BVH
 * MTBVH enables stackless BVH traversal on GPU, saving many registers that were used in stack-based
 * traversal. It's simple and efficient.
 * 
 * @see https://cs.uwaterloo.ca/~thachisu/tdf2015.pdf
 * @see https://developer.nvidia.com/blog/thinking-parallel-part-ii-tree-traversal-gpu/
 */
struct MTBVHNode {
    int primitiveId;
    int boundingBoxId;
    int nextNodeIfMiss;

    MTBVHNode() = default;
    MTBVHNode(int primId, int boxId, int next) :
        primitiveId(primId), boundingBoxId(boxId), nextNodeIfMiss(next) {}
};

/**
 * MTBVH builder
 */
class BVHBuilder {
private:
    struct NodeInfo {
        bool isLeaf;
        int primIdOrSize;
    };

    struct PrimInfo {
        int primId;
        AABB bound;
        glm::vec3 center;
    };

    struct BuildInfo {
        int offset;
        int start;
        int end;
    };
public:
    static int build(
        const std::vector<glm::vec3>& vertices,
        std::vector<AABB>& boundingBoxes,
        std::vector<std::vector<MTBVHNode>>& BVHNodes);

private:
    static void buildMTBVH(
        const std::vector<AABB>& boundingBoxes,
        const std::vector<NodeInfo>& nodeInfo,
        int BVHSize,
        std::vector<std::vector<MTBVHNode>>& BVHNodes);

};

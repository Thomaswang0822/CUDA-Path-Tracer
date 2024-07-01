#pragma once

#include "camera.h"
#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))


struct RenderState {
    Camera camera;
    unsigned int iterations;
    //int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PrevBSDFSampleInfo {
    float BSDFPdf;
    bool deltaSample;
};

struct PathSegment {
    Ray ray;
    glm::vec3 throughput;
    glm::vec3 radiance;
    PrevBSDFSampleInfo prev;
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct Intersection {
    //float t;  // hitting distance of current ray
    int primId;
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
    glm::vec3 inDir;
    union {
        glm::vec3 wo;  // for BSDF sampling
        glm::vec3 prevPos;  // for light sampling
    };
    int materialId;
    PrevBSDFSampleInfo prev;

    __device__ Intersection() {}

    __device__ Intersection(const Intersection& rhs) {
        *this = rhs;
    }

    __device__ void operator = (const Intersection& rhs) {
        primId = rhs.primId;
        materialId = rhs.materialId;
        position = rhs.position;
        normal = rhs.normal;
        uv = rhs.uv;
        wo = rhs.wo;
        prev = rhs.prev;
    }
};

struct CompactTerminatedPaths {
    __host__ __device__ bool operator() (const PathSegment& segment) {
        return !(segment.pixelIndex >= 0 && segment.remainingBounces <= 0);
    }
};

struct RemoveInvalidPaths {
    __host__ __device__ bool operator() (const PathSegment& segment) {
        return segment.pixelIndex < 0 || segment.remainingBounces <= 0;
    }
};

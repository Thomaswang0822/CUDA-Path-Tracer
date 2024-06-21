#pragma once

#include "camera.h"
#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum class GeomType {
    SPHERE,
    CUBE,
};

struct Geom {
    GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct Triangle {
    glm::vec3 vertex[3];
    glm::vec3 normal[3];
    glm::vec2 texcoord[3];
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment {
    Ray ray;
    glm::vec3 throughput;
    glm::vec3 radiance;
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct Intersection {
    //float t;  // hitting distance of current ray
    int primitive;
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texcoord;
    glm::vec3 inDir;
    int materialId;
};

struct CompactTerminatedPaths {
    __host__ __device__ bool operator() (const PathSegment& segment) {
        return !(segment.pixelIndex >= 0 && segment.remainingBounces == 0);
    }
};

struct RemoveInvalidPaths {
    __host__ __device__ bool operator() (const PathSegment& segment) {
        return segment.pixelIndex < 0 || segment.remainingBounces == 0;
    }
};

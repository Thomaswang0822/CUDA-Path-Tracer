#pragma once

#include <string>
#include <vector>

#include "camera.h"

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
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
    float t;
    glm::vec3 surfaceNormal;
    glm::vec3 surfaceUV;
    glm::vec3 inDir;
    int materialId;
};

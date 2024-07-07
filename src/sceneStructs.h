#pragma once

#include <iomanip>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "mathUtil.h"

struct Ray {
    __host__ __device__ glm::vec3 getPoint(float dist) {
        return origin + direction * dist;
    }

    glm::vec3 origin;
    glm::vec3 direction;
};

struct Camera {
    void update() {
        float yaw = glm::radians(rotation.x);
        float pitch = glm::radians(rotation.y);
        float roll = glm::radians(rotation.z);
        view.x = glm::cos(yaw) * glm::cos(pitch);
        view.z = glm::sin(yaw) * glm::cos(pitch);
        view.y = glm::sin(pitch);

        view = glm::normalize(view);
        right = glm::normalize(glm::cross(view, glm::vec3(0, 1, 0)));
        up = glm::normalize(glm::cross(right, view));
    }

    /*
    * Antialiasing and physically based camera (lens effect)
    */
    __device__ Ray sample(int x, int y, glm::vec4 r) const {
        Ray ray;
        float aspect = float(resolution.x) / resolution.y;
        float tanFovY = glm::tan(glm::radians(fov.y));
        glm::vec2 pixelSize = 1.f / glm::vec2(resolution);
        glm::vec2 scr = glm::vec2(x, y) * pixelSize;
        glm::vec2 ruv = scr + pixelSize * glm::vec2(r.x, r.y);
        ruv = 1.f - ruv * 2.f;

        glm::vec2 pAperture = Math::toConcentricDisk(r.z, r.w);
        glm::vec3 pLens = glm::vec3(pAperture * lensRadius, 0.f);
        glm::vec3 pFocusPlane = glm::vec3(ruv * glm::vec2(aspect, 1.f) * tanFovY, 1.f) * focalDist;
        glm::vec3 dir = pFocusPlane - pLens;

        ray.direction = glm::normalize(glm::mat3(right, up, view) * dir);
        ray.origin = position + right * pLens.x + up * pLens.y;
        return ray;
    }

    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 rotation;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
    float lensRadius;
    float focalDist;
    float tanFovY;
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

struct RenderState {
    unsigned int iterations;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct Intersection {
    __device__ Intersection() {}

    __device__ Intersection(const Intersection& rhs) {
        *this = rhs;
    }

    __device__ void operator = (const Intersection& rhs) {
        primId = rhs.primId;
        matId = rhs.matId;
        pos = rhs.pos;
        norm = rhs.norm;
        uv = rhs.uv;
        wo = rhs.wo;
        prev = rhs.prev;
    }

    int primId;
    int matId;

    glm::vec3 pos;
    glm::vec3 norm;
    glm::vec2 uv;

    union {
        glm::vec3 wo;
        glm::vec3 prevPos;
    };

    PrevBSDFSampleInfo prev;
};
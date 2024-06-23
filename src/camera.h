#pragma once

#include "mathUtility.h"
#include <cuda_runtime.h>


struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;

    /**
     * Used on PathSegment.ray: update to next (sampled direction) ray.
     */
    __host__ __device__ inline void nextRay(glm::vec3 pos, glm::vec3 dir) {
        direction = dir;
        origin = pos + EPSILON * dir;
    }

    /** Get a point at certain distance */
    __host__ __device__ inline glm::vec3 getPoint(float t) const {
        return origin + direction * t;
    }

    /**
     * @param dir should be normalized.
     */
    __host__ __device__ inline static Ray makeOffsetRay(glm::vec3 orig, glm::vec3 dir) {
        return { orig + EPSILON * dir, dir };
    }
};

struct Camera {
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;  // in degree
    glm::vec2 pixelLength;
    float lensRadius;
    float focalDist;

    /**
     * When the user move the camera (by mouse), update the camera in renderState in place
     * 
     */
    __host__ void updateCamParam(const float zoom, const float phi, const float theta) {
        glm::vec3 cameraPosition;  // tmp var
        cameraPosition.x = zoom * sin(phi) * sin(theta);
        cameraPosition.y = zoom * cos(theta);
        cameraPosition.z = zoom * cos(phi) * sin(theta);

        // view, up, right unit vectors
        view = -glm::normalize(cameraPosition);
        glm::vec3 v = view;
        glm::vec3 u = glm::vec3(0, 1, 0);//glm::normalize(cam.up);
        glm::vec3 r = glm::cross(v, u);
        up = glm::cross(r, v);
        right = r;

        position = cameraPosition + lookAt;
    }

    /**
     * Generate a camera ray (random within the pixel). The ray is modified in-place
     */
    __host__ __device__ void generateCameraRay(Ray& ray, 
            const int x, const int y, glm::vec2 rand2) {
        
        ray.origin = position;

        float moveNumPixelX = (float)x - 0.5f * (float)resolution.x;
        float moveNumPixelY = (float)y - 0.5f * (float)resolution.y;
        ray.direction = glm::normalize(view
            - right * pixelLength.x * (moveNumPixelX + rand2.x - 0.5f)
            - up * pixelLength.y * (moveNumPixelY - rand2.y - 0.5f)
        );
    }
};

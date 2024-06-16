#pragma once

#include "utilities.h"

struct Material {
    glm::vec3 color;
    struct {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
};


__device__ inline glm::vec3 schlickFresnel(glm::vec3 f0, float cosTheta) {
    //return f0 + (glm::vec3(1.0f) - f0) * powf(1.0 - cosTheta, 5.0);
    return glm::mix(f0, glm::vec3(1.f), powf(1.f - cosTheta, 5.f));
}
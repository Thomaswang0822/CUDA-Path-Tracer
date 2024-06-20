#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>


// define const MACROS here
#define PI                3.1415926535897932384626422832795028841971f
#define TWO_PI            6.2831853071795864769252867665590057683943f
#define INV_PI            0.3183098861837906912164442019275156781077f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define EPSILON           1e-6f

namespace mathUtil {
    __host__ __device__ inline float nonnegativeDot(glm::vec3 a, glm::vec3 b) {
        return glm::max(glm::dot(a, b), 0.f);
    }

    __host__ __device__ inline float absDot(glm::vec3 a, glm::vec3 b) {
        return glm::abs(glm::dot(a, b));
    }

    __host__ __device__ inline float pow5(float x) {
        float x2 = x * x;
        return x2 * x2 * x;
    }

    __host__ __device__ inline float square(float x) {
        return x * x;
    }

    /**
    * Map a pair of uniformly distributed [0, 1] coordinate to unit disc
    */
    __device__ static glm::vec2 toUnitDisk(float x, float y) {
        float r = glm::sqrt(x);
        float theta = y * TWO_PI;
        return glm::vec2(glm::cos(theta), glm::sin(theta)) * r;
    }

    __device__ static glm::mat3 localRefMatrix(glm::vec3 n) {
        glm::vec3 t = (glm::abs(n.y) > 0.9f) ? glm::vec3(0.f, 0.f, 1.f) : glm::vec3(0.f, 1.f, 0.f);
        glm::vec3 b = glm::cross(n, t);
        t = glm::cross(b, n);
        return glm::mat3(t, b, n);
    }

    __device__ static glm::vec3 localToWorld(glm::vec3 n, glm::vec3 v) {
        // build an orthonormal basis (tan, bitan, normal)
        glm::vec3 _some = (glm::abs(n.y) > 0.9f) ? glm::vec3(0.f, 0.f, 1.f) : glm::vec3(0.f, 1.f, 0.f);
        glm::vec3 t = glm::normalize(glm::cross(n, _some));
        glm::vec3 b = glm::cross(n, t);
        //// Eq to 
        //glm::vec3 b = glm::cross(n, _some);
        //glm::vec3 t = glm::cross(b, n);

        return glm::normalize(glm::vec3(t * v.x + b * v.y + n * v.z));
    }

    /**
     * @see https://ameye.dev/notes/sampling-the-hemisphere/#cosine-weighted-hemisphere.
     */
    __device__ static glm::vec3 sampleHemisphereCosine(glm::vec3 n, float u1, float u2) {
        /*glm::vec2 d = toUnitDisk(rx, ry);
        float z = glm::sqrt(1.f - glm::dot(d, d));
        return localToWorld(n, glm::vec3(d, z));*/

        // compute the local {cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)}
        float sin_theta = sqrt(1.f - u2);  // float cos_theta = sqrt(u2);
        float phi = TWO_PI * u1;
        glm::vec3 local_dir(
            cos(phi) * sin_theta,
            sin(phi) * sin_theta,
            sqrt(u2)
        );
        return localToWorld(n, local_dir);
    }

    /**
     * @brief Determines whether an incoming ray is refracted based on the index of refraction
     *        and computes the refracted direction.
     *
     * @param[in] n The normal vector at the surface where refraction is taking place.
     * @param[in] wi The incoming ray direction.
     * @param[in] ior The index of refraction of the material.
     * @param[out] wt The refracted ray direction, if refraction occurs.
     * @return bool True if refraction occurs, false if total internal reflection occurs.
     */
    __device__ static bool refract(glm::vec3 n, glm::vec3 wi, float ior, glm::vec3& wt) {
        float cosIn = glm::dot(n, wi);
        if (cosIn < 0) {  // internal ray: reverse 1. ior; 2. out angle
            ior = 1.f / ior;
        }
        float sin2In = glm::max(0.f, 1.f - cosIn * cosIn);
        float sin2Tr = sin2In / (ior * ior);  // Snell's Law

        if (sin2Tr >= 1.f) {  // total internal refelction
            return false;
        }
        float cosTr = glm::sqrt(1.f - sin2Tr);
        if (cosIn < 0) {
            cosTr = -cosTr;
        }
        // Compute the Transmitted Direction
        wt = glm::normalize(-wi / ior + n * (cosIn / ior - cosTr));
        return true;
    }

    __host__ __device__ inline glm::vec3 mapACES(glm::vec3 color) {
        return (color * (color * 2.51f + 0.03f)) / (color * (color * 2.43f + 0.59f) + 0.14f);
    }

    __host__ __device__ inline glm::vec3 correctGamma(glm::vec3 color) {
        return glm::pow(color, glm::vec3(1.f / 2.2f));
    }

}



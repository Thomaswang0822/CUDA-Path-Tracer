#pragma once

#include <vector>
#include <map>
#include <fstream>
#include "materials.h"
#include "sceneStructs.h"
#include "bvh.h"
#include "image.h"
#include "texture.h"
#include "camera.h"
#include "mesh.h"
#include "sampler.h"
#include "intersections.h"

#define SCENE_LIGHT_SINGLE_SIDED true

class Scene;  // declare for DevScene

/**
 * Scene definition used on GPU.
 * Its primary functionality is to compute intersection.
 */
struct DevScene {
    /** Geom data */
    glm::vec3* devVertices = nullptr;
    glm::vec3* devNormals = nullptr;
    /** BVH data */
    AABB* devBoundingBoxes = nullptr;
    MTBVHNode* devBVHNodes[NUM_FACES] = { nullptr };
    int BVHSize;
    /** Shading data */
    int* devMaterialIds = nullptr;
    Material* devMaterials = nullptr;
    glm::vec2* devTexcoords = nullptr;
    glm::vec3* devTextureData = nullptr;
    DevTextureObj* devTextureObjs = nullptr;
    /** Light data */
    int* devLightPrimIds = nullptr;
    glm::vec3* devLightUnitRadiance = nullptr;
    float* devProbTable = nullptr;
    int* devAliasTable = nullptr;
    int numLightPrims;
    float sumLightPowerInv;

    void createDevData(const Scene& scene);
    void freeDevData();

    /**
     * Given a ray direction, pick a MTBVH.
     * @see bvh.cpp and bvh.h for more details
     *
     * @param dir Direction of the ray
     * @return An index in 0-5
     *
     * |dir.x| dominates, then (dir.x > 0) ? 0 : 1
     * similar stories for dir.y and dir.z
     */
    __device__ int getMTBVHId(glm::vec3 dir) {
        glm::vec3 absDir = glm::abs(dir);
        if (absDir.x > absDir.y) {
            if (absDir.x > absDir.z) {
                return dir.x > 0 ? 0 : 1;
            }
            else {
                return dir.z > 0 ? 4 : 5;
            }
        }
        else {
            if (absDir.y > absDir.z) {
                return dir.y > 0 ? 2 : 3;
            }
            else {
                return dir.z > 0 ? 4 : 5;
            }
        }
    }

    /**
     * After intersection test, fetch info of intersected triangle.
     *
     * @param intersec Output parameter to be updated
     */
    __device__ glm::vec3 getPrimitiveNormal(const int primId) {
        glm::vec3 v0 = devVertices[primId * 3 + 0];
        glm::vec3 v1 = devVertices[primId * 3 + 1];
        glm::vec3 v2 = devVertices[primId * 3 + 2];
        return glm::normalize(glm::cross(v1 - v0, v2 - v0));
    }

    /**
     * Grab the triangle from the vertices pool and perform ray-triangle test.
     *
     * @param dist Output parameter to be updated (if a closer hit occurs)
     * @param bary Output parameter to be updated (if a closer hit occurs)
     */
    __device__ void getIntersecGeomInfo(int primId, const glm::vec2 bary, Intersection& intersec) {
        glm::vec3 va = devVertices[primId * 3 + 0];
        glm::vec3 vb = devVertices[primId * 3 + 1];
        glm::vec3 vc = devVertices[primId * 3 + 2];

        glm::vec3 na = devNormals[primId * 3 + 0];
        glm::vec3 nb = devNormals[primId * 3 + 1];
        glm::vec3 nc = devNormals[primId * 3 + 2];

        glm::vec2 ta = devTexcoords[primId * 3 + 0];
        glm::vec2 tb = devTexcoords[primId * 3 + 1];
        glm::vec2 tc = devTexcoords[primId * 3 + 2];

        intersec.position = vb * bary.x + vc * bary.y + va * (1.f - bary.x - bary.y);
        intersec.normal = nb * bary.x + nc * bary.y + na * (1.f - bary.x - bary.y);
        intersec.uv = tb * bary.x + tc * bary.y + ta * (1.f - bary.x - bary.y);
    }

    __device__ bool intersectPrimitive(int primId, const Ray& ray, float& dist, glm::vec2& bary) {
        glm::vec3 va = devVertices[primId * 3 + 0];
        glm::vec3 vb = devVertices[primId * 3 + 1];
        glm::vec3 vc = devVertices[primId * 3 + 2];

        /*if (!intersectTriangle(ray, va, vb, vc, bary, dist)) {
            return false;
        }
        glm::vec3 hitPoint = vb * bary.x + vc * bary.y + va * (1.f - bary.x - bary.y);
        return true;*/

        return intersectTriangle(ray, va, vb, vc, bary, dist);
    }

    __device__ bool intersectPrimitive(int primId, const Ray& ray, float distRange) {
        glm::vec3 va = devVertices[primId * 3 + 0];
        glm::vec3 vb = devVertices[primId * 3 + 1];
        glm::vec3 vc = devVertices[primId * 3 + 2];

        glm::vec2 bary;
        float dist;
        bool hit = intersectTriangle(ray, va, vb, vc, bary, dist);
        return (hit && dist < distRange);
    }

    /** NOT USED YET */
    __device__ bool intersectPrimitiveDetailed(int primId, const Ray& ray, Intersection& intersec) {
        glm::vec3 va = devVertices[primId * 3 + 0];
        glm::vec3 vb = devVertices[primId * 3 + 1];
        glm::vec3 vc = devVertices[primId * 3 + 2];
        float dist;
        glm::vec2 bary;

        if (!intersectTriangle(ray, va, vb, vc, bary, dist)) {
            return false;
        }

        glm::vec3 na = devNormals[primId * 3 + 0];
        glm::vec3 nb = devNormals[primId * 3 + 1];
        glm::vec3 nc = devNormals[primId * 3 + 2];

        glm::vec2 ta = devTexcoords[primId * 3 + 0];
        glm::vec2 tb = devTexcoords[primId * 3 + 1];
        glm::vec2 tc = devTexcoords[primId * 3 + 2];

        intersec.position = vb * bary.x + vc * bary.y + va * (1.f - bary.x - bary.y);
        intersec.normal = nb * bary.x + nc * bary.y + na * (1.f - bary.x - bary.y);
        intersec.uv = tb * bary.x + tc * bary.y + ta * (1.f - bary.x - bary.y);
        return true;
    }

    __device__ void naiveIntersect(Ray ray, Intersection& intersec) {
        float closestDist = FLT_MAX;
        int closestPrimId = NullPrimitive;
        glm::vec2 closestBary;

        for (int i = 0; i < (BVHSize + 1) / 2; i++) {
            float dist;
            glm::vec2 bary;
            bool hit = intersectPrimitive(i, ray, dist, bary);

            if (hit && dist < closestDist) {
                closestDist = dist;
                closestBary = bary;
                closestPrimId = i;
            }
        }

        if (closestPrimId != NullPrimitive) {
            getIntersecGeomInfo(closestPrimId, closestBary, intersec);
            intersec.primId = closestPrimId;
            intersec.materialId = devMaterialIds[closestPrimId];
        }
        else {
            intersec.primId = NullPrimitive;
        }
    }

    /**
     * Given a ray, find the cloest intersection, if one exist, in the entire scene.
     *
     * @param intersec Output parameter
     */
    __device__ void intersect(const Ray& ray, Intersection& intersec) {
        float closestDist = FLT_MAX;
        int closestPrimId = NullPrimitive;
        glm::vec2 closestBary;

        MTBVHNode* nodes = devBVHNodes[getMTBVHId(-ray.direction)];
        int node = 0;

        while (node != BVHSize) {
            AABB& bound = devBoundingBoxes[nodes[node].boundingBoxId];
            float boundDist;
            bool boundHit = bound.intersect(ray, boundDist);

            // Only intersect a primitive if its bounding box is hit and
            // that box is closer than previous hit record
            if (boundHit && boundDist < closestDist) {
                int primId = nodes[node].primitiveId;
                if (primId != NullPrimitive) {
                    float dist;
                    glm::vec2 bary;
                    // hit info is written to dist and bary.
                    bool hit = intersectPrimitive(primId, ray, dist, bary);

                    if (hit && dist < closestDist) {
                        closestDist = dist;
                        closestBary = bary;
                        closestPrimId = primId;
                    }
                }
                node++;
            }
            else {
                node = nodes[node].nextNodeIfMiss;
            }
        }
        if (closestPrimId != NullPrimitive) {
            getIntersecGeomInfo(closestPrimId, closestBary, intersec);
            intersec.primId = closestPrimId;
            //intersec.inDir = -ray.direction;
            intersec.materialId = devMaterialIds[closestPrimId];
        }
        else {
            intersec.primId = NullPrimitive;
        }
    }

    /**
     * Shoot a ray from x to y and test occulsion.
     */
    __device__ bool testOcclusion(glm::vec3 x, glm::vec3 y) {
        glm::vec3 dir = glm::normalize(y - x);
        float dist = glm::length(y - x) - EPSILON * 2.f;   // BUG
        Ray ray = Ray::makeOffsetRay(x, dir);

        MTBVHNode* nodes = devBVHNodes[getMTBVHId(-ray.direction)];
        int node = 0;
        while (node != BVHSize) {
            AABB& bound = devBoundingBoxes[nodes[node].boundingBoxId];
            float boundDist;
            bool boundHit = bound.intersect(ray, boundDist);

            if (boundHit && boundDist < dist) {
                int primId = nodes[node].primitiveId;
                if (primId != NullPrimitive) {
                    if (intersectPrimitive(primId, ray, dist)) {
                        return true;
                    }
                }
                node++;
            }
            else {
                node = nodes[node].nextNodeIfMiss;
            }
        }
        return false;
    }

    /**
     * DEBUG version intersection test.
     *
     * intersec.primId will be written with #triangles hit
     */
    __device__ void visualizedIntersect(const Ray& ray, Intersection& intersec) {
        float closestDist = FLT_MAX;
        int closestPrimId = NullPrimitive;
        glm::vec2 closestBary;

        MTBVHNode* nodes = devBVHNodes[getMTBVHId(-ray.direction)];
        int node = 0;
        int maxDepth = 0;

        while (node != BVHSize) {
            AABB& bound = devBoundingBoxes[nodes[node].boundingBoxId];
            float boundDist;
            bool boundHit = bound.intersect(ray, boundDist);

            // Only intersect a primitive if its bounding box is hit and
            // that box is closer than previous hit record
            if (boundHit && boundDist < closestDist) {
                int primId = nodes[node].primitiveId;
                if (primId != NullPrimitive) {
                    float dist;
                    glm::vec2 bary;
                    bool hit = intersectPrimitive(primId, ray, dist, bary);

                    if (hit && dist < closestDist) {
                        closestDist = dist;
                        closestBary = bary;
                        closestPrimId = primId;
                        maxDepth += 1.f;
                    }
                }
                node++;
                maxDepth += 1.f;
            }
            else {
                node = nodes[node].nextNodeIfMiss;
            }
        }
        if (closestPrimId == 0) {
            maxDepth = 100.f;
        }
        intersec.primId = maxDepth;
    }

    /**
     * Randomly pick a light, test shadow ray.
     * 
     * @param pos: Current shading point
     * @param r: 4 random numbers
     * @param radiance: Output; randiance of sampled light
     * @param wi: Output; unit direction from pos to sampled point on light
     * 
     * @return pdf
     */
    __device__ float sampleDirectLight(glm::vec3 pos, glm::vec4 r, glm::vec3& radiance, glm::vec3& wi) {
        int bucketId = static_cast<int>(r.x * numLightPrims);
        int lightId = (r.y < devProbTable[bucketId]) ? bucketId : devAliasTable[bucketId];
        int primId = devLightPrimIds[lightId];

        glm::vec3 v0 = devVertices[primId * 3 + 0];
        glm::vec3 v1 = devVertices[primId * 3 + 1];
        glm::vec3 v2 = devVertices[primId * 3 + 2];
        glm::vec3 sampled = Math::sampleTriangleUniform(v0, v1, v2, r.z, r.w);

        if (testOcclusion(pos, sampled)) {
            return InvalidPdf;
        }
        glm::vec3 normal = Math::triangleNormal(v0, v1, v2);
        glm::vec3 posToSampled = sampled - pos;

#if SCENE_LIGHT_SINGLE_SIDED
        if (glm::dot(normal, posToSampled) > 0.f) {
            return InvalidPdf;
        }
#endif
        float area = Math::triangleArea(v0, v1, v2);
        radiance = devLightUnitRadiance[lightId];
        wi = glm::normalize(posToSampled);
        float power = Math::luminance(radiance) /*/ (area * TWO_PI)*/;
        return Math::pdfAreaToSolidAngle(power * sumLightPowerInv, pos, sampled, normal);
    }
};

/**
 * Scene definition on CPU.
 * It does all the heavy liftings, which include loading everything,
 * building BVH, and managing a copy of data on GPU.
 */
class Scene {
private:
    std::ifstream fp_in;

    void createLightSampler();
    void loadMaterial(const std::string& materialId);
    void loadModel(const std::string& objectId);
    void loadCamera();

public:
    friend struct DevScene;

    Scene(const std::string& filename);
    ~Scene();
    void buildDevData();
    void clear();

    RenderState state;
    std::vector<ModelInstance> modelInstances;
    std::vector<Image*> textures;
    std::vector<Material> materials;
    std::map<std::string, int> materialMap;

    std::vector<int> materialIds;
    size_t BVHSize;
    std::vector<AABB> boundingBoxes;
    std::vector<std::vector<MTBVHNode>> BVHNodes;
    MeshData meshData;

    /** Light Data */
    std::vector<int> lightPrimIds;
    std::vector<float> lightPower;
    std::vector<glm::vec3> lightUnitRadiance;
    LightSampler lightSampler;
    int numLightPrims = 0;
    float sumLightPower = 0.f;

    DevScene hostScene;  // Thus, DevScene is a "subset" of Scene
    DevScene* devScene = nullptr;
};


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

class Scene;  // declare for DevScene

/**
 * Scene definition used on GPU.
 * Its primary functionality is to compute intersection.
 */
struct DevScene {
    glm::vec3* devVertices = nullptr;
    glm::vec3* devNormals = nullptr;
    glm::vec2* devTexcoords = nullptr;
    AABB* devBoundingBoxes = nullptr;
    MTBVHNode* devBVHNodes[NUM_FACES] = { nullptr };
    int BVHSize;

    int* devMaterialIds = nullptr;
    Material* devMaterials = nullptr;
    glm::vec3* devTextureData = nullptr;
    DevTextureObj* devTextureObjs = nullptr;

    void createDevData(Scene& scene);
    void freeDevData();
    __device__ int getMTBVHId(glm::vec3 dir);
    __device__ void getIntersecGeomInfo(int primId, const glm::vec2 bary, Intersection& intersec);
    __device__ bool intersectPrimitive(int primId, const Ray& ray, float& dist, glm::vec2& bary);
    __device__ bool intersectPrimitiveDetailed(int primId, const Ray& ray, Intersection& intersec);
    __device__ void intersect(const Ray& ray, Intersection& intersec);
    __device__ void visualizedIntersect(const Ray& ray, Intersection& intersec);
};

/**
 * Scene definition on CPU.
 * It does all the heavy liftings, which include loading everything,
 * building BVH, and managing a copy of data on GPU.
 */
class Scene {
private:
    std::ifstream fp_in;

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

    DevScene hostScene;  // Thus, DevScene is a "subset" of Scene
    DevScene* devScene = nullptr;
};


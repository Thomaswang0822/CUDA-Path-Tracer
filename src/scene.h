#pragma once

#include <vector>
#include <map>
#include <filesystem>
#include <sstream>
#include <fstream>
#include <iostream>
#include "materials.h"
#include "utilities.h"
#include "sceneStructs.h"
#include "TinyObjLoader/tiny_obj_loader.h"
#include "bvh.h"
#include "image.h"
#include "texture.h"
#include "camera.h"
#include "intersections.h"

/** Options */
#define ABS_SCENES_PATH "D:/Code/CUDA-Path-Tracer/scenes"
#define MESH_DATA_STRUCT_OF_ARRAY false
#define MESH_DATA_INDEXED false

/** Constants */
#define NUM_FACES 6

struct MeshData {
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> texcoords;
#if MESH_DATA_INDEXED
    std::vector<glm::ivec3> indices;
#endif
};

struct ModelInstance {
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;

    glm::mat4 transform;
    glm::mat4 transfInv;
    glm::mat3 normalMat;

    int materialId;
    MeshData* meshData;
};

class Resource {
private:
    static std::map<std::string, MeshData*> meshDataPool;
    static std::map<std::string, Image*> texturePool;
    static std::filesystem::path scenes_path;

public:
    // A caller loading different types of mesh format
    static MeshData* loadModelMeshData(const std::string& filename);
    static MeshData* loadOBJMesh(const std::string& filename);
    static MeshData* loadGLTFMesh(const std::string& filename);
    
    static Image* loadTexture(const std::string& filename);

    static void clear();
};

class Scene;  // declare for DevScene

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
    __device__ void getIntersecGeomInfo(int primId, glm::vec2 bary, Intersection& intersec);
    __device__ bool intersectPrimitive(int primId, Ray ray, float& dist, glm::vec2& bary);
    __device__ bool intersectPrimitiveDetailed(int primId, Ray ray, Intersection& intersec);
    __device__ void intersect(Ray ray, Intersection& intersec);
    __device__ void visualizedIntersect(Ray ray, Intersection& intersec);
};

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
    // TODO:
    void buildDevData();
    void clear();

    RenderState state;
    std::vector<ModelInstance> modelInstances;
    std::vector<Image*> textures;
    std::vector<Material> materials;
    std::map<std::string, int> materialMap;

    std::vector<int> materialIds;
    int BVHSize;
    std::vector<AABB> boundingBoxes;
    std::vector<std::vector<MTBVHNode>> BVHNodes;
    MeshData meshData;

    DevScene hstScene;
    DevScene* devScene = nullptr;
};

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

#define ABS_SCENES_PATH "D:/Code/CUDA-Path-Tracer/scenes"
#define MESH_DATA_STRUCT_OF_ARRAY false
#define MESH_DATA_INDEXED false

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

struct DevResource {
    glm::vec3* devVertices = nullptr;
    glm::vec3* devNormals = nullptr;
    glm::vec2* devTexcoords = nullptr;
    AABB* devBoundingBoxes = nullptr;
    MTBVHNode* devBVHNodes[6] = { nullptr };
    int BVHSize;

    int* devMaterialIds = nullptr;
    Material* devMaterials = nullptr;
    glm::vec3* devTextureData = nullptr;
    DevTextureObj* devTextureObjs = nullptr;
};

class Scene {
private:
    std::ifstream fp_in;

    void loadMaterial(const std::string& materialId);
    void loadModel(const std::string& objectId);
    void loadCamera();

public:
    Scene(const std::string& filename);
    ~Scene();
    // TODO:
    void buildDevData();
    void clear();

    RenderState state;
    std::vector<Geom> geoms;
    std::vector<ModelInstance> modelInstances;
    std::vector<Image*> textures;
    std::vector<Material> materials;
    std::map<std::string, int> materialMap;

    std::vector<int> materialIds;
    int BVHSize;
    std::vector<AABB> boundingBoxes;
    std::vector<std::vector<MTBVHNode>> BVHNodes;
    MeshData meshData;

    DevResource devResources;
};

#pragma once

#include "image.h"
#include <filesystem>
#include <glm/glm.hpp>
#include <map>
#include <vector>


/** Options */
#define ABS_SCENES_PATH "D:/Code/CUDA-Path-Tracer/scenes"
#define MESH_DATA_INDEXED false

/**
 * Stores the definition of one mesh.
 */
struct MeshData {
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> texcoords;
#if MESH_DATA_INDEXED
    std::vector<glm::ivec3> indices;
#endif
};

/**
 * An instance of a mesh. For example, you can put many
 * Stanford bunnies in a Cornell box. Each bunny is a different
 * instance with different scale, transformation, etc., but they
 * share mesh data.
 */
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

/**
 * A utility class that calls 3rd-party loaders like tiny_obj_loader.
 * This class will be used in static.
 */
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

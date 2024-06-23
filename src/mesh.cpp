#include "mesh.h"
#include "TinyObjLoader/tiny_obj_loader.h"
#include <iostream>

std::map<std::string, MeshData*> Resource::meshDataPool;
std::map<std::string, Image*> Resource::texturePool;
std::filesystem::path Resource::scenes_path = std::filesystem::path(ABS_SCENES_PATH);

/**
 * Allocate memory for a MeshData and load an obj file to it, if it's a new mesh.
 */
MeshData* Resource::loadOBJMesh(const std::string& filename) {
    auto find = meshDataPool.find(filename);
    if (find != meshDataPool.end()) {
        return find->second;
    }
    auto model = new MeshData;

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::string warn, err;

    std::filesystem::path full_path = scenes_path / filename;
    const std::string full_path_string = full_path.string();

    std::cout << "MeshData::loading {" << full_path << "}" << std::endl;
    if (!tinyobj::LoadObj(&attrib, &shapes, nullptr, &warn, &err, full_path_string.c_str())) {
        std::cout << "failed\n\tError msg {" << err << "}" << std::endl;
        return nullptr;
    }
    bool hasTexcoord = !attrib.texcoords.empty();

#if INDEXED_MESH_DATA
    model->vertices.resize(attrib.vertices.size() / 3);
    model->normals.resize(attrib.normals.size() / 3);
    memcpy(model->vertices.data(), attrib.vertices.data(), attrib.vertices.size() * sizeof(float));
    memcpy(model->normals.data(), attrib.normals.data(), attrib.normals.size() * sizeof(float));
    if (hasTexcoord) {
        model->uv.resize(attrib.texcoords.size() / 2);
        memcpy(model->texcoords.data(), attrib.texcoords.data(), attrib.texcoords.size() * sizeof(float));
    }
    else {
        model->uv.resize(attrib.vertices.size() / 3);
    }

    for (const auto& shape : shapes) {
        for (auto idx : shape.mesh.indices) {
            model->indices.push_back({ idx.vertex_index, idx.normal_index,
                hasTexcoord ? idx.texcoord_index : idx.vertex_index });
        }
    }
#else
    for (const auto& shape : shapes) {
        for (auto idx : shape.mesh.indices) {
            model->vertices.push_back(*((glm::vec3*)attrib.vertices.data() + idx.vertex_index));
            model->normals.push_back(*((glm::vec3*)attrib.normals.data() + idx.normal_index));

            model->texcoords.push_back(hasTexcoord ?
                *((glm::vec2*)attrib.texcoords.data() + idx.texcoord_index) :
                glm::vec2(0.f)
            );
        }
    }
#endif
    std::cout << "\t\t[Vertex count = " << model->vertices.size() << "]" << std::endl;
    meshDataPool[filename] = model;
    return model;
}

MeshData* Resource::loadGLTFMesh(const std::string& filename) {
    throw std::invalid_argument("Resource::loadGLTFMesh NOT IMPLEMENTED");
    return nullptr;
}

/**
 * A global switcher.
 */
MeshData* Resource::loadModelMeshData(const std::string& filename) {
    if (filename.find(".obj") != filename.npos) {
        return loadOBJMesh(filename);
    }
    else {
        return loadGLTFMesh(filename);
    }
}

Image* Resource::loadTexture(const std::string& filename) {
    auto find = texturePool.find(filename);
    if (find != texturePool.end()) {
        return find->second;
    }
    auto texture = new Image(filename);
    texturePool[filename] = texture;
    return texture;
}

void Resource::clear() {
    for (auto i : meshDataPool) {
        delete i.second;
    }
    meshDataPool.clear();

    for (auto i : texturePool) {
        delete i.second;
    }
    texturePool.clear();
}
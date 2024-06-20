#include <iostream>
#include "scene.h"
//#include <cstring>
#include <string>
#include <map>
#include <filesystem>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/intersect.hpp>

map<string, Material::Type> MaterialTypeTokenMap = {
    { "Lambertian", Material::Type::Lambertian},
    { "MetallicWorkflow", Material::Type::MetallicWorkflow },
    { "Dielectric", Material::Type::Dielectric },
    { "Light", Material::Type::Light }
};

std::map<std::string, MeshData*> Resource::meshDataPool;
std::map<std::string, Image*> Resource::texturePool;
std::filesystem::path Resource::scenes_path = std::filesystem::path(ABS_SCENES_PATH);

#pragma region Resource
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
        model->texcoord.resize(attrib.texcoords.size() / 2);
        memcpy(model->texcoords.data(), attrib.texcoords.data(), attrib.texcoords.size() * sizeof(float));
    }
    else {
        model->texcoord.resize(attrib.vertices.size() / 3);
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
    return nullptr;
}

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
#pragma endregion

#pragma region Scene
Scene::Scene(const std::string& filename) {
    std::cout << "Scene::Reading from {" << filename << "}..." << std::endl;
    std::cout << " " << std::endl;
    char* fname = (char*)filename.c_str();
    fp_in.open(fname);
    if (!fp_in.is_open()) {
        std::cout << "Error reading from file - aborting!" << std::endl;
        throw;
    }
    while (fp_in.good()) {
        std::string line;
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty()) {
            std::vector<std::string> tokens = utilityCore::tokenizeString(line);
            if (tokens[0] == "Material") {
                loadMaterial(tokens[1]);
                std::cout << " " << std::endl;
            }
            else if (tokens[0] == "Object") {
                loadModel(tokens[1]);
                std::cout << " " << std::endl;
            }
            else if (tokens[0] == "Camera") {
                loadCamera();
                std::cout << " " << std::endl;
            }
        }
    }
}

Scene::~Scene() {
    clear();
}

void Scene::buildDevData() {
    // Put all texture devData in a big buffer
    // and setup device texture objects to manage
    /*std::vector<DevTextureObj> textureObjs;
    size_t textureTotalSize = 0;
    for (auto tex : textures) {
        textureTotalSize += tex->byteSize();
    }
    cudaMalloc(&devTextureData, textureTotalSize);
    size_t textureOffset = 0;
    for (auto tex : textures) {
        cudaMemcpy(devTextureData + textureOffset, tex->data(), tex->byteSize(), cudaMemcpyKind::cudaMemcpyHostToDevice);
        textureObjs.push_back({ tex, devTextureData + textureOffset });
        textureOffset += tex->byteSize();
    }
    cudaMalloc(&devTextureObjs, textureObjs.size() * sizeof(DevTextureObj));
    cudaMemcpy(devTextureObjs, textureObjs.data(), textureObjs.size() * sizeof(DevTextureObj),
        cudaMemcpyKind::cudaMemcpyHostToDevice);*/

#if MESH_DATA_INDEXED
#else
    for (const auto& inst : modelInstances) {
        for (size_t i = 0; i < inst.meshData->vertices.size(); i++) {
            meshData.vertices.push_back(glm::vec3(inst.transform * glm::vec4(inst.meshData->vertices[i], 1.f)));
            meshData.normals.push_back(glm::normalize(inst.normalMat * inst.meshData->normals[i]));
            meshData.texcoords.push_back(inst.meshData->texcoords[i]);
            if (i % 3 == 0) {
                materialIds.push_back(inst.materialId);
            }
        }
    }
#endif
    BVHSize = BVHBuilder::build(meshData.vertices, boundingBoxes, BVHNodes);
    checkCUDAError("BVH Build");
    hstScene.createDevData(*this);
    cudaMalloc(&devScene, sizeof(DevScene));
    cudaMemcpyHostToDev(devScene, &hstScene, sizeof(DevScene));
    checkCUDAError("Dev Scene");
}

void Scene::clear() {
    hstScene.freeDevData();
    cudaSafeFree(devScene);
}

void Scene::loadModel(const std::string& objId) {
    std::cout << "Scene::Loading MeshData {" << objId << "}..." << std::endl;

    ModelInstance instance;

    std::string line;
    utilityCore::safeGetline(fp_in, line);

    std::string filename = line;
    std::cout << "\tFrom file " << filename << std::endl;
    instance.meshData = Resource::loadModelMeshData(filename);

    //link material
    utilityCore::safeGetline(fp_in, line);
    if (!line.empty() && fp_in.good()) {
        std::vector<std::string> tokens = utilityCore::tokenizeString(line);
        if (materialMap.find(tokens[1]) == materialMap.end()) {
            std::cout << "\tMaterial {" << tokens[1] << "} doesn't exist" << std::endl;
            throw;
        }
        instance.materialId = materialMap[tokens[1]];
        std::cout << "\tLink to Material {" << tokens[1] << "(" << instance.materialId << ")}..." << std::endl;
    }

    //load transformations
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        std::vector<std::string> tokens = utilityCore::tokenizeString(line);

        //load tranformations
        if (tokens[0] == "Translate") {
            instance.translation = glm::vec3(std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]));
            //std::cout << glm::to_string(instance.translation) << "\n";
        }
        else if (tokens[0] == "Rotate") {
            instance.rotation = glm::vec3(std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]));
        }
        else if (tokens[0] == "Scale") {
            instance.scale = glm::vec3(std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]));
        }

        utilityCore::safeGetline(fp_in, line);
    }

    instance.transform = utilityCore::buildTransformationMatrix(
        instance.translation, instance.rotation, instance.scale
    );
    instance.transfInv = glm::inverse(instance.transform);
    instance.normalMat = glm::transpose(glm::mat3(instance.transfInv));

    std::cout << "\tComplete" << std::endl;
    modelInstances.push_back(instance);
}

void Scene::loadCamera() {
    cout << "Loading Camera ..." << endl;
    RenderState& state = this->state;
    Camera& camera = state.camera;
    float fovy;

    //load static properties
    for (int i = 0; i < 7; i++) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "Resolution") == 0) {
            camera.resolution.x = stoi(tokens[1]);
            camera.resolution.y = stoi(tokens[2]);
        }
        else if (tokens[0] == "FovY") {
            fovy = stof(tokens[1]);
        }
        else if (tokens[0] == "LensRadius") {
            camera.lensRadius = stof(tokens[1]);
        }
        else if (tokens[0] == "FocalDist") {
            camera.focalDist = stof(tokens[1]);
        }
        else if (tokens[0] == "Sample") {
            state.iterations = stoi(tokens[1]);
        }
        else if (tokens[0] == "Depth") {
            state.traceDepth = stoi(tokens[1]);
        }
        else if (tokens[0] == "File") {
            state.imageName = tokens[1];
        }
    }

    string line;
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (tokens[0] == "Eye") {
            camera.position = glm::vec3(stof(tokens[1]), stof(tokens[2]), stof(tokens[3]));
        }
        else if (tokens[0] == "LookAt") {
            camera.lookAt = glm::vec3(stof(tokens[1]), stof(tokens[2]), stof(tokens[3]));
        }
        else if (tokens[0] == "Up") {
            camera.up = glm::vec3(stof(tokens[1]), stof(tokens[2]), stof(tokens[3]));
        }

        utilityCore::safeGetline(fp_in, line);
    }

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    fill(state.image.begin(), state.image.end(), glm::vec3());

    cout << "Loaded camera!" << endl;
}

void Scene::loadMaterial(const std::string& materialId) {
    std::cout << "Scene::Loading Material {" << materialId << "}..." << std::endl;
    Material material;

    //load static properties
    for (int i = 0; i < 6; i++) {
        std::string line;
        utilityCore::safeGetline(fp_in, line);
        auto tokens = utilityCore::tokenizeString(line);
        if (tokens[0] == "Type") {
            std::cout << "\t\t[Type " << tokens[1] << "]" << std::endl;
            material.type = MaterialTypeTokenMap[tokens[1]];
        }
        else if (tokens[0] == "BaseColor") {
            glm::vec3 baseColor(std::stof(tokens[1]), std::stof(tokens[2]), std::stof(tokens[3]));
            material.baseColor = baseColor;
        }
        else if (tokens[0] == "Metallic") {
            material.metallic = std::stof(tokens[1]);
        }
        else if (tokens[0] == "Roughness") {
            material.roughness = std::stof(tokens[1]);
        }
        else if (tokens[0] == "Ior") {
            material.ior = std::stof(tokens[1]);
        }
        else if (tokens[0] == "Emittance") {
            material.emittance = std::stof(tokens[1]);
        }
    }
    materialMap[materialId] = materials.size();
    materials.push_back(material);
    std::cout << "\tComplete" << std::endl;
}
#pragma endregion

#pragma region DevScene
void DevScene::createDevData(Scene& scene) {
    // Put all texture devData in a big buffer
    // and setup device texture objects to manage
    std::vector<DevTextureObj> textureObjs;

    size_t textureTotalSize = 0;
    for (auto tex : scene.textures) {
        textureTotalSize += tex->byteSize();
    }
    cudaMalloc(&devTextureData, textureTotalSize);

    size_t textureOffset = 0;
    for (auto tex : scene.textures) {
        cudaMemcpy(devTextureData + textureOffset, tex->data(), tex->byteSize(), cudaMemcpyKind::cudaMemcpyHostToDevice);
        textureObjs.push_back({ tex, devTextureData + textureOffset });
        textureOffset += tex->byteSize();
    }
    cudaMalloc(&devTextureObjs, textureObjs.size() * sizeof(DevTextureObj));
    cudaMemcpyHostToDev(devTextureObjs, textureObjs.data(), textureObjs.size() * sizeof(DevTextureObj));
    checkCUDAError("DevScene::texture");

    cudaMalloc(&devMaterials, byteSizeOf(scene.materials));
    cudaMemcpyHostToDev(devMaterials, scene.materials.data(), byteSizeOf(scene.materials));
    checkCUDAError("DevScene::materials");

    cudaMalloc(&devMaterialIds, byteSizeOf(scene.materialIds));
    cudaMemcpyHostToDev(devMaterialIds, scene.materialIds.data(), byteSizeOf(scene.materialIds));
    checkCUDAError("DevScene::materialIds");

    cudaMalloc(&devVertices, byteSizeOf(scene.meshData.vertices));
    cudaMemcpyHostToDev(devVertices, scene.meshData.vertices.data(), byteSizeOf(scene.meshData.vertices));
    checkCUDAError("DevScene::vertices");

    cudaMalloc(&devNormals, byteSizeOf(scene.meshData.normals));
    cudaMemcpyHostToDev(devNormals, scene.meshData.normals.data(), byteSizeOf(scene.meshData.normals));
    checkCUDAError("DevScene::normals");

    cudaMalloc(&devTexcoords, byteSizeOf(scene.meshData.texcoords));
    cudaMemcpyHostToDev(devTexcoords, scene.meshData.texcoords.data(), byteSizeOf(scene.meshData.texcoords));
    checkCUDAError("DevScene::texcoords");

    cudaMalloc(&devBoundingBoxes, byteSizeOf(scene.boundingBoxes));
    cudaMemcpyHostToDev(devBoundingBoxes, scene.boundingBoxes.data(), byteSizeOf(scene.boundingBoxes));
    checkCUDAError("DevScene::boundingBoxes");

    for (int i = 0; i < NUM_FACES; i++) {
        cudaMalloc(&devBVHNodes[i], byteSizeOf(scene.BVHNodes[i]));
        cudaMemcpyHostToDev(devBVHNodes[i], scene.BVHNodes[i].data(), byteSizeOf(scene.BVHNodes[i]));
    }
    BVHSize = scene.BVHSize;
    checkCUDAError("DevScene::BVHNodes[6]");
}

void DevScene::freeDevData() {
    cudaSafeFree(devTextureData);
    cudaSafeFree(devTextureObjs);
    cudaSafeFree(devMaterials);
    cudaSafeFree(devMaterialIds);

    cudaSafeFree(devVertices);
    cudaSafeFree(devNormals);
    cudaSafeFree(devTexcoords);
    cudaSafeFree(devBoundingBoxes);

    for (int i = 0; i < NUM_FACES; i++) {
        cudaSafeFree(devBVHNodes[i]);
    }
}

__device__ int DevScene::getMTBVHId(glm::vec3 dir) {
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

__device__ void DevScene::getIntersecGeomInfo(int primId, glm::vec2 bary, Intersection& intersec) {
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
    intersec.texcoord = tb * bary.x + tc * bary.y + ta * (1.f - bary.x - bary.y);
}

__device__ bool DevScene::intersectPrimitive(int primId, Ray ray, float& dist, glm::vec2& bary) {
    glm::vec3 va = devVertices[primId * 3 + 0];
    glm::vec3 vb = devVertices[primId * 3 + 1];
    glm::vec3 vc = devVertices[primId * 3 + 2];

    if (!intersectTriangle(ray, va, vb, vc, bary, dist)) {
        return false;
    }
    glm::vec3 hitPoint = vb * bary.x + vc * bary.y + va * (1.f - bary.x - bary.y);
    return true;
}

__device__ bool DevScene::intersectPrimitiveDetailed(int primId, Ray ray, Intersection& intersec) {
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
    intersec.texcoord = tb * bary.x + tc * bary.y + ta * (1.f - bary.x - bary.y);
    return true;
}

__device__ void DevScene::intersect(Ray ray, Intersection& intersec) {
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
        intersec.primitive = closestPrimId;
        intersec.inDir = -ray.direction;
        intersec.materialId = devMaterialIds[closestPrimId];
    }
    else {
        intersec.primitive = NullPrimitive;
    }
}

__device__ void DevScene::debugIntersect(Ray ray, Intersection& intersec) {
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
    intersec.primitive = maxDepth;
}
#pragma endregion

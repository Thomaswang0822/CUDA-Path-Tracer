#include "bvh.h"
#include "materials.h"
#include "mesh.h"
#include "scene.h"
#include <iostream>
#include <map>
#include <string>
#include "intersections.h"

#define SCENE_LIGHT_SINGLE_SIDED true

std::map<std::string, Material::Type> MaterialTypeTokenMap = {
    { "Lambertian", Material::Type::Lambertian},
    { "MetallicWorkflow", Material::Type::MetallicWorkflow },
    { "Dielectric", Material::Type::Dielectric },
    { "Light", Material::Type::Light }
};

#pragma region Scene

/**
 * A "shallow" constructor.
 * Only data directly linked to scene def file is initialized, which include
 * materials, objects (meshes), and camera.
 */
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

/**
 * Finish remaining details of Scene (and its DevScene).
 * 
 * This include:
 * - populate meshData and materialIds;
 * - build BVH (boundingBoxes and BVHNodes);
 * - manage and copy to device memory;
 */
void Scene::buildDevData() {
    // Put all texture devData in a big buffer
    // and setup device texture objects to manage

#if MESH_DATA_INDEXED
#else
    int primId = 0;
    for (const auto& inst : modelInstances) {
        // grab material info
        const Material& material = materials[inst.materialId];
        glm::vec3 radianceUnitArea = material.baseColor * material.emittance;
        float powerUnitArea = mathUtil::luminance(radianceUnitArea);

        for (size_t i = 0; i < inst.meshData->vertices.size(); i++) {
            meshData.vertices.push_back(glm::vec3(inst.transform * glm::vec4(inst.meshData->vertices[i], 1.f)));
            meshData.normals.push_back(glm::normalize(inst.normalMat * inst.meshData->normals[i]));
            meshData.texcoords.push_back(inst.meshData->texcoords[i]);
            if (i % 3 == 0) {
                materialIds.push_back(inst.materialId);
            }
            else if (i % 3 == 2 && material.type == Material::Type::Light) {
                glm::vec3 v0 = meshData.vertices[i - 2];
                glm::vec3 v1 = meshData.vertices[i - 1];
                glm::vec3 v2 = meshData.vertices[i - 0];
                float area = mathUtil::triangleArea(v0, v1, v2);
                float power = powerUnitArea * area;

                lightPrimIds.push_back(primId);
                lightUnitRadiance.push_back(radianceUnitArea);
                lightPower.push_back(power);
                sumLightPower += power;
                numLightPrims++;
            }

            primId += int(i % 3 == 2);
        }
    }
#endif
    createLightSampler();
    BVHSize = BVHBuilder::build(meshData.vertices, boundingBoxes, BVHNodes);
    checkCUDAError("BVH Build");
    hostScene.createDevData(*this);
    cudaMalloc(&devScene, sizeof(DevScene));
    cudaMemcpyHostToDev(devScene, &hostScene, sizeof(DevScene));
    checkCUDAError("Allocate device memory and copy everything");

    meshData.clear();
    boundingBoxes.clear();
    BVHNodes.clear();

    lightPrimIds.clear();
    lightPower.clear();
    lightSampler.probTable.clear();
    lightSampler.aliasTable.clear();
}

/**
 * Free memory of CPU Scene and GPU DevScene.
 * But since everything on CPU can be auto-destroyed (e.g. std::vector),
 * we only take care of GPU memory.
 */
void Scene::clear() {
    hostScene.freeDevData();
    cudaSafeFree(devScene);
}

/**
 * Allocate memory of a ModelInstance and populate it.
 */
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

/**
 * Load camera (one def per scene assumed) and precompute several parameters.
 */
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

void Scene::createLightSampler() {
    lightSampler = LightSampler(lightPower);
    std::cout << "[Light sampler size = " << lightPower.size() << "]" << std::endl;
}

/**
 * Load a material.
 * 
 * @note when adding/removing/changing a property,
 * also change scene def and this function.
 */
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
    materialMap[materialId] = int(materials.size());
    materials.push_back(material);
    std::cout << "\tComplete" << std::endl;
}
#pragma endregion

#pragma region DevScene

/**
 * Brainless cudaMalloc() and cudaMemcpy().
 */
void DevScene::createDevData(const Scene& scene) {
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
    BVHSize = int(scene.BVHSize);
    checkCUDAError("DevScene::BVHNodes[6]");

    cudaMalloc(&devLightPrimIds, byteSizeOf(scene.lightPrimIds));
    cudaMemcpyHostToDev(devLightPrimIds, scene.lightPrimIds.data(), byteSizeOf(scene.lightPrimIds));

    cudaMalloc(&devLightUnitRadiance, byteSizeOf(scene.lightUnitRadiance));
    cudaMemcpyHostToDev(devLightUnitRadiance, scene.lightPower.data(), byteSizeOf(scene.lightUnitRadiance));

    cudaMalloc(&devProbTable, byteSizeOf(scene.lightSampler.probTable));
    cudaMemcpyHostToDev(devProbTable, scene.lightSampler.probTable.data(),
        byteSizeOf(scene.lightSampler.probTable));
    
    cudaMalloc(&devAliasTable, byteSizeOf(scene.lightSampler.aliasTable));
    cudaMemcpyHostToDev(devAliasTable, scene.lightSampler.aliasTable.data(),
        byteSizeOf(scene.lightSampler.aliasTable));
    
    numLightPrims = scene.numLightPrims;
    sumLightPowerInv = 1.f / scene.sumLightPower;
    checkCUDAError("DevScene::LightData");
}

/**
 * Brainless cudaSafeFree().
 */
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

    cudaSafeFree(devLightPrimIds);
    cudaSafeFree(devLightUnitRadiance);
    cudaSafeFree(devProbTable);
    cudaSafeFree(devAliasTable);
}

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

__device__ glm::vec3 DevScene::getPrimitiveNormal(const int primId) {
    glm::vec3 v0 = devVertices[primId * 3 + 0];
    glm::vec3 v1 = devVertices[primId * 3 + 1];
    glm::vec3 v2 = devVertices[primId * 3 + 2];
    return glm::normalize(glm::cross(v1 - v0, v2 - v0));
}

/**
 * After intersection test, fetch info of intersected triangle.
 * 
 * @param intersec Output parameter to be updated
 */
__device__ void DevScene::getIntersecGeomInfo(int primId, const glm::vec2 bary, Intersection& intersec) {
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

/**
 * Grab the triangle from the vertices pool and perform ray-triangle test.
 * 
 * @param dist Output parameter to be updated (if a closer hit occurs)
 * @param bary Output parameter to be updated (if a closer hit occurs)
 */
__device__ bool DevScene::intersectPrimitive(int primId, const Ray& ray, float& dist, glm::vec2& bary) {
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

__device__ bool DevScene::intersectPrimitive(int primId, const Ray& ray, float distRange) {
    glm::vec3 va = devVertices[primId * 3 + 0];
    glm::vec3 vb = devVertices[primId * 3 + 1];
    glm::vec3 vc = devVertices[primId * 3 + 2];

    glm::vec2 bary;
    float dist;
    bool hit = intersectTriangle(ray, va, vb, vc, bary, dist);
    return (hit && dist < distRange);
}

/** NOT USED YET */
__device__ bool DevScene::intersectPrimitiveDetailed(int primId, const Ray& ray, Intersection& intersec) {
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

/**
 * Given a ray, find the cloest intersection, if one exist, in the entire scene.
 * 
 * @param intersec Output parameter
 */
__device__ void DevScene::intersect(const Ray& ray, Intersection& intersec) {
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
__device__ bool DevScene::testOcclusion(glm::vec3 x, glm::vec3 y) {
    glm::vec3 dir = glm::normalize(y - x);
    float dist = glm::length(y - x);
    Ray ray = Ray::makeOffsetRay(x, dir);
    bool hit = false;

    MTBVHNode* nodes = devBVHNodes[getMTBVHId(-ray.direction)];
    int node = 0;
    while (node != BVHSize) {
        AABB& bound = devBoundingBoxes[nodes[node].boundingBoxId];
        float boundDist;
        bool boundHit = bound.intersect(ray, boundDist);

        if (boundHit && boundDist < dist) {
            int primId = nodes[node].primitiveId;
            if (primId != NullPrimitive) {
                hit |= intersectPrimitive(primId, ray, dist);
            }
            node++;
        }
        else {
            node = nodes[node].nextNodeIfMiss;
        }
    }
    return hit;
}

/**
 * DEBUG version intersection test.
 * 
 * intersec.primId will be written with #triangles hit
 */
__device__ void DevScene::visualizedIntersect(const Ray& ray, Intersection& intersec) {
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
__device__ float DevScene::sampleDirectLight(glm::vec3 pos, glm::vec4 r, glm::vec3& radiance, glm::vec3& wi) {
    int bucketId = static_cast<int>(r.x * numLightPrims);
    int lightId = (r.y < devProbTable[bucketId]) ? bucketId : devAliasTable[bucketId];
    int primId = devLightPrimIds[lightId];

    glm::vec3 v0 = devVertices[primId * 3 + 0];
    glm::vec3 v1 = devVertices[primId * 3 + 1];
    glm::vec3 v2 = devVertices[primId * 3 + 2];
    glm::vec3 sampled = mathUtil::sampleTriangleUniform(v0, v1, v2, r.z, r.w);

    if (testOcclusion(pos, sampled)) {
        return InvalidPdf;
    }
    glm::vec3 normal = mathUtil::triangleNormal(v0, v1, v2);
    glm::vec3 posToSampled = sampled - pos;

    if (glm::dot(normal, posToSampled) > 0.f) {
        return InvalidPdf;
    }
    radiance = devLightUnitRadiance[lightId];
    wi = glm::normalize(posToSampled);
    return mathUtil::pdfAreaToSolidAngle(mathUtil::luminance(radiance) * sumLightPowerInv, pos, sampled, normal);
}
#pragma endregion

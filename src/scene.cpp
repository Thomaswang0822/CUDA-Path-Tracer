#include "bvh.h"
#include "materials.h"
#include "mesh.h"
#include "scene.h"
#include <iostream>
#include <map>
#include <string>
#include "intersections.h"


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
        Core::safeGetline(fp_in, line);
        if (!line.empty()) {
            std::vector<std::string> tokens = Core::tokenizeString(line);
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
        float powerUnitArea = Math::luminance(radianceUnitArea);

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
                float area = Math::triangleArea(v0, v1, v2);
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
    Cuda::memcpyHostToDev(devScene, &hostScene, sizeof(DevScene));
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
    Cuda::safeFree(devScene);
}

/**
 * Allocate memory of a ModelInstance and populate it.
 */
void Scene::loadModel(const std::string& objId) {
    std::cout << "Scene::Loading MeshData {" << objId << "}..." << std::endl;

    ModelInstance instance;

    std::string line;
    Core::safeGetline(fp_in, line);

    std::string filename = line;
    std::cout << "\tFrom file " << filename << std::endl;
    instance.meshData = Resource::loadModelMeshData(filename);

    //link material
    Core::safeGetline(fp_in, line);
    if (!line.empty() && fp_in.good()) {
        std::vector<std::string> tokens = Core::tokenizeString(line);
        if (materialMap.find(tokens[1]) == materialMap.end()) {
            std::cout << "\tMaterial {" << tokens[1] << "} doesn't exist" << std::endl;
            throw;
        }
        instance.materialId = materialMap[tokens[1]];
        std::cout << "\tLink to Material {" << tokens[1] << "(" << instance.materialId << ")}..." << std::endl;
    }

    //load transformations
    Core::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        std::vector<std::string> tokens = Core::tokenizeString(line);

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

        Core::safeGetline(fp_in, line);
    }

    instance.transform = Core::buildTransformationMatrix(
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
        Core::safeGetline(fp_in, line);
        vector<string> tokens = Core::tokenizeString(line);
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
            Settings::traceDepth = stoi(tokens[1]);
        }
        else if (tokens[0] == "File") {
            state.imageName = tokens[1];
        }
    }

    string line;
    Core::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = Core::tokenizeString(line);
        if (tokens[0] == "Eye") {
            camera.position = glm::vec3(stof(tokens[1]), stof(tokens[2]), stof(tokens[3]));
        }
        else if (tokens[0] == "LookAt") {
            camera.lookAt = glm::vec3(stof(tokens[1]), stof(tokens[2]), stof(tokens[3]));
        }
        else if (tokens[0] == "Up") {
            camera.up = glm::vec3(stof(tokens[1]), stof(tokens[2]), stof(tokens[3]));
        }

        Core::safeGetline(fp_in, line);
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
        Core::safeGetline(fp_in, line);
        auto tokens = Core::tokenizeString(line);
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
    Cuda::memcpyHostToDev(devTextureObjs, textureObjs.data(), textureObjs.size() * sizeof(DevTextureObj));
    checkCUDAError("DevScene::texture");

    cudaMalloc(&devMaterials, Core::byteSizeOf(scene.materials));
    Cuda::memcpyHostToDev(devMaterials, scene.materials.data(), Core::byteSizeOf(scene.materials));
    checkCUDAError("DevScene::materials");

    cudaMalloc(&devMaterialIds, Core::byteSizeOf(scene.materialIds));
    Cuda::memcpyHostToDev(devMaterialIds, scene.materialIds.data(), Core::byteSizeOf(scene.materialIds));
    checkCUDAError("DevScene::materialIds");

    cudaMalloc(&devVertices, Core::byteSizeOf(scene.meshData.vertices));
    Cuda::memcpyHostToDev(devVertices, scene.meshData.vertices.data(), Core::byteSizeOf(scene.meshData.vertices));
    checkCUDAError("DevScene::vertices");

    cudaMalloc(&devNormals, Core::byteSizeOf(scene.meshData.normals));
    Cuda::memcpyHostToDev(devNormals, scene.meshData.normals.data(), Core::byteSizeOf(scene.meshData.normals));
    checkCUDAError("DevScene::normals");

    cudaMalloc(&devTexcoords, Core::byteSizeOf(scene.meshData.texcoords));
    Cuda::memcpyHostToDev(devTexcoords, scene.meshData.texcoords.data(), Core::byteSizeOf(scene.meshData.texcoords));
    checkCUDAError("DevScene::texcoords");

    cudaMalloc(&devBoundingBoxes, Core::byteSizeOf(scene.boundingBoxes));
    Cuda::memcpyHostToDev(devBoundingBoxes, scene.boundingBoxes.data(), Core::byteSizeOf(scene.boundingBoxes));
    checkCUDAError("DevScene::boundingBoxes");

    for (int i = 0; i < NUM_FACES; i++) {
        cudaMalloc(&devBVHNodes[i], Core::byteSizeOf(scene.BVHNodes[i]));
        Cuda::memcpyHostToDev(devBVHNodes[i], scene.BVHNodes[i].data(), Core::byteSizeOf(scene.BVHNodes[i]));
    }
    BVHSize = int(scene.BVHSize);
    checkCUDAError("DevScene::BVHNodes[6]");

    cudaMalloc(&devLightPrimIds, Core::byteSizeOf(scene.lightPrimIds));
    Cuda::memcpyHostToDev(devLightPrimIds, scene.lightPrimIds.data(), Core::byteSizeOf(scene.lightPrimIds));

    cudaMalloc(&devLightUnitRadiance, Core::byteSizeOf(scene.lightUnitRadiance));
    Cuda::memcpyHostToDev(devLightUnitRadiance, scene.lightUnitRadiance.data(), Core::byteSizeOf(scene.lightUnitRadiance));

    cudaMalloc(&devProbTable, Core::byteSizeOf(scene.lightSampler.probTable));
    Cuda::memcpyHostToDev(devProbTable, scene.lightSampler.probTable.data(),
        Core::byteSizeOf(scene.lightSampler.probTable));
    
    cudaMalloc(&devAliasTable, Core::byteSizeOf(scene.lightSampler.aliasTable));
    Cuda::memcpyHostToDev(devAliasTable, scene.lightSampler.aliasTable.data(),
        Core::byteSizeOf(scene.lightSampler.aliasTable));
    
    numLightPrims = scene.numLightPrims;
    sumLightPowerInv = 1.f / scene.sumLightPower;
    checkCUDAError("DevScene::LightData");
}

/**
 * Brainless Cuda::safeFree().
 */
void DevScene::freeDevData() {
    Cuda::safeFree(devTextureData);
    Cuda::safeFree(devTextureObjs);
    Cuda::safeFree(devMaterials);
    Cuda::safeFree(devMaterialIds);

    Cuda::safeFree(devVertices);
    Cuda::safeFree(devNormals);
    Cuda::safeFree(devTexcoords);
    Cuda::safeFree(devBoundingBoxes);

    for (int i = 0; i < NUM_FACES; i++) {
        Cuda::safeFree(devBVHNodes[i]);
    }

    Cuda::safeFree(devLightPrimIds);
    Cuda::safeFree(devLightUnitRadiance);
    Cuda::safeFree(devProbTable);
    Cuda::safeFree(devAliasTable);
}

//__device__ int DevScene::getMTBVHId(glm::vec3 dir)

//__device__ glm::vec3 DevScene::getPrimitiveNormal(const int primId)

//__device__ void DevScene::getIntersecGeomInfo(int primId, const glm::vec2 bary, Intersection& intersec)

//__device__ bool DevScene::intersectPrimitive(int primId, const Ray& ray, float& dist, glm::vec2& bary)

//__device__ bool DevScene::intersectPrimitive(int primId, const Ray& ray, float distRange)

//__device__ bool DevScene::intersectPrimitiveDetailed(int primId, const Ray& ray, Intersection& intersec)

//__device__ void DevScene::intersect(const Ray& ray, Intersection& intersec)

//__device__ bool DevScene::testOcclusion(glm::vec3 x, glm::vec3 y)

//__device__ void DevScene::visualizedIntersect(const Ray& ray, Intersection& intersec)

//__device__ float DevScene::sampleDirectLight(glm::vec3 pos, glm::vec4 r, glm::vec3& radiance, glm::vec3& wi)
#pragma endregion

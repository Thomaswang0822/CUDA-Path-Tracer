#include <cuda_runtime.h>
#include <cmath>

#include "sceneStructs.h"
#include "material.h"
#include "scene.h"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "mathUtil.h"
#include "sampler.h"
#include "restir.h"

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
    glm::vec3* Image, int toneMapping
) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);

        // Tonemapping and gamma correction
        glm::vec3 color = Image[index];

        switch (toneMapping) {
        case ToneMapping::Filmic:
            color = Math::filmic(color);
            break;
        case ToneMapping::ACES:
            color = Math::ACES(color);
            break;
        case ToneMapping::None:
            break;
        }
        color = Math::correctGamma(color);
        glm::ivec3 iColor = glm::clamp(glm::ivec3(color * 255.f), glm::ivec3(0), glm::ivec3(255));

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = iColor.x;
        pbo[index].y = iColor.y;
        pbo[index].z = iColor.z;
    }
}

#if ENABLE_GBUFFER
static Intersection* devGBuffer = nullptr;
#endif

__global__ void renderGBuffer(DevScene* scene, Camera cam, Intersection *GBuffer) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    if (idx >= cam.resolution.x || idy >= cam.resolution.y) {
        return;
    }

    float aspect = float(cam.resolution.x) / cam.resolution.y;
    float tanFovY = glm::tan(glm::radians(cam.fov.y));
    glm::vec2 pixelSize = 1.f / glm::vec2(cam.resolution);
    glm::vec2 scr = glm::vec2(idx, idy) * pixelSize;
    glm::vec2 ruv = scr + pixelSize * glm::vec2(.5f);
    ruv = 1.f - ruv * 2.f;

    glm::vec3 pLens(0.f);
    glm::vec3 pFocusPlane = glm::vec3(ruv * glm::vec2(aspect, 1.f) * tanFovY, 1.f) * cam.focalDist;
    glm::vec3 dir = pFocusPlane - pLens;

    Ray ray;
    ray.direction = glm::normalize(glm::mat3(cam.right, cam.up, cam.view) * dir);
    ray.origin = cam.position + cam.right * pLens.x + cam.up * pLens.y;

    Intersection intersec;
    scene->intersect(ray, intersec);

    if (intersec.primId != NullPrimitive) {
        if (scene->materials[intersec.matId].type == Material::Type::Light) {
#if SCENE_LIGHT_SINGLE_SIDED
            if (glm::dot(intersec.norm, ray.direction) < 0.f) {
                intersec.primId = NullPrimitive;
            }
#endif
        }
        else {
            intersec.wo = -ray.direction;
        }
    }
    GBuffer[idy * cam.resolution.x + idx] = intersec;
}

void pathTraceInit(Scene* scene) {
    //hstScene = scene;

#if ENABLE_GBUFFER
    cudaMalloc(&devGBuffer, pixelcount * sizeof(Intersection));
    const int BlockSize = 8;
    dim3 blockSize(BlockSize, BlockSize);

    dim3 blockNum((cam.resolution.x + BlockSize - 1) / BlockSize,
        (cam.resolution.y + BlockSize - 1) / BlockSize
    );
    renderGBuffer<<<blockNum, blockSize>>>(hstScene->devScene, cam, devGBuffer);
    checkCUDAError("GBuffer");
    std::cout << "[GBuffer generated]" << std::endl;
#endif
}

void pathTraceFree() {
#if ENABLE_GBUFFER
    CUDA::safeFree(devGBuffer);
#endif
}

__global__ void previewGBuffer(int iter, DevScene* scene, const Camera cam, glm::vec3* image, int kind) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= cam.resolution.x || y >= cam.resolution.y) {
        return;
    }
    int index = y * cam.resolution.x + x;
    Sampler rng = makeSeededRandomEngine(iter, index, 0, scene->sampleSequence);

    Ray ray = cam.sample(x, y, sample4D(rng));
    Intersection intersec;
    scene->intersect(ray, intersec);

    if (kind == 0) {
        image[index] = intersec.pos;
    }
    else if (kind == 1) {
        image[index] = (intersec.norm + 1.f) * .5f;
    }
    else if (kind == 2) {
        image[index] = glm::vec3(intersec.uv, 1.f);
    }
}

__global__ void kernelPT(int iter, int maxDepth, DevScene* scene, Camera cam, glm::vec3* image) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= cam.resolution.x || y >= cam.resolution.y) {
        return;
    }
    glm::vec3 accRadiance(0.f);

    int index = y * cam.resolution.x + x;
    Sampler rng = makeSeededRandomEngine(iter, index, 0, scene->sampleSequence);

    Ray ray = cam.sample(x, y, sample4D(rng));
    Intersection intersec;
    scene->intersect(ray, intersec);

    if (intersec.primId == NullPrimitive) {
        if (scene->envMap != nullptr) {
            glm::vec2 uv = Math::toPlane(ray.direction);
            accRadiance += scene->envMap->linearSample(uv);
        }
        goto WriteRadiance;
    }

    Material material = scene->getTexturedMaterialAndSurface(intersec);

    // camera ray hits a light
    if (material.type == Material::Type::Light) {
        if (glm::dot(intersec.norm, ray.direction) > 0.f) {
            accRadiance = material.baseColor;
        }
        goto WriteRadiance;
    }

    glm::vec3 throughput(1.f);
    intersec.wo = -ray.direction;

    for (int depth = 1; depth <= maxDepth; depth++) {
        bool deltaBSDF = (material.type == Material::Type::Dielectric);

        if (material.type != Material::Type::Dielectric && glm::dot(intersec.norm, intersec.wo) < 0.f) {
            intersec.norm = -intersec.norm;
        }

        // NEE (explicit light sampling) with MIS weight
        if (!deltaBSDF) {
            glm::vec3 radiance;
            glm::vec3 wi;
            float lightPdf = scene->sampleDirectLight(intersec.pos, sample4D(rng), radiance, wi);

            if (lightPdf > 0) {
                float BSDFPdf = material.pdf(intersec.norm, intersec.wo, wi);
                accRadiance += throughput * material.BSDF(intersec.norm, intersec.wo, wi) *
                    radiance * Math::satDot(intersec.norm, wi) / lightPdf * Math::balanceHeuristic(lightPdf, BSDFPdf);
            }
        }

        // BSDF sampling
        BSDFSample sample;
        material.sample(intersec.norm, intersec.wo, sample3D(rng), sample);

        if (sample.type == BSDFSampleType::Invalid) {
            // Terminate path if sampling fails
            break;
        }
        else if (sample.pdf < 1e-8f) {
            // also for numerically unstable pdf
            break;
        }

        bool deltaSample = (sample.type & BSDFSampleType::Specular);
        throughput *= sample.bsdf / sample.pdf *
            (deltaSample ? 1.f : Math::absDot(intersec.norm, sample.dir));
        
        // bounce the ray to the sampled direction
        ray = makeOffsetedRay(intersec.pos, sample.dir);
        // intersection test and update `intersec` data
        glm::vec3 curPos = intersec.pos;
        scene->intersect(ray, intersec);
        intersec.wo = -ray.direction;

        // break from PT loop if bounced ray hits background
        if (intersec.primId == NullPrimitive) {
            if (scene->envMap != nullptr) {
                glm::vec3 radiance = scene->envMap->linearSample(Math::toPlane(ray.direction))
                    * throughput;

                float weight = deltaSample ? 1.f :
                    Math::powerHeuristic(sample.pdf, scene->environmentMapPdf(ray.direction));

                accRadiance += radiance * weight;
            }
            break;
        }
        material = scene->getTexturedMaterialAndSurface(intersec);

        // if bounced ray hit a light,
        // accumulate radiance and break
        if (material.type == Material::Type::Light) {
#if SCENE_LIGHT_SINGLE_SIDED
            if (glm::dot(intersec.norm, ray.direction) < 0.f) {
                break;
            }
#endif
            glm::vec3 radiance = material.baseColor;

            float lightPdf = Math::pdfAreaToSolidAngle(
                Math::luminance(radiance) * 2.f * glm::pi<float>() * scene->getPrimitiveArea(intersec.primId) * scene->sumLightPowerInv,
                curPos, intersec.pos, intersec.norm
            );

            float weight = deltaSample ? 1.f : Math::powerHeuristic(sample.pdf, lightPdf);
            accRadiance += radiance * throughput * weight;
            break;
        }
    }  // end of PT loop
WriteRadiance:
    if (isnan(accRadiance.x) || isnan(accRadiance.y) || isnan(accRadiance.z) ||
        isinf(accRadiance.x) || isinf(accRadiance.y) || isinf(accRadiance.z)) {
        return;
    }
    image[index] = (image[index] * float(iter) + accRadiance) / float(iter + 1);
}

__global__ void BVHVisualize(int iter, DevScene* scene, Camera cam, glm::vec3* image) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= cam.resolution.x || y >= cam.resolution.y) {
        return;
    }
    int index = y * cam.resolution.x + x;

    Sampler rng = makeSeededRandomEngine(iter, index, 0, scene->sampleSequence);
    Ray ray = cam.sample(x, y, sample4D(rng));

    Intersection intersec;
    scene->visualizedIntersect(ray, intersec);

    float logDepth = 0.f;
    int size = scene->BVHSize;
    while (size) {
        logDepth += 1.f;
        size >>= 1;
    }
    image[index] = glm::vec3(float(intersec.primId) / logDepth * .06f);
}

void pathTrace(uchar4* pbo, glm::vec3* devImage, Scene* hstScene) {
    const Camera& cam = hstScene->camera;
    const int pixelCount = cam.resolution.x * cam.resolution.y;

    const int BlockSizeSinglePTX = 8;
    const int BlockSizeSinglePTY = 8;
    int blockNumSinglePTX = (cam.resolution.x + BlockSizeSinglePTX - 1) / BlockSizeSinglePTX;
    int blockNumSinglePTY = (cam.resolution.y + BlockSizeSinglePTY - 1) / BlockSizeSinglePTY;

    dim3 singlePTBlockNum(blockNumSinglePTX, blockNumSinglePTY);
    dim3 singlePTBlockSize(BlockSizeSinglePTX, BlockSizeSinglePTY);

    switch (Settings::tracer)
    {
    case Tracer::PathTrace:
        kernelPT << <singlePTBlockNum, singlePTBlockSize >> > (State::iteration, Settings::traceDepth, hstScene->devScene, cam, devImage);
        break;
    case Tracer::BVHVisualize:
        BVHVisualize << <singlePTBlockNum, singlePTBlockSize >> > (State::iteration, hstScene->devScene, cam, devImage);
        break;
    case Tracer::GBufferPreview:
        previewGBuffer << <singlePTBlockNum, singlePTBlockSize >> > (State::iteration, hstScene->devScene, cam, devImage,
            Settings::GBufferPreviewOpt);
        break;
    /*case Tracer::ReSTIR_DI:
        kernelRIS << <singlePTBlockNum, singlePTBlockSize >> > (State::iteration,
            ReSTIRSettings::M_Light, ReSTIRSettings::M_BSDF,
            hstScene->devScene, cam, devImage);*/
    default:
        break;
    }

    // 2D block sending pixel data
    const dim3 blockSize2D(8, 8);
    const dim3 blocksPerGrid2D(
        (cam.resolution.x + blockSize2D.x - 1) / blockSize2D.x,
        (cam.resolution.y + blockSize2D.y - 1) / blockSize2D.y);
    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2D, blockSize2D>>>(pbo, cam.resolution, devImage, Settings::toneMapping);

    // Retrieve image from GPU
    CUDA::copyDevToHost(hstScene->state.image.data(), devImage,
        pixelCount * sizeof(glm::vec3));

    checkCUDAError("pathTrace");
}
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
    int iter, glm::vec3* Image, int toneMapping) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

#if AVERAGE_SPP
    float spp = static_cast<float>(iter);
#else
    float spp = 1.0f;
#endif // AVERAGE_SPP

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);

        // Tonemapping and gamma correction
        glm::vec3 color = Image[index] / spp;

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

static Scene* hstScene = nullptr;
static glm::vec3* devImage = nullptr;

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
    hstScene = scene;

    const Camera& cam = hstScene->camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    devImage = CUDA::safeMalloc<glm::vec3>(pixelcount);

    checkCUDAError("pathTraceInit");

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
    CUDA::safeFree(devImage);  // no-op if devImage is null
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
        image[index] += intersec.pos;
    }
    else if (kind == 1) {
        if (intersec.primId != NullPrimitive) {
            const Material m = scene->getTexturedMaterialAndSurface(intersec);
            // put 
        }
        image[index] += (intersec.norm + 1.f) * .5f;
    }
    else if (kind == 2) {
        image[index] += glm::vec3(intersec.uv, 1.f);
    }
}

__global__ void shadeReSTIR_DI(
    int iter, 
    int M_Light, int M_BSDF,
    DevScene* scene, 
    const Camera cam, 
    glm::vec3* image
) {
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

    // don't bother with ReSTIR when ray misses
    if (intersec.primId == NullPrimitive) {
        if (scene->envMap != nullptr) {
            glm::vec2 uv = Math::toPlane(ray.direction);
            accRadiance += scene->envMap->linearSample(uv);
        }
        goto WriteRadiance;
    }

    Material material = scene->getTexturedMaterialAndSurface(intersec);
    intersec.wo = -ray.direction;  // another killing bug

    // same for camera ray hitting a light
    if (material.type == Material::Type::Light) {
        if (glm::dot(intersec.norm, ray.direction) > 0.f) {
            accRadiance = material.baseColor;
        }
        goto WriteRadiance;
    }

    bool deltaBSDF = (material.type == Material::Type::Dielectric);

    if (material.type != Material::Type::Dielectric && glm::dot(intersec.norm, intersec.wo) < 0.f) {
        intersec.norm = -intersec.norm;
    }

#pragma region version1
    /**
     * DI version 1: only sample light.
     *
     * p^ is full f without visibility = brdf * Le * <wi, normal>
     * p  is proportional to Le; let p = Le
     *
     * @note Everything in solid angle measure (not the same as DI paper)
     */
    /*
    if (!deltaBSDF) {
        ReSTIR::Reservoir r;
        glm::vec3 p_hat;     // Li * BSDF * cos
        float p;             // "cheap" pdf
        float w_proposal;    // p^/p

        for (int i = 0; i < M_Light; i++) {
            glm::vec3 radiance;  // Li
            glm::vec3 wi;        // sampled direction
            glm::vec3 xi;        // wi turned into position (for occlucsion test)
            // generate wi
            p = scene->sampleDirectLight_Cheap(intersec.pos, sample4D(rng), radiance, wi, xi);
            if (isnan(p) || isinf(p) || p < 0.f) {
                // a invalid sample, thus weight = 0.
                w_proposal = 0.f;
            }
            else {
                p_hat = radiance * material.BSDF(intersec.norm, intersec.wo, wi) * Math::satDot(intersec.norm, wi);
                w_proposal = ReSTIR::toScalar(p_hat) / (p * M_Light);  // Note Eq 3.2
            }

            // Update reservoir even if sampling failed, otherwise biased.
            // See Course Notes p10-11
            r.update({ wi, xi, p_hat }, w_proposal, sample1D(rng));
        }

        // Algorithm 3 line 8; reuse some variables
        glm::vec3 wi = r.y.dir;
        glm::vec3 xi = r.y.position;
        p_hat = r.y.targetFunc;
        float p_hat_q = ReSTIR::toScalar(p_hat);
        r.W = r.w_sum / p_hat_q;  // Note Eq 3.2
        
        // Algorithm 3 line 11
        glm::vec3 f_q = r.y.targetFunc *
            static_cast<float>(!scene->testOcclusion(intersec.pos, xi));
        accRadiance = f_q * r.W;
    }
    // don't consider difficult Dielectric
    */
#pragma endregion

#pragma region version2
    /**
     * DI version 2: combine light sampling and BSDF sampling with MIS.
     */

    
    ReSTIR::Reservoir r;
    glm::vec3 p_hat;     // Li * BSDF * cos
    float pLight, pBSDF; // "cheap" pdf, light or BSDF
    float w_proposal;    // p^ * mi * W_xi, where W_xi is just 1/p, and
    float mi;            // mi is MIS weight = p_{1/2} / (M1 * p_1 + M2 * p_2)

    // Light Sampling for non-dielectric
    if (!deltaBSDF) {
        for (int i = 0; i < M_Light; i++) {
            glm::vec3 radiance(0.f);  // Li
            glm::vec3 wi;        // sampled direction
            glm::vec3 xi;        // wi turned into position (for occlucsion test)
            // generate wi
            pLight = scene->sampleDirectLight_Cheap(intersec.pos, sample4D(rng), radiance, wi, xi);
            if (isnan(pLight) || isinf(pLight) || pLight < 0.f) {
                // a invalid sample, thus weight = 0.
                w_proposal = 0.f;
            }
            else {
                p_hat = radiance * material.BSDF(intersec.norm, intersec.wo, wi) * Math::satDot(intersec.norm, wi);
                pBSDF = material.pdf(intersec.norm, intersec.wo, wi);
                mi = ReSTIR::MIS_BalanceWeight(pLight, pBSDF, M_Light, M_BSDF);
                w_proposal = ReSTIR::toScalar(p_hat) * mi / pLight;
            }

            // Update reservoir even if sampling failed, otherwise biased.
            // See Course Notes p10-11
            r.update({ wi, xi, p_hat }, w_proposal, sample1D(rng));
        }
    }
    // BSDF sampling
    BSDFSample sampledInfo;
    bool deltaSample;
    for (int i = 0; i < M_BSDF; i++) {
        glm::vec3 radiance(0.f);  // Li
        glm::vec3 wi;        // sampled direction
        glm::vec3 xi;        // wi turned into position (for occlucsion test)
        sampledInfo.invalidate();
        // generate wi
        material.sample(intersec.norm, intersec.wo, sample3D(rng), sampledInfo);
        wi = sampledInfo.dir; pBSDF = sampledInfo.pdf;
        deltaSample = (sampledInfo.type & BSDFSampleType::Specular);
        /// xi is only used for testOcclusion() for the only survived sample y;
        /// If Ray(intersec.pos, wi) failed to hit a light, then
        /// radiance (updated by lightPdf) thus p_hat thus w_proposal = 0 => 
        /// sampled_i is garabage and we don't care;
        /// If the Ray does hit a light, we want to "skip" the intersection test,
        /// by making xi really close to shading point.
        xi = intersec.pos + 1e-6f * wi;
        if (isnan(pBSDF) || isinf(pBSDF) || pBSDF < 0.f) {
            // a invalid sample, thus weight = 0.
            w_proposal = 0.f;
        }
        else {            
            // Heavy-lifting to find pLight counterpart; moved to a member function of DevScene
            pLight = scene->lightPdf(intersec.pos, wi, radiance);
            // Should know radiance before computing p_hat
            // Tricky bug: shouldn't recompute BSDF (will be 0 for dielectric).
            // Careless bug: should stop MIS for dielectric.
            p_hat = radiance * sampledInfo.bsdf *
                (deltaSample ? 1.f : Math::satDot(intersec.norm, wi));
            pLight = deltaSample ? 0.f : pLight;
            mi = ReSTIR::MIS_BalanceWeight(pBSDF, pLight, M_BSDF, M_Light);
            w_proposal = ReSTIR::toScalar(p_hat) * mi / pBSDF;
        }

        // Update reservoir even if sampling failed, otherwise biased.
        // See Course Notes p10-11
        r.update({ wi, xi, p_hat }, w_proposal, sample1D(rng));
    }

    // Algorithm 3 line 8; reuse some variables
    glm::vec3 wi = r.y.dir;
    glm::vec3 xi = r.y.position;
    p_hat = r.y.targetFunc;
    float p_hat_q = ReSTIR::toScalar(p_hat);
    r.W = r.w_sum / p_hat_q;  // Note Eq 3.2

    // Algorithm 3 line 11
    glm::vec3 f_q = r.y.targetFunc *
        static_cast<float>(!scene->testOcclusion(intersec.pos, xi));
    accRadiance = f_q * r.W;

    //accRadiance = glm::vec3(r.w_sum);  // 0 => every w_proposal is 0
    
#pragma endregion

WriteRadiance:
    if (isnan(accRadiance.x) || isnan(accRadiance.y) || isnan(accRadiance.z) ||
        isinf(accRadiance.x) || isinf(accRadiance.y) || isinf(accRadiance.z)) {
        return;
    }
#if AVERAGE_SPP
    image[index] += accRadiance;
#else
    image[index] = accRadiance;
#endif // AVERAGE_SPP
}

__global__ void singleKernelPT(int iter, int maxDepth, DevScene* scene, Camera cam, glm::vec3* image) {
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
#if AVERAGE_SPP
    image[index] += accRadiance;
#else
    image[index] = accRadiance;
#endif // AVERAGE_SPP
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
    image[index] += glm::vec3(float(intersec.primId) / logDepth * .06f);
}

void pathTrace(uchar4* pbo, int frame, int iter) {
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
    case Tracer::SingleKernel:
        singleKernelPT << <singlePTBlockNum, singlePTBlockSize >> > (iter, Settings::traceDepth, hstScene->devScene, cam, devImage);
        break;
    case Tracer::BVHVisualize:
        BVHVisualize << <singlePTBlockNum, singlePTBlockSize >> > (iter, hstScene->devScene, cam, devImage);
        break;
    case Tracer::GBufferPreview:
        previewGBuffer << <singlePTBlockNum, singlePTBlockSize >> > (iter, hstScene->devScene, cam, devImage,
            Settings::GBufferPreviewOpt);
        break;
    case Tracer::ReSTIR_DI:
        shadeReSTIR_DI << <singlePTBlockNum, singlePTBlockSize >> > (iter,
            ReSTIRSettings::M_Light, ReSTIRSettings::M_BSDF,
            hstScene->devScene, cam, devImage);
    default:
        break;
    }

    // 2D block sending pixel data
    const dim3 blockSize2D(8, 8);
    const dim3 blocksPerGrid2D(
        (cam.resolution.x + blockSize2D.x - 1) / blockSize2D.x,
        (cam.resolution.y + blockSize2D.y - 1) / blockSize2D.y);
    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2D, blockSize2D>>>(pbo, cam.resolution, iter, devImage, Settings::toneMapping);

    // Retrieve image from GPU
    CUDA::copyDevToHost(hstScene->state.image.data(), devImage,
        pixelCount * sizeof(glm::vec3));

    checkCUDAError("pathTrace");
}
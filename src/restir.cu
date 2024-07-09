#include <cuda_runtime.h>
#include <cmath>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include "sceneStructs.h"
#include "material.h"
#include "scene.h"
#include "utilities.h"
#include "intersections.h"
#include "mathUtil.h"
#include "sampler.h"
#include "restir.h"

#ifdef __INTELLISENSE__
void __syncthreads() {};
#endif

extern __global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
    glm::vec3* Image, int toneMapping
);

////////////////////////////////////
// Reservoir GBuffer memory used by ReSTIR
Reservoir* spatialRSV = nullptr;
////////////////////////////////////


void ReSTIR::init(Scene* hostScene) {
    const Camera& cam = hostScene->camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    spatialRSV = CUDA::safeMalloc<Reservoir>(pixelcount);
}

void ReSTIR::free() {
    CUDA::safeFree(spatialRSV);
}

void ReSTIR::reset() {

}

__global__ void kernelRIS(
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
        Reservoir r;
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

    Reservoir r;
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

    // Algorithm 3 line 8
    float p_hat_q = ReSTIR::toScalar(r.y.targetFunc);
    r.W = r.w_sum / p_hat_q;  // Note Eq 3.2

    // Algorithm 3 line 11
    glm::vec3 f_q = r.y.targetFunc *
        static_cast<float>(!scene->testOcclusion(intersec.pos, r.y.position));
    accRadiance = f_q * r.W;

#pragma endregion

WriteRadiance:
    if (isnan(accRadiance.x) || isnan(accRadiance.y) || isnan(accRadiance.z) ||
        isinf(accRadiance.x) || isinf(accRadiance.y) || isinf(accRadiance.z)) {
        accRadiance = glm::vec3(0.f);
    }
    image[index] = (image[index] * float(iter) + accRadiance) / float(iter + 1);
}

__global__ void kernelSpatial(
    int iter,
    int M_Light, int M_BSDF,
    DevScene* scene,
    const Camera cam,
    glm::vec3* image,
    Reservoir* reservoirs
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

    Reservoir r;
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

    // Evaluate visibility to the initial candidate
    bool visible = !scene->testOcclusion(intersec.pos, r.y.position);
    r.W = visible ? r.W : 0.f;

    // Store to current-frame Reservoir GBuffer
    reservoirs[index] = r;
    __syncthreads();

    // pick 5 neighbors
    for (int i = 0; i < 5; i++) {
        int index_qi = ReSTIR::sampleNeighbor(x, y, cam.resolution, sample2D(rng));
        if (index_qi == INVALID_INDEX || index_qi == index) {
            continue;
        }
        // combine
        // TODO: check and invalidate with "unbias" tricks, DI paper Sec5
        r.combine(reservoirs[index_qi], sample1D(rng));
    }
    r.W = r.w_sum / ReSTIR::toScalar(r.y.targetFunc);

    // Algorithm 3 line 11
    glm::vec3 f_q = r.y.targetFunc *
        static_cast<float>(!scene->testOcclusion(intersec.pos, xi));
    accRadiance = f_q * r.W;


WriteRadiance:
    if (isnan(accRadiance.x) || isnan(accRadiance.y) || isnan(accRadiance.z) ||
        isinf(accRadiance.x) || isinf(accRadiance.y) || isinf(accRadiance.z)) {
        accRadiance = glm::vec3(0.f);
    }
    image[index] = (image[index] * float(iter) + accRadiance) / float(iter + 1);
}

void ReSTIR::trace(uchar4* pbo, glm::vec3* devImage, Scene* hstScene) {
    assert(Settings::enableReSTIR && "Should be in ReSTIR mode");

    const Camera& cam = hstScene->camera;
    const int pixelCount = cam.resolution.x * cam.resolution.y;

    const int BlockSizeSinglePTX = 8;
    const int BlockSizeSinglePTY = 8;
    int blockNumSinglePTX = (cam.resolution.x + BlockSizeSinglePTX - 1) / BlockSizeSinglePTX;
    int blockNumSinglePTY = (cam.resolution.y + BlockSizeSinglePTY - 1) / BlockSizeSinglePTY;

    dim3 singlePTBlockNum(blockNumSinglePTX, blockNumSinglePTY);
    dim3 singlePTBlockSize(BlockSizeSinglePTX, BlockSizeSinglePTY);


    /*kernelRIS << <singlePTBlockNum, singlePTBlockSize >> > (State::iteration,
        ReSTIRSettings::M_Light, ReSTIRSettings::M_BSDF,
        hstScene->devScene, cam, devImage);*/

    kernelSpatial << <singlePTBlockNum, singlePTBlockSize >> > (State::iteration,
        ReSTIRSettings::M_Light, ReSTIRSettings::M_BSDF,
        hstScene->devScene, cam, devImage,
        spatialRSV);

    // 2D block sending pixel data
    const dim3 blockSize2D(8, 8);
    const dim3 blocksPerGrid2D(
        (cam.resolution.x + blockSize2D.x - 1) / blockSize2D.x,
        (cam.resolution.y + blockSize2D.y - 1) / blockSize2D.y);
    // Send results to OpenGL buffer for rendering
    sendImageToPBO << <blocksPerGrid2D, blockSize2D >> > (pbo, cam.resolution, devImage, Settings::toneMapping);

    // Retrieve image from GPU
    CUDA::copyDevToHost(hstScene->state.image.data(), devImage,
        pixelCount * sizeof(glm::vec3));

    checkCUDAError("ReSTIR::trace");
}

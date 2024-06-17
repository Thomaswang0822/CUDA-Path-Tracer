#pragma once

#include "utilities.h"

#define InvalidPdf -1.f


/**
 * Not enum class becuase we need to OR and AND them.
 */
enum BSDFSampleType {
    Diffuse = 1 << 0,
    Glossy = 1 << 1,
    Specular = 1 << 2,

    Reflection = 1 << 4,
    Transmission = 1 << 5,

    Invalid = 1 << 15
};

struct BSDFSample {
    glm::vec3 dir;
    glm::vec3 bsdf;
    float pdf;
    uint32_t type;
};

/**
 * vec3 Fresnel term; need to precompute F0.
 */
__device__ inline glm::vec3 fresnel(float cosTheta, glm::vec3 f0) {
    //return f0 + (glm::vec3(1.0f) - f0) * powf(1.0 - cosTheta, 5.0);
    return glm::mix(f0, glm::vec3(1.f), powf(1.f - cosTheta, 5.f));
}

/**
 * float Fresnel term.
 */
__device__ inline float fresnel(float cosTheta, float ior) {
    float f0 = (1.f - ior) / (1.f + ior);
    return glm::mix(f0, 1.f, powf(1.f - cosTheta, 5.f));
}

struct Material {
    enum class Type {
        Lambertian, 
        MetallicWorkflow, 
        Dielectric, 
        Light
    };

    Type type;
    glm::vec3 baseColor;
    float metallic;
    float roughness;
    float ior;
    float emittance;

    int textureId;
};


__device__ static glm::vec3 lambertianBSDF(glm::vec3 n, glm::vec3 wo, glm::vec3 wi, const Material& m) {
    return m.baseColor * mathUtil::nonnegativeDot(n, wi) * INV_PI;
}
__device__ static float lambertianPdf(glm::vec3 n, glm::vec3 wo, glm::vec3 wi, const Material& m) {
    return glm::dot(n, wi) * INV_PI;
}

__device__ static void lambertianSample(glm::vec3 n, glm::vec3 wo, const Material& m, glm::vec3 r, BSDFSample& sample) {
    sample.dir = mathUtil::sampleHemisphereCosine(n, r.x, r.y);
    sample.bsdf = m.baseColor * INV_PI;
    sample.pdf = glm::dot(n, sample.dir) * INV_PI;
    sample.type = Diffuse | Reflection;
}

__device__ static glm::vec3 dielectricBSDF(glm::vec3 n, glm::vec3 wo, glm::vec3 wi, const Material& m) {
    return glm::vec3(0.f);
}

__device__ static float dielectricPdf(glm::vec3 n, glm::vec3 wo, glm::vec3 wi, const Material& m) {
    return 0.f;
}

__device__ static void dielectricSample(glm::vec3 n, glm::vec3 wo, const Material& m, glm::vec3 r, BSDFSample& sample) {
    float ior = m.ior;
    // these 2 pdfs are deterministic
    float pdfRefl = fresnel(glm::dot(n, wo), ior);
    float pdfTran = 1.f - pdfRefl;
    //pdfRefl = 1.f;
    sample.pdf = 1.f;
    sample.bsdf = m.baseColor;

    // and we "flip a coin" to decide reflect or refract
    if (r.z < pdfRefl) {
        sample.dir = glm::reflect(-wo, n);
        sample.type = Specular | Reflection;
    }
    else {
        if (!mathUtil::refract(n, wo, ior, sample.dir)) {
            sample.type = Invalid;
            return;
        }
        if (glm::dot(n, wo) < 0) {
            ior = 1.f / ior;
        }
        sample.bsdf /= ior * ior;
        sample.type = Specular | Transmission;
    }
}

__device__ static float schlickG(float cosTheta, float alpha) {
    float a = alpha * .5f;
    return cosTheta / (cosTheta * (1.f - a) + a);
}

__device__ inline float smithG(float cosWo, float cosWi, float alpha) {
    return schlickG(glm::abs(cosWo), alpha) * schlickG(glm::abs(cosWi), alpha);
}

__device__ static float ggxDistrib(float cosTheta, float alpha) {
    if (cosTheta < 1e-6f) {
        return 0.f;
    }
    float aa = alpha * alpha;
    float nom = aa;
    float denom = cosTheta * cosTheta * (aa - 1.f) + 1.f;
    denom = denom * denom * PI;
    return nom / denom;
}

__device__ static float ggxPdf(glm::vec3 n, glm::vec3 m, glm::vec3 wo, float alpha) {
    return ggxDistrib(glm::dot(n, m), alpha) * schlickG(glm::dot(n, wo), alpha) *
        mathUtil::absDot(m, wo) / mathUtil::absDot(n, wo);
}

/**
* Sample GGX microfacet distribution, but only visible normals.
* This reduces invalid samples and make pdf values at grazing angles more stable
* See [Sampling the GGX Distribution of Visible Normals, Eric Heitz, JCGT 2018]:
* https://jcgt.org/published/0007/04/01/
*/
__device__ static glm::vec3 ggxSample(glm::vec3 n, glm::vec3 wo, float alpha, glm::vec2 r) {
    glm::mat3 transMat = mathUtil::localRefMatrix(n);
    glm::mat3 transInv = glm::inverse(transMat);

    glm::vec3 vh = glm::normalize((transInv * wo) * glm::vec3(alpha, alpha, 1.f));

    float lenSq = vh.x * vh.x + vh.y * vh.y;
    glm::vec3 t = lenSq > 0.f ? glm::vec3(-vh.y, vh.x, 0.f) / sqrt(lenSq) : glm::vec3(1.f, 0.f, 0.f);
    glm::vec3 b = glm::cross(vh, t);

    glm::vec2 p = mathUtil::toUnitDisk(r.x, r.y);
    float s = 0.5f * (vh.z + 1.f);
    p.y = (1.f - s) * glm::sqrt(1.f - p.x * p.x) + s * p.y;

    glm::vec3 h = t * p.x + b * p.y + vh * glm::sqrt(glm::max(0.f, 1.f - glm::dot(p, p)));
    h = glm::normalize(glm::vec3(h.x * alpha, h.y * alpha, glm::max(0.f, h.z)));
    return transMat * h;
}

__device__ static glm::vec3 metallicWorkflowBSDF(glm::vec3 n, glm::vec3 wo, glm::vec3 wi, const Material& m) {
    float alpha = m.roughness * m.roughness;
    glm::vec3 h = glm::normalize(wo + wi);

    float cosO = glm::dot(n, wo);
    float cosI = glm::dot(n, wi);
    if (cosI * cosO < 1e-7f) {
        return glm::vec3(0.f);
    }

    glm::vec3 f = fresnel(glm::dot(h, wo), m.baseColor * m.metallic);
    float g = smithG(cosO, cosI, alpha);
    float d = ggxDistrib(glm::dot(n, h), alpha);

    return glm::mix(m.baseColor * INV_PI * (1.f - m.metallic), glm::vec3(g * d / (4.f * cosI * cosO)), f);
}

__device__ static float metallicWorkflowPdf(glm::vec3 n, glm::vec3 wo, glm::vec3 wi, const Material& m) {
    glm::vec3 h = glm::normalize(wo + wi);
    return glm::mix(
        mathUtil::nonnegativeDot(n, wi) * INV_PI,
        ggxPdf(n, h, wo, m.roughness * m.roughness) / (4.f * mathUtil::absDot(h, wo)),
        1.f / (2.f - m.metallic)
    );
}

__device__ static void metallicWorkflowSample(glm::vec3 n, glm::vec3 wo, const Material& m, glm::vec3 r, BSDFSample& sample) {
    float alpha = m.roughness * m.roughness;

    if (r.z > (1.f / (2.f - m.metallic))) {
        sample.dir = mathUtil::sampleHemisphereCosine(n, r.x, r.y);
    }
    else {
        glm::vec3 h = ggxSample(n, wo, alpha, glm::vec2(r));
        sample.dir = -glm::reflect(wo, h);
    }

    if (glm::dot(n, sample.dir) < 0.f) {
        sample.type = Invalid;
    }
    else {
        sample.bsdf = metallicWorkflowBSDF(n, wo, sample.dir, m);
        sample.pdf = metallicWorkflowPdf(n, wo, sample.dir, m);
        sample.type = Glossy | Reflection;
    }
}


__device__ static glm::vec3 materialBSDF(glm::vec3 n, glm::vec3 wo, glm::vec3 wi, const Material& m) {
    switch (m.type) {
    case Material::Type::Lambertian:
        return lambertianBSDF(n, wo, wi, m);
    case Material::Type::MetallicWorkflow:
        return metallicWorkflowBSDF(n, wo, wi, m);
    case Material::Type::Dielectric:
        return dielectricBSDF(n, wo, wi, m);
    }
    return glm::vec3(0.f);
}

__device__ static float materialPdf(glm::vec3 n, glm::vec3 wo, glm::vec3 wi, const Material& m) {
    switch (m.type) {
    case Material::Type::Lambertian:
        return lambertianPdf(n, wo, wi, m);
    case Material::Type::MetallicWorkflow:
        return metallicWorkflowPdf(n, wo, wi, m);
    case Material::Type::Dielectric:
        return dielectricPdf(n, wo, wi, m);
    }
    return 0.f;
}

__device__ static void materialSample(glm::vec3 n, glm::vec3 wo, const Material& m, glm::vec3 r, BSDFSample& sample) {
    switch (m.type) {
    case Material::Type::Lambertian:
        lambertianSample(n, wo, m, r, sample);
        break;
    case Material::Type::MetallicWorkflow:
        // NOT IMPLEMENTED
        metallicWorkflowSample(n, wo, m, r, sample);
        break;
    case Material::Type::Dielectric:
        dielectricSample(n, wo, m, r, sample);
        break;
    default:
        sample.type = Invalid;
    }
}

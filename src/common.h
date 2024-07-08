#pragma once
#include <iostream>

#define SAMPLER_USE_SOBOL false
#define SCENE_LIGHT_SINGLE_SIDED true
#define BVH_DISABLE false
#define ENABLE_GBUFFER false
#define CAMERA_PANORAMA false

#define DEBUG_RED   glm::vec3(1.f, 0.f, 0.f);
#define DEBUG_GREEN glm::vec3(0.f, 1.f, 0.f);
#define DEBUG_BLUE  glm::vec3(0.f, 0.f, 1.f);

#define ABS_SCENE_PATH "D:/Code/CUDA-Path-Tracer/scenes"

struct ToneMapping {
    enum {
        None = 0, Filmic = 1, ACES = 2
    };
};

struct Tracer {
    enum {
        PathTrace = 0, BVHVisualize = 1, GBufferPreview = 2
    };
};

struct Settings {
    static int traceDepth;
    static int toneMapping;
    static int tracer;
    static int GBufferPreviewOpt;
    static bool averageSPP;
    static bool enableReSTIR;
};

struct State {
    static bool camChanged;
    static int iteration;  // spp so far
};

struct ReSTIRSettings {
    static int M_Light;
    static int M_BSDF;
};

#pragma once
#include <iostream>

#define SAMPLER_USE_SOBOL false
#define SCENE_LIGHT_SINGLE_SIDED true
#define BVH_DISABLE false
#define ENABLE_GBUFFER false
#define CAMERA_PANORAMA false
#define AVERAGE_SPP true

struct ToneMapping {
    enum {
        None = 0, Filmic = 1, ACES = 2
    };
};

struct Tracer {
    enum {
        SingleKernel = 0, BVHVisualize = 1, GBufferPreview = 2,
        ReSTIR_DI = 3
    };
};

struct Settings {
    static int traceDepth;
    static int toneMapping;
    static int tracer;
    static bool sortMaterial;
    static int GBufferPreviewOpt;

    static bool enableReSTIR;
};

struct State {
    static bool camChanged;
};

struct ReSTIRSettings {
    static int M_Light;
    static int M_BSDF;
};

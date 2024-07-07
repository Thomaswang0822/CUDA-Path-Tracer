#include "common.h"
/// Default values of each option; they can be modified in preview.cpp

int Settings::traceDepth = 0;
int Settings::toneMapping = ToneMapping::None;
int Settings::tracer = Tracer::SingleKernel;
bool Settings::sortMaterial = false;
int Settings::GBufferPreviewOpt = 0;

bool Settings::enableReSTIR = false;

bool State::camChanged = true;

int ReSTIRSettings::M_Light = 16;
int ReSTIRSettings::M_BSDF = 4;

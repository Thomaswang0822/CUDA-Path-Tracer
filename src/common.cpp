#include "common.h"
/// Default values of each option; they can be modified in preview.cpp

int Settings::traceDepth = 0;
int Settings::toneMapping = ToneMapping::None;
int Settings::tracer = Tracer::PathTrace;
int Settings::GBufferPreviewOpt = 0;
bool Settings::averageSPP = false;
bool Settings::enableReSTIR = true;

bool State::camChanged = true;
int State::iteration = 0;

int ReSTIRSettings::M_Light = 16;
int ReSTIRSettings::M_BSDF = 4;
//int ReSTIRSettings::spatialRadius = 3;
int ReSTIRSettings::reuseOption = Reuse::Spatial;

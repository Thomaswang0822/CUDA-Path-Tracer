#include "common.h"

int Settings::traceDepth = 0;
int Settings::toneMapping = ToneMapping::None;
int Settings::tracer = Tracer::ReSTIR_DI;
bool Settings::sortMaterial = false;
int Settings::GBufferPreviewOpt = 0;

bool State::camChanged = true;

int ReSTIRSettings::M_Light = 16;
int ReSTIRSettings::M_BSDF = 4;

#include "common.h"

int Settings::traceDepth = 0;
int Settings::toneMapping = ToneMapping::None;
int Settings::tracer = Tracer::ReSTIR_DI;
bool Settings::sortMaterial = false;
int Settings::GBufferPreviewOpt = 0;
int Settings::M_ReSTIR = 128;

bool State::camChanged = true;
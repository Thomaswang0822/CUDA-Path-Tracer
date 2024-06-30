#include "settings.h"

int Settings::traceDepth = 4;
int Settings::toneMapping = ToneMapping::None;
int Settings::tracer = Tracer::Streamed;

bool State::camChanged = true;

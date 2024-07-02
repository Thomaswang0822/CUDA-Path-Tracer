#pragma once

/** Compile time options */
#define SCENE_LIGHT_SINGLE_SIDED true
#define ENABLE_GBUFFER false

struct ToneMapping {
	enum {
		None = 0, ACES = 1
	};
};

struct Tracer {
	enum {
		Streamed = 0, BVHVisualize = 1, ReSTIR_DI = 2
	};
};

/**
 * A wrapper class that enables chaning options real-time with ImGui.
 */
struct Settings {
	static int traceDepth;
	static int toneMapping;
	static int tracer;
};

struct State {
	static bool camChanged;
};

#pragma once

#include "utilities.h"
#include "sampler.h"

namespace ReSTIR {
	/**
	 * When given to a DevScene, should be able to compute p^, p, etc.
	 * This is also "the element" in the Reservoir
	 */
	struct SampledLight {
		/** Data */
		int lightId = -1;
		int primId = -1;
		glm::vec3 position = glm::vec3(0.f);
		glm::vec3 normal = glm::vec3(0.f, 1.f, 0.f);

		__device__ void operator = (const SampledLight& rhs) {
			lightId = rhs.lightId;
			primId = rhs.primId;
			position = rhs.position;
			normal = rhs.normal;
		}
	};

	/**
	 * Reservoir that consists of N=1 sample.
	 */
	struct Reservoir {
		SampledLight y;  // output sample, a position
		float w_sum = 0.f;  // sum of weights
		uint32_t M = 0;  // number of samples seen so far
		float W = 0.f;  // see Eq. 6 in DI paper

		__device__ void update(SampledLight x_i, float w_i, float rand) {
			w_sum += w_i;
			M++;
			if (rand * w_sum < w_i) {
				y = x_i;
			}
		}
	};
}

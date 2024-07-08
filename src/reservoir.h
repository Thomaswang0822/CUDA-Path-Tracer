#pragma once

#include "mathUtil.h"
#include <cuda_runtime.h>

#define INVALID_INDEX -1
#define PIXEL_RADIUS 3.f

// ReSTIR utility functions
namespace ReSTIR {
	/**
	 * Calculate the weight = p^/p,
	 * 0 if p is a bad value (failed sample), checked outside.
	 */
	__device__ inline static float scalarWeight(glm::vec3 p_hat, float p) {
		return Math::luminance(p_hat / p);
	}

	/**
	 * Turn the high-dim distribution into a float distribution.
	 */
	__device__ inline static float toScalar(glm::vec3 p) {
		return Math::luminance(p);
	}

	/**
	 * MIS weight of using sample strategy A.
	 *
	 * @param pA: Distribution of A, which is the sampling used
	 * @param pB: Distribution of B, the "counterpart"
	 * @param MA, MB: Numbers of samples for each.
	 */
	__device__ inline static float MIS_BalanceWeight(float pA, float pB, int MA, int MB) {
		return pA / (MA * pA + MB * pB);
	}


	__device__ static int sampleNeighbor(int x, int y, glm::ivec2 resolution, glm::vec2 rand2) {
		glm::vec2 offset = Math::toConcentricDisk(rand2.x, rand2.y) * PIXEL_RADIUS;
		int offx = x + offset.x;
		int offy = y + offset.y;
		int idx = offy * resolution.x + offx;

		if (offx < 0 || offx >= resolution.x || offy < 0 || offy > resolution.y) {
			return INVALID_INDEX;
		}
		return idx;
	}
}

/**
* When given to a DevScene, should be able to compute p^, p, etc.
* This is also "the element" in the Reservoir
*/
struct SampledLight {
	/** Data */
	glm::vec3 dir = glm::vec3(0.f, 1.f, 0.f);  // shading-point dependent
	glm::vec3 position = glm::vec3(0.f);
	glm::vec3 targetFunc = glm::vec3(0.f);  // p_hat = Le * bsdf * cosTerm

	__device__ void operator = (const SampledLight& rhs) {
		dir = rhs.dir;
		position = rhs.position;
		targetFunc = rhs.targetFunc;
	}
};

/**
	* Reservoir that consists of N=1 sample.
	*/
struct Reservoir {
	SampledLight y;  // output sample
	float w_sum = 0.f;  // sum of weights
	uint32_t M = 0;  // number of samples seen so far
	float W = 0.f;  // Unbiased Contribution Weight (UCW) see Eq. 6 in DI paper

	/**
	 * ReSTIR DI paper Algorithm 2.
	 *
	 * @param w_i: Proposal weight
	 */
	__device__ void update(SampledLight x_i, float w_i, float rand) {
		w_sum += w_i;
		M++;
		if (rand * w_sum < w_i) {
			y = x_i;
		}
	}

	__device__ void combine(const Reservoir& r, float rand) {
		// This is biased weight: need to eval self.p_hat on r;
		float weight = ReSTIR::toScalar(r.y.targetFunc) * r.W;
		w_sum += weight;
		M += r.M;
		if (rand * w_sum < weight) {
			y = r.y;
		}
	}
};

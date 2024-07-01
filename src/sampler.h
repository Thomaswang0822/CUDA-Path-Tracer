#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <thrust/random.h>
#include <vector>
#include <numeric> // For std::accumulate
#include <algorithm> // For std::transform
#include <stack>
#include "cudaUtil.h"


/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

__host__ __device__ static
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

struct Sampler {
    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<float> u01;

    // constructor
    __host__ __device__
    Sampler(int iter = 0, int threadIdx = 0, int recurDepth = 0) {
        rng = makeSeededRandomEngine(iter, threadIdx, recurDepth);
        u01 = thrust::uniform_real_distribution<float>(0, 1);
    }

    __host__ __device__ inline float sample1D() {
        return u01(rng);
    }

    __host__ __device__ inline glm::vec2 sample2D() {
        return glm::vec2(u01(rng), u01(rng));
    }

    __host__ __device__ inline glm::vec3 sample3D() {
        return glm::vec3(u01(rng), u01(rng), u01(rng));
    }

    __host__ __device__ inline glm::vec4 sample4D() {
        return glm::vec4(u01(rng), u01(rng), u01(rng), u01(rng));
    }
};

#pragma region LightSampler
struct Slot {
    float prob;
    int aliasId;
};

/**
 * Vose's Alias Method to sample from a discrete distribution in O(1).
 * 
 * @see https://keithschwarz.com/darts-dice-coins/
 * @ref Vose, Michael D. "A linear algorithm for generating random numbers with a given distribution." IEEE Transactions on software engineering 17.9 (1991): 972-975.
 */
struct LightSampler {
    /** Data */
    std::vector<float> probTable;
    std::vector<int> aliasTable;
    size_t N;

    LightSampler() = default;
    /**
     * Actual constructor.
     * 
     * @param values: It stores luminance*area of all lighting TRIANGLES in the scene.
     */
    LightSampler(std::vector<float> values) {
        N = values.size();
        probTable.resize(N);
        aliasTable.resize(N);

        /// normalization to 1 and multiplication by N together: N / sum(values)
        /// learned 2 new std functions ^()^
        float norm_factor = static_cast<float>(values.size()) / 
            std::accumulate(values.begin(), values.end(), 0.0f);
        // values = [k * norm_factor for k in values]; contains scaled probabilities now
        std::transform(values.begin(), values.end(), values.begin(),
            [norm_factor](float k) { return k * norm_factor; });

        /// Create two worklists (stacks), Small and Large
        /// Use vector as stack to save memory: https://stackoverflow.com/a/71677261/14697376
        /// Init size to 2N because we don't pop from it
        std::stack<int> small, large;
        for (int i = 0; i < N; i++) {
            (values[i] < 1.f ? small : large).push(i);
        }
        
        int l, g;  // top element of small and large
        /// main loop: keep cutting from great (scaled prob >= 1)
        /// such that (cut portion + less == 1)
        while (!small.empty() && !large.empty()) {
            l = small.top();    small.pop();
            g = large.top();    large.pop();
            // only update data for l
            probTable[l] = values[l];
            aliasTable[l] = g;
            // "pollute" scaled prob of g and add it back
            values[g] = values[g] + values[l] - 1.f;
            (values[g] < 1.f ? small : large).push(g);
        }

        /// some polluted scaled probs which should be exactly == 1
        /// second loop merely for numerical issue.
        while (!large.empty()) {
            g = large.top();    large.pop();
            probTable[g] = 1.f;
            aliasTable[g] = g;
        }
        while (!small.empty()) {
            l = small.top();    small.pop();
            probTable[l] = 1.f;
            aliasTable[l] = l;
        }
    }

    inline int sample(float r1, float r2) {
        // pick a bucket
        int bucketId = static_cast<int>(r1 * N);
        return (r2 < probTable[bucketId]) ? bucketId : aliasTable[bucketId];
    }
};

/**
 * LightSampler used on device. Its data (alias table) is copied from a LightSampler
 * 
 * @code Can I just use unified memory to save the trouble?
 */
struct DevLightSampler {
    /** Same data */
    float* probTable = nullptr;
    int* aliasTable = nullptr;
    size_t N;

    /** Different constructor */
    DevLightSampler() = default;
    DevLightSampler(const LightSampler& hostSampler) {
        N = hostSampler.N;
        // int and float are both 4 bytes
        size_t byteSize = sizeof(float) * N;

        // malloc and copy
        cudaMalloc(&probTable, byteSize);
        Cuda::memcpyHostToDev(probTable, hostSampler.probTable.data(), byteSize);
        cudaMalloc(&aliasTable, byteSize);
        Cuda::memcpyHostToDev(aliasTable, hostSampler.aliasTable.data(), byteSize);
    }

    ~DevLightSampler() {
        Cuda::safeFree(probTable);
        Cuda::safeFree(aliasTable);
    }

    __device__ int sample(float r1, float r2) {
        // pick a bucket
        int bucketId = static_cast<int>(r1 * N);
        return (r2 < probTable[bucketId]) ? bucketId : aliasTable[bucketId];
    }

    
};
#pragma endregion

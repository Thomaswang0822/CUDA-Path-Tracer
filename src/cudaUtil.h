#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <stdio.h>

/**
 * Cuda utilties namespace.
 * 
 * Main purpose is to simplify calling cudaMemcpy.
 * 
 * checkCUDAError(msg) comes very handy.
 */
namespace Cuda {

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) Cuda::checkCUDAErrorFn(msg, FILENAME, __LINE__)

    static void checkCUDAErrorFn(const char* msg, const char* file, int line) {
#if ERRORCHECK
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (cudaSuccess == err) {
            return;
        }

        fprintf(stderr, "CUDA error");
        if (file) {
            fprintf(stderr, " (%s:%d)", file, line);
        }
        fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
        getchar();
#  endif
        exit(EXIT_FAILURE);
#endif
    }

    template<typename T>
    inline void safeFree(T*& ptr) {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }

    inline cudaError_t __stdcall memcpyHostToDev(void* dev, const void* host, size_t size) {
        return cudaMemcpy(dev, host, size, cudaMemcpyKind::cudaMemcpyHostToDevice);
    }

    inline cudaError_t __stdcall memcpyDevToDev(void* dst, const void* src, size_t size) {
        return cudaMemcpy(dst, src, size, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
    }

    inline cudaError_t __stdcall memcpyDevToHost(void* host, const void* dev, size_t size) {
        return cudaMemcpy(host, dev, size, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    }
}

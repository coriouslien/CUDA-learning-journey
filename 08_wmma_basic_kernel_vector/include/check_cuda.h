// check_cuda.h

#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(ans)                        \
    {                                          \
        cudaAssert((ans), __FILE__, __LINE__); \
    }

inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "BPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
        {
            exit(code);
        }
    }
}

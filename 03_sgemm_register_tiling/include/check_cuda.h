#pragma once

#include <cuda_runtime.h>
#include <cstdio>

#define CHECK_CUDA(call)                                   \
    do                                                     \
    {                                                      \
        cudaError_t err = call;                            \
        if (err != cudaSuccess)                            \
        {                                                  \
            printf("CUDA Error: %s - at %s at line %d:\n", \
                   cudaGetErrorString(err),                \
                   __FILE__,                               \
                   __LINE__);                              \
            exit(EXIT_FAILURE);                            \
        }                                                  \
    } while (0);

// check_cuda.h
#pragma once

#include <cstdio>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                   \
    {                                                                      \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess)                                            \
        {                                                                  \
            printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), \
                   __LINE__);                                              \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    }

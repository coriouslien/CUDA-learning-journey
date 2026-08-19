#pragma once

#include <cstdio>
#include <cuda_runtime.h>


static void cudaHandleError(cudaError_t err,
                            const char *file,
                            int line)
{
    if (err != cudaSuccess)
    {
        std::printf("%s in %s at line %d\n", cudaGetErrorString(err),
                    file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(err) (cudaHandleError(err, __FILE__, __LINE__))



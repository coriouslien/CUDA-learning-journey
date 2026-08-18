// check_cublas.h
#pragma once

#include <cublas_v2.h>
#include <cstdio>

#define CHECK_CUBLAS(ans)                        \
    {                                            \
        cublasAssert((ans), __FILE__, __LINE__); \
    }
inline void cublasAssert(cublasStatus_t code, const char *file, int line, bool abort = true)
{
    if (code != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "cuBLAS assert: code &d in %d %s %d\n", code, file, line);
        if (abort)
        {
            exit(code);
        }
    }
}


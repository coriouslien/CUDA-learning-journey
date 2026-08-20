// include/parallel_reduction.h
#include "check_cuda.h"
#include "nvtx3_utils.h"
#include "warmup.h"

#include <omp.h>
#include <stdio.h>
#include <cuComplex.h>

#pragma once

void verify_result(cuFloatComplex *deviceOut, int numElements, float ms);
void initDeviceData(cuFloatComplex *deviceIn, int numElements);
float atomicAddKernelNaive(int numElements, int iterations);
float atomicAddKernelSharedMem(int numElements, int iterations);
float partialStoreSharedMem(int numElements, int iterations);
float atomicAddPartialStoreSharedMem(int numElements, int iterations);



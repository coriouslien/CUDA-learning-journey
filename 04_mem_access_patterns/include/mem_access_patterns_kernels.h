#include "handle_errors.h"
#include "nvtx3_utils.h"
#include "warmup.h"

#include <omp.h>
#include <stdio.h>
#include <cuComplex.h>

#pragma once

__global__ void bandwidth_coalesced(float *__restrict__ d_out,
                                    const float *__restrict__ d_in,
                                    int n);
__global__ void bandwidth_strided(float *__restrict__ d_out,
                                  const float *__restrict__ d_in,
                                  int n, int stride);

__global__ void bandwidth_random(float *__restrict__ d_out,
                                 const float *__restrict__ d_in,
                                 int *indices, int n);

__global__ void warmed_up_kernel(float *d_out);



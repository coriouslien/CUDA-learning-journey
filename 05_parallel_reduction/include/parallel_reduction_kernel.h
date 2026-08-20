// include/parallel_reduction_kernel.h
#include "check_cuda.h"
#include "nvtx3_utils.h"
#include "warmup.h"

#include <omp.h>
#include <stdio.h>
#include <cuComplex.h>

#pragma once

__global__ void atomic_add_kernel_naive(const cuFloatComplex *__restrict__ d_in,
                                        cuFloatComplex *__restrict__ d_out,
                                        int num_elements);
__global__ void atomic_add_kernel_shared_mem(const cuFloatComplex *__restrict__ d_in,
                                             cuFloatComplex *__restrict__ d_out,
                                             int num_elements);
__global__ void partial_store_shared(const cuFloatComplex *__restrict__ d_in,
                                     cuFloatComplex *__restrict__ d_out,
                                     int num_elements);
__global__ void final_reduce_kernel(const cuFloatComplex *__restrict__ d_partial_sums,
                                    cuFloatComplex *__restrict__ d_grand_total,
                                    int num_partials);
__global__ void atomic_add_partial_store(const cuFloatComplex *__restrict__ d_partial,
                                         cuFloatComplex *__restrict__ d_out,
                                         int total_elements);



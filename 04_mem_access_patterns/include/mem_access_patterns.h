#include "handle_errors.h"
#include "nvtx3_utils.h"

#include <omp.h>
#include <stdio.h>
#include <cuComplex.h>

#pragma once


bool verification(const float *output, const float *expected_in,
                  int N, const char *kernel_name,
                  int *indices, // for random
                  int stride);         // for strided

void run_benchmark(int N, int *strides, int num_strides, bool use_pinned);


// wmma_basic.h
#pragma once
#include <cuda_fp16.h>

float launchWMMABasic(const half *a, const half *b, float *c,
                      int m, int n, int k,
                      int block_size, int tile_m, int tile_n, int iterations);

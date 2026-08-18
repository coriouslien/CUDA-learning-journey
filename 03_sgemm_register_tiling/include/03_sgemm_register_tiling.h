#pragma once
#include "sgemm_config.h"

// kernel and helper declarations
__device__ void load_s_A(const float *__restrict__ A,
                         float s_A[SgemmCfg::BM][SgemmCfg::BK],
                         int tid, int step, int M, int K);

__device__ void load_s_B(const float *__restrict__ B,
                         float s_B[SgemmCfg::BK][SgemmCfg::BN],
                         int tid, int step, int K, int N);

__device__ void compute_slice(const float s_A[SgemmCfg::BM][SgemmCfg::BK],
                              const float s_B[SgemmCfg::BK][SgemmCfg::BN],
                              float c_regs[SgemmCfg::TM][SgemmCfg::TN],
                              float a_regs[SgemmCfg::TM],
                              float b_regs[SgemmCfg::TN],
                              int tx, int ty);

__device__ void store_c(float *__restrict__ C,
                        const float c_regs[SgemmCfg::TM][SgemmCfg::TN],
                        int thread_row, int thread_col,
                        int M, int N);

__global__ void sgemm_kernel(const float *__restrict__ A,
                             const float *__restrict__ B,
                             float *__restrict__ C,
                             int M, int N, int K);
__global__ void sgemm_register_tiling_kernel(const float *__restrict__ A,
                                             const float *__restrict__ B,
                                             float *__restrict__ C,
                                             int M, int N, int K);

// 03_sgemm_register_tiling_float4.cu

#include "sgemm_config.h"
#include <cstdio>
#include <omp.h>

using Cfg = SgemmCfg;

// 128 x 8
__device__ void load_s_A(const float *__restrict__ A,
                         float s_A[Cfg::BM][Cfg::BK],
                         int tid, int step,
                         int M, int K)
{
    constexpr int ELEMS = Cfg::BM * Cfg::BK;        // 1024
    constexpr int ITERS = ELEMS / Cfg::NUM_THREADS; // 4
#pragma unroll
    for (int i = 0; i < ITERS; i++)
    {
        int flat   = tid + i * Cfg::NUM_THREADS;
        int s_row  = flat / Cfg::BK;
        int s_col  = flat % Cfg::BK;
        int g_row  = blockIdx.y * Cfg::BM + s_row;
        int g_col  = step * Cfg::BK + s_col;
        s_A[s_row][s_col] = (g_row < M && g_col < K)
                            ? A[g_row * K + g_col]
                            : 0.0f;
    }
}

// ── No changes to load_s_B ──────────────────────────────────────────────────
__device__ void load_s_B(const float *__restrict__ B,
                         float s_B[Cfg::BK][Cfg::BN],
                         int tid, int step,
                         int K, int N)
{
    constexpr int ELEMS = Cfg::BK * Cfg::BN;
    constexpr int ITERS = ELEMS / Cfg::NUM_THREADS;

#pragma unroll
    for (int i = 0; i < ITERS; i++)
    {
        int flat  = tid + i * Cfg::NUM_THREADS;
        int s_row = flat / Cfg::BN;
        int s_col = flat % Cfg::BN;
        int g_row = step * Cfg::BK + s_row;
        int g_col = blockIdx.x * Cfg::BN + s_col;
        s_B[s_row][s_col] = (g_row < K && g_col < N)
                            ? B[g_row * N + g_col]
                            : 0.0f;
    }
}

// ── float4 applied here ─────────────────────────────────────────────────────
__device__ void compute_slice(const float s_A[Cfg::BM][Cfg::BK],
                              const float s_B[Cfg::BK][Cfg::BN],
                              float c_regs[Cfg::TM][Cfg::TN],
                              float a_regs[Cfg::TM],
                              float b_regs[Cfg::TN],
                              int tx, int ty)
{
#pragma unroll
    for (int k = 0; k < Cfg::BK; k++)
    {
        // ── Load a_regs from s_A (unchanged) ────────────────────────────────
#pragma unroll
        for (int m = 0; m < Cfg::TM; m++)
            a_regs[m] = s_A[ty * Cfg::TM + m][k];

        // ── Load b_regs from s_B using float4 ───────────────────────────────
        // TN = 8, so we load two float4s (4 floats each) per thread.
        // Each float4 reads 4 consecutive floats from the same row of s_B.
        // Consecutive floats map to consecutive banks → zero bank conflict.
        //
        // tx * TN     = starting column for this thread
        // tx * TN + 4 = second group of 4 columns
        //
        // Requirement: tx * TN must be 16-byte aligned.
        // With TN=8 and float=4 bytes: tx*8*4 = tx*32 bytes → always 16-byte aligned ✓

        const float4 b0 = *reinterpret_cast<const float4 *>(
                              &s_B[k][tx * Cfg::TN]);
        const float4 b1 = *reinterpret_cast<const float4 *>(
                              &s_B[k][tx * Cfg::TN + 4]);

        b_regs[0] = b0.x;
        b_regs[1] = b0.y;
        b_regs[2] = b0.z;
        b_regs[3] = b0.w;
        b_regs[4] = b1.x;
        b_regs[5] = b1.y;
        b_regs[6] = b1.z;
        b_regs[7] = b1.w;

        // ── Outer product accumulation (unchanged) ───────────────────────────
#pragma unroll
        for (int m = 0; m < Cfg::TM; m++)
#pragma unroll
            for (int n = 0; n < Cfg::TN; n++)
                c_regs[m][n] += a_regs[m] * b_regs[n];
    }
}

__device__ void store_c(float *__restrict__ C,
                        const float c_regs[Cfg::TM][Cfg::TN],
                        int thread_row, int thread_col,
                        int M, int N)
{
#pragma unroll
    for (int m = 0; m < Cfg::TM; m++)
    {
        int grow = thread_row + m;
        int gcol = thread_col;

        if (grow < M && gcol + 7 < N)
        {
            float4 c_val0 = make_float4(c_regs[m][0], c_regs[m][1],
                                        c_regs[m][2], c_regs[m][3]);
            float4 c_val1 = make_float4(c_regs[m][4], c_regs[m][5],
                                        c_regs[m][6], c_regs[m][7]);

            float4 *c_vec = reinterpret_cast<float4 *>(&C[grow * N + gcol]);
            c_vec[0] = c_val0;
            c_vec[1] = c_val1;
        }
        else
        {
#pragma unroll
            for (int n = 0; n < Cfg::TN; n++)
            {
                if (grow < M && gcol + n < N)
                    C[grow * N + gcol + n] = c_regs[m][n];
            }
        }
    }
}

__global__ void sgemm_register_tiling_kernel(const float *__restrict__ A,
                                             const float *__restrict__ B,
                                             float *__restrict__ C,
                                             int M, int N, int K)
{
    __shared__ float s_A[Cfg::BM][Cfg::BK];
    __shared__ float s_B[Cfg::BK][Cfg::BN];   // original layout, no padding

    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    float c_regs[Cfg::TM][Cfg::TN] = {0.0f};
    float a_regs[Cfg::TM];
    float b_regs[Cfg::TN];

    const int thread_row = blockIdx.y * Cfg::BM + ty * Cfg::TM;
    const int thread_col = blockIdx.x * Cfg::BN + tx * Cfg::TN;

    const int num_steps = (K + Cfg::BK - 1) / Cfg::BK;
    for (int step = 0; step < num_steps; step++)
    {
        load_s_A(A, s_A, tid, step, M, K);
        load_s_B(B, s_B, tid, step, K, N);
        __syncthreads();

        compute_slice(s_A, s_B, c_regs, a_regs, b_regs, tx, ty);
        __syncthreads();
    }

    store_c(C, c_regs, thread_row, thread_col, M, N);
}


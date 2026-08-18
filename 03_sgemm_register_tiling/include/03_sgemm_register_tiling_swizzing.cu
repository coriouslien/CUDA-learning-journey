// 03_sgemm_register_tiling_swizzing.cu

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
        int flat = tid + i * Cfg::NUM_THREADS;
        int s_row = flat / Cfg::BK;
        int s_col = flat % Cfg::BK;
        int g_row = blockIdx.y * Cfg::BM + s_row;
        int g_col = step * Cfg::BK + s_col;
        s_A[s_row][s_col] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : 0.0f;
    }
}

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
        int flat = tid + i * Cfg::NUM_THREADS;
        int s_row = flat / Cfg::BN;
        int s_col = flat % Cfg::BN;
        int g_row = step * Cfg::BK + s_row;
        int g_col = blockIdx.x * Cfg::BN + s_col;

        // THE SWIZZLE: Scramble the column address before saving to Shared Memory
        int swizzled_col = s_col ^ ((s_col >> 5) << 2);

        s_B[s_row][swizzled_col] = (g_row < K && g_col < N) ? B[g_row * N + g_col] : 0.0f;
    }
}

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
#pragma unroll
        for (int m = 0; m < Cfg::TM; m++)
            a_regs[m] = s_A[ty * Cfg::TM + m][k];

#pragma unroll
        for (int n = 0; n < Cfg::TN; n++)
        {
            int col = tx * Cfg::TN + n;

            // THE UNSWIZZLE: Apply the exact same logic to find the scrambled data
            int swizzled_col = col ^ ((col >> 5) << 2);

            b_regs[n] = s_B[k][swizzled_col];
        }

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

        // Fast Path: Fully inside matrix boundaries
        if (grow < M && gcol + 7 < N)
        {
            // Build the 128-bit chunks using registers
            float4 c_val0 = make_float4(c_regs[m][0], c_regs[m][1], c_regs[m][2], c_regs[m][3]);
            float4 c_val1 = make_float4(c_regs[m][4], c_regs[m][5], c_regs[m][6], c_regs[m][7]);

            // Write massive coalesced chunks safely to Global Memory
            float4 *c_vec = reinterpret_cast<float4 *>(&C[grow * N + gcol]);
            c_vec[0] = c_val0;
            c_vec[1] = c_val1;
        }
        else
        {
            // Slow Path: Edge-case fallback bounds checking
#pragma unroll
            for (int n = 0; n < Cfg::TN; n++)
            {
                if (grow < M && gcol + n < N)
                {
                    C[grow * N + gcol + n] = c_regs[m][n];
                }
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
    __shared__ float s_B[Cfg::BK][Cfg::BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
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

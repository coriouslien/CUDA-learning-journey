// 03_sgemm_register_tiling.cu

#include "sgemm_config.h"
#include <cstdio>
#include <omp.h>

using Cfg = SgemmCfg;

__global__ void sgemm_register_tiling_kernel(const float *__restrict__ A,
                                             const float *__restrict__ B,
                                             float *__restrict__ C,
                                             int M, int N, int K)
{
    // Standard Shared Memory Allocation
    __shared__ float s_A[Cfg::BM][Cfg::BK];
    __shared__ float s_B[Cfg::BK][Cfg::BN];

    // Pillar 1: 1D Thread Mapping
    const int tid = threadIdx.x; // 0 to 255

    // Calculate 2D logical positions for the math
    const int c_row = (tid / 16) * Cfg::TM;
    const int c_col = (tid % 16) * Cfg::TN;

    const int global_row = blockIdx.y * Cfg::BM + c_row;
    const int global_col = blockIdx.x * Cfg::BN + c_col;

    float c_regs[Cfg::TM][Cfg::TN] = {0.0f};
    float a_regs[Cfg::TM];
    float b_regs[Cfg::TN];

    // Calculate Global Load Coordinates
    const int load_a_row = tid / 2;
    const int load_a_col = (tid % 2) * 4;

    const int load_b_row = tid / 32;
    const int load_b_col = (tid % 32) * 4;

    const int num_steps = (K + Cfg::BK - 1) / Cfg::BK;
    for (int step = 0; step < num_steps; step++)
    {
        // ==========================================
        // Pillar 2: Vectorized Global Loads
        // ==========================================
        int g_a_row = blockIdx.y * Cfg::BM + load_a_row;
        int g_a_col = step * Cfg::BK + load_a_col;

        float4 a_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (g_a_row < M && g_a_col < K)
        {
            a_vec = reinterpret_cast<const float4 *>(&A[g_a_row * K + g_a_col])[0];
        }
        reinterpret_cast<float4 *>(&s_A[load_a_row][load_a_col])[0] = a_vec;

        int g_b_row = step * Cfg::BK + load_b_row;
        int g_b_col = blockIdx.x * Cfg::BN + load_b_col;

        float4 b_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (g_b_row < K && g_b_col < N)
        {
            b_vec = reinterpret_cast<const float4 *>(&B[g_b_row * N + g_b_col])[0];
        }

        // ==========================================
        // Pillar 3: Chunk XOR Swizzling for Matrix B
        // ==========================================
        int chunk_in = load_b_col / 4;
        int swizzled_chunk_in = chunk_in ^ load_b_row;
        int swizzled_col_in = swizzled_chunk_in * 4;

        reinterpret_cast<float4 *>(&s_B[load_b_row][swizzled_col_in])[0] = b_vec;

        __syncthreads();

// ==========================================
// The Compute Slice
// ==========================================
#pragma unroll
        for (int k = 0; k < Cfg::BK; k++)
        {
// Hardware Broadcast Load for A (Scalar)
#pragma unroll
            for (int m = 0; m < Cfg::TM; m++)
            {
                a_regs[m] = s_A[c_row + m][k];
            }

// Vectorized & Unswizzled Load for B (float4)
#pragma unroll
            for (int n = 0; n < Cfg::TN; n += 4)
            {
                int col_out = c_col + n;
                int chunk_out = col_out / 4;
                int swizzled_chunk_out = chunk_out ^ k;
                int swizzled_col_out = swizzled_chunk_out * 4;

                float4 b_val = reinterpret_cast<const float4 *>(&s_B[k][swizzled_col_out])[0];
                b_regs[n + 0] = b_val.x;
                b_regs[n + 1] = b_val.y;
                b_regs[n + 2] = b_val.z;
                b_regs[n + 3] = b_val.w;
            }

// Outer Product Accumulation
#pragma unroll
            for (int m = 0; m < Cfg::TM; m++)
            {
#pragma unroll
                for (int n = 0; n < Cfg::TN; n++)
                {
                    c_regs[m][n] += a_regs[m] * b_regs[n];
                }
            }
        }
        __syncthreads();
    }

// ==========================================
// Vectorized 1D Write-Back
// ==========================================
#pragma unroll
    for (int m = 0; m < Cfg::TM; m++)
    {
        int grow = global_row + m;
        int gcol = global_col;

        if (grow < M && gcol + 7 < N)
        {
            float4 c_val0 = make_float4(c_regs[m][0], c_regs[m][1], c_regs[m][2], c_regs[m][3]);
            float4 c_val1 = make_float4(c_regs[m][4], c_regs[m][5], c_regs[m][6], c_regs[m][7]);

            reinterpret_cast<float4 *>(&C[grow * N + gcol])[0] = c_val0;
            reinterpret_cast<float4 *>(&C[grow * N + gcol + 4])[0] = c_val1;
        }
        else
        {
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

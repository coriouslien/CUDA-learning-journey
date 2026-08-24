// wmma_swizzle_async_kernel.cu

#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>
#include <cstdio>

#ifndef __INTELLISENSE__
using namespace nvcuda;
#else
namespace nvcuda
{
    namespace wmma
    {
    }
}
namespace wmma = nvcuda::wmma;
#endif

// Tensor Core 16x16x16 shape
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Define the total size of the tile computed by the entire block
constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 64;

const int SMEM_STRIDE = 64;

// --- MAIN KERNEL ---
__global__ void wmma_swizzle_async_kernel(const half *__restrict__ a,
                                          const half *__restrict__ b,
                                          float *__restrict__ c,
                                          int m, int n, int k)
{
    __shared__ half a_smem[2][BLOCK_M * BLOCK_K];
    __shared__ half b_smem[2][BLOCK_K * BLOCK_N];

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    //int elem_thread = 8;

    int block_row = blockIdx.y * BLOCK_M;
    int block_col = blockIdx.x * BLOCK_N;
    //int load_iterations = (BLOCK_M * BLOCK_K) / (blockDim.x * elem_thread);

    int num_tiles = k / BLOCK_K;

    int load_stage = 0;
    int write_stage = 1;

    int loads_A = (BLOCK_M * BLOCK_K) / 8; // 8 halfs per 16-byte vectorized load
    int loads_B = (BLOCK_K * BLOCK_N) / 8;

// ---------------------------------------------------------------
// PROLOGUE: load tile 0 into smem[0], wait for it
// ---------------------------------------------------------------
#pragma unroll 1
    for (int i = tid; i < loads_A; i += blockDim.x)
    {
        int smem_row = i / (BLOCK_K / 8);
        int smem_col = (i % (BLOCK_K / 8)) * 8;
        int global_a_idx = (block_row + smem_row) * k + smem_col;

        // XOR Swizzle for A
        int swizzled_col = smem_col ^ ((smem_row % 8) * 8);
        int swizzled_idx = (smem_row * SMEM_STRIDE) + swizzled_col;
        // vectorized memory loads, the asynchronous copy instruction grabs 16 bytes at a time
        __pipeline_memcpy_async(&a_smem[0][swizzled_idx], &a[global_a_idx], 16);
    }

#pragma unroll 1
    for (int i = tid; i < loads_B; i += blockDim.x)
    {
        int smem_row = i / (BLOCK_N / 8);
        int smem_col = (i % (BLOCK_N / 8)) * 8;
        int global_b_idx = smem_row * n + (block_col + smem_col);

        // XOR Swizzle for B
        int swizzled_col = smem_col ^ ((smem_row % 8) * 8);
        int swizzled_idx = (smem_row * SMEM_STRIDE) + swizzled_col;

        __pipeline_memcpy_async(&b_smem[0][swizzled_idx], &b[global_b_idx], 16);
    }
    __pipeline_commit();
    __pipeline_wait_prior(0); // wait for tile 0 — nothing to overlap yet
    __syncthreads();

    // --- 3. INITIAL PREFETCH (Load Tile 1 into Buffer 1) ---
    if (num_tiles > 1)
    {
        int next_k_offset = BLOCK_K; // Hardcoded to Tile 1

        for (int i = tid; i < loads_A; i += blockDim.x)
        {
            int smem_row = i / (BLOCK_K / 8);
            int smem_col = (i % (BLOCK_K / 8)) * 8;
            int global_a_idx = (block_row + smem_row) * k + next_k_offset + smem_col;

            int swizzled_col = smem_col ^ ((smem_row % 8) * 8);
            int swizzled_idx = (smem_row * SMEM_STRIDE) + swizzled_col;

            __pipeline_memcpy_async(&a_smem[1][swizzled_idx], &a[global_a_idx], 16);
        }

        for (int i = tid; i < loads_B; i += blockDim.x)
        {
            int smem_row = i / (BLOCK_N / 8);
            int smem_col = (i % (BLOCK_N / 8)) * 8;
            int global_b_idx = (next_k_offset + smem_row) * n + (block_col + smem_col);

            int swizzled_col = smem_col ^ ((smem_row % 8) * 8);
            int swizzled_idx = (smem_row * SMEM_STRIDE) + swizzled_col;

            __pipeline_memcpy_async(&b_smem[1][swizzled_idx], &b[global_b_idx], 16);
        }
        __pipeline_commit();
    }

    // accumulator setup
    int warp_row = warp_id / 2;
    int warp_col = warp_id % 2;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[2];
    wmma::fill_fragment(acc[0], 0.0f);
    wmma::fill_fragment(acc[1], 0.0f);

    load_stage = 0;
    write_stage = 1;

    // ---------------------------------------------------------------
    // MAIN LOOP: tile 0 .. num_tiles-2
    // Each iteration: compute tile_k, wait for tile_k+1, prefetch tile_k+2
    // ---------------------------------------------------------------
    for (int tile_k = 0; tile_k < num_tiles - 1; tile_k++)
    {
        // ── COMPUTE TILE `tile_k` (From `load_stage` buffer) ──
        for (int k_step = 0; k_step < BLOCK_K / WMMA_K; k_step++)
        {
            wmma::fragment<wmma::matrix_a,
                           WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
                frag_a;
            wmma::fragment<wmma::matrix_b,
                           WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
                frag_b[2];

            int a_row_offset = warp_row * WMMA_M * SMEM_STRIDE;
            int a_col_offset = k_step * WMMA_K;

            // A reads: change SMEM_STRIDE to SMEM_A_STRIDE
            wmma::load_matrix_sync(frag_a,
                                   &a_smem[load_stage][a_row_offset + a_col_offset],
                                   SMEM_STRIDE); // was SMEM_STRIDE

            for (int w_col = 0; w_col < 2; w_col++)
            {
                int b_row_offset = k_step * WMMA_K * SMEM_STRIDE;
                int b_col_offset = (warp_col * 2 + w_col) * WMMA_N;

                // B reads: change SMEM_STRIDE to SMEM_B_STRIDE
                wmma::load_matrix_sync(frag_b[w_col],
                                       &b_smem[load_stage][b_row_offset + b_col_offset],
                                       SMEM_STRIDE); // was SMEM_STRIDE
                wmma::mma_sync(acc[w_col], frag_a, frag_b[w_col], acc[w_col]);
            }
        }
        // all WMMA for tile tile_k is done

        // ── WAIT: tile (tile_k+1) must be ready before we read it ──
        __pipeline_wait_prior(0);
        __syncthreads();

        // ── PREFETCH TILE `tile_k + 2` (Into `write_stage` buffer) ──
        if (tile_k + 2 < num_tiles)
        {
            int next_k_offset = (tile_k + 2) * BLOCK_K;

            for (int i = tid; i < loads_A; i += blockDim.x)
            {
                int smem_row = i / (BLOCK_K / 8);
                int smem_col = (i % (BLOCK_K / 8)) * 8;
                int global_a_idx = (block_row + smem_row) * k + next_k_offset + smem_col;

                int swizzled_col = smem_col ^ ((smem_row % 8) * 8);
                int swizzled_idx = (smem_row * SMEM_STRIDE) + swizzled_col;

                __pipeline_memcpy_async(&a_smem[write_stage][swizzled_idx], &a[global_a_idx], 16);
            }

            for (int i = tid; i < loads_B; i += blockDim.x)
            {
                int smem_row = i / (BLOCK_N / 8);
                int smem_col = (i % (BLOCK_N / 8)) * 8;
                int global_b_idx = (next_k_offset + smem_row) * n + (block_col + smem_col);

                int swizzled_col = smem_col ^ ((smem_row % 8) * 8);
                int swizzled_idx = (smem_row * SMEM_STRIDE) + swizzled_col;

                __pipeline_memcpy_async(&b_smem[write_stage][swizzled_idx], &b[global_b_idx], 16);
            }
            __pipeline_commit();
        }
        // flip stages
        load_stage ^= 1;
        write_stage ^= 1;
    }

    // ---------------------------------------------------------------
    // EPILOGUE: compute the last tile (already in load_stage, already waited)
    // ---------------------------------------------------------------
    for (int k_step = 0; k_step < BLOCK_K / WMMA_K; k_step++)
    {
        wmma::fragment<wmma::matrix_a,
                       WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
            frag_a;
        wmma::fragment<wmma::matrix_b,
                       WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
            frag_b[2];

        int a_row_offset = warp_row * WMMA_M * SMEM_STRIDE;
        int a_col_offset = k_step * WMMA_K;
        wmma::load_matrix_sync(frag_a,
                               &a_smem[load_stage][a_row_offset + a_col_offset],
                               SMEM_STRIDE);

        for (int w_col = 0; w_col < 2; w_col++)
        {
            int b_row_offset = k_step * WMMA_K * SMEM_STRIDE;
            int b_col_offset = (warp_col * 2 + w_col) * WMMA_N;
            wmma::load_matrix_sync(frag_b[w_col],
                                   &b_smem[load_stage][b_row_offset + b_col_offset],
                                   SMEM_STRIDE);
            wmma::mma_sync(acc[w_col], frag_a, frag_b[w_col], acc[w_col]);
        }
    }

    // ---------------------------------------------------------------
    // STORE RESULTS
    // ---------------------------------------------------------------
    for (int w_col = 0; w_col < 2; w_col++)
    {
        int global_c_row = block_row + (warp_row * WMMA_M);
        int global_c_col = block_col + ((warp_col * 2 + w_col) * WMMA_N);
        int global_c_idx = global_c_row * n + global_c_col;

        wmma::store_matrix_sync(&c[global_c_idx], acc[w_col], n,
                                wmma::mem_row_major);
    }
}

float launchWMMABasic(const half *a, const half *b, float *c,
                      int m, int n, int k,
                      int block_size, int tile_m, int tile_n, int iterations)
{
    dim3 blockDim(block_size);
    dim3 gridDim((n + tile_n - 1) / tile_n, (m + tile_m - 1) / tile_m);

    // warm up loop
    for (int i = 0; i < 10; i++)
    {
        wmma_swizzle_async_kernel<<<gridDim, blockDim>>>(a, b, c, m, n, k);
    }
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++)
    {
        wmma_swizzle_async_kernel<<<gridDim, blockDim>>>(a, b, c, m, n, k);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / iterations;
}

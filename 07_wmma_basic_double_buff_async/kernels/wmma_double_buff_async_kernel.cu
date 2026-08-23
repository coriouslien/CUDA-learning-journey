// wmma_double_buff_async_kernel.cu

#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>

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

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

const int WARP_SIZE = 32;
const int BLOCK_ROW_WARPS = 4;
const int BLOCK_COL_WARPS = 2;

// Define the total size of the tile computed by the entire block
const int BLOCK_M = 128;
const int BLOCK_N = 128;
const int BLOCK_K = WMMA_K; // 16

// How much territory 1 warp is responsible for:
// 128 / 4 = 32 (Two 16x16 fragments tall)
const int WARP_TILE_M = BLOCK_M / BLOCK_ROW_WARPS;

// 128 / 2 = 64 (Four 16x16 fragments wide)
const int WARP_TILE_N = BLOCK_N / BLOCK_COL_WARPS;
const int PAD = 8;

__global__ void wmma_double_buff_async_kernel(const half *__restrict__ a,
                                              const half *__restrict__ b,
                                              float *__restrict__ c,
                                              int m, int n, int k)
{
    // 1. Properly sized shared memory for all 4 warps
    __shared__ half a_smem[2][BLOCK_M][BLOCK_K + PAD];
    __shared__ half b_smem[2][BLOCK_K][BLOCK_N + PAD];

    const int elem_thread = 8; // 128-bit vectorized load
    // const int elem_thread = 4;

    // Global block offsets based on the overall BLOCK_M / BLOCK_N sizes
    int block_row = blockIdx.y * BLOCK_M;
    int block_col = blockIdx.x * BLOCK_N;
    int tid = threadIdx.x;

    int warp_id = threadIdx.x / WARP_SIZE;
    int warp_row = warp_id / BLOCK_COL_WARPS;
    int warp_col = warp_id % BLOCK_COL_WARPS;

    // Dimensions of our 2D fragment array (2 rows, 4 columns)
    const int FRAGS_M = WARP_TILE_M / WMMA_M;
    const int FRAGS_N = WARP_TILE_N / WMMA_N;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[FRAGS_M][FRAGS_N];

    // Initialize all 8 fragments to 0.0f
    for (int i = 0; i < FRAGS_M; i++)
    {
        for (int j = 0; j < FRAGS_N; j++)
        {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    int write_stage = 0;
    int read_stage = 0;

    // -------------------------------------------------------
    // PROLOGUE: Load tile 0
    // -------------------------------------------------------

    {
        const __half *a_ptr = a + (block_row * k) + 0;
        const __half *b_ptr = b + (0 * n) + block_col;

        // Loop over the BLOCK tile size, not just the WMMA tile size
        for (int i = tid * elem_thread; i < BLOCK_M * BLOCK_K; i += blockDim.x * elem_thread)
        {
            int row = i / BLOCK_K;
            int col = i % BLOCK_K;
            __pipeline_memcpy_async(&a_smem[write_stage][row][col], &a_ptr[row * k + col], sizeof(half) * elem_thread);
        }
        for (int i = tid * elem_thread; i < BLOCK_K * BLOCK_N; i += blockDim.x * elem_thread)
        {
            int row = i / BLOCK_N;
            int col = i % BLOCK_N;
            __pipeline_memcpy_async(&b_smem[write_stage][row][col], &b_ptr[row * n + col], sizeof(half) * elem_thread);
        }
        __pipeline_commit();
        __pipeline_wait_prior(0); // Wait for prologue to finish before starting math
        __syncthreads();

        write_stage ^= 1; // Toggle to 1
    }

    // -------------------------------------------------------
    // MAIN LOOP
    // -------------------------------------------------------
    for (int step = 0; step < k; step += BLOCK_K)
    {
        // 1. Issue async load for the NEXT tile
        if (step + BLOCK_K < k)
        {
            const __half *a_ptr = a + (block_row * k) + (step + BLOCK_K);
            const __half *b_ptr = b + ((step + BLOCK_K) * n) + block_col;

            for (int i = tid * elem_thread; i < BLOCK_M * BLOCK_K; i += blockDim.x * elem_thread)
            {
                int row = i / BLOCK_K;
                int col = i % BLOCK_K;
                __pipeline_memcpy_async(&a_smem[write_stage][row][col], &a_ptr[row * k + col], sizeof(half) * elem_thread);
            }
            for (int i = tid * elem_thread; i < BLOCK_K * BLOCK_N; i += blockDim.x * elem_thread)
            {
                int row = i / BLOCK_N;
                int col = i % BLOCK_N;
                __pipeline_memcpy_async(&b_smem[write_stage][row][col], &b_ptr[row * n + col], sizeof(half) * elem_thread);
            }
            __pipeline_commit();
        }

        // 2. Wait ONLY for the previous tile to finish (allows next tile to fetch in background)
        __pipeline_wait_prior(1);
        __syncthreads();

        // 3. Math on the CURRENT tile
        // Create 1D arrays to hold the slices of A and B in ultra-fast registers
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag[FRAGS_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag[FRAGS_N];

        // --- LOAD PHASE ---
        // Load the 2 fragments for Matrix A (Iterating vertically)
        for (int i = 0; i < FRAGS_M; i++)
        {
            int shared_row = warp_row * WARP_TILE_M + (i * WMMA_M);
            wmma::load_matrix_sync(a_frag[i], &a_smem[read_stage][shared_row][0], BLOCK_K);
        }

        // Load the 4 fragments for Matrix B (Iterating horizontally)
        for (int j = 0; j < FRAGS_N; j++)
        {
            int shared_col = warp_col * WARP_TILE_N + (j * WMMA_N);
            wmma::load_matrix_sync(b_frag[j], &b_smem[read_stage][0][shared_col], BLOCK_N);
        }

        // --- MATH PHASE (Outer Product) ---
        // Multiply the 2 A-fragments by the 4 B-fragments to update all 8 C-fragments
        for (int i = 0; i < FRAGS_M; i++)
        {
            for (int j = 0; j < FRAGS_N; j++)
            {
                wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
            }
        }

        __syncthreads();
        write_stage ^= 1;
        read_stage ^= 1;
    }

    // -------------------------------------------------------
    // EPILOGUE: Store the accumulated results
    // -------------------------------------------------------
    for (int i = 0; i < FRAGS_M; i++)
    {
        for (int j = 0; j < FRAGS_N; j++)
        {
            // Calculate the exact global matrix coordinates for this specific sub-fragment
            int global_row = block_row + (warp_row * WARP_TILE_M) + (i * WMMA_M);
            int global_col = block_col + (warp_col * WARP_TILE_N) + (j * WMMA_N);

            float *c_ptr = c + (global_row * n) + global_col;
            wmma::store_matrix_sync(c_ptr, c_frag[i][j], n, wmma::mem_row_major);
        }
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
        wmma_double_buff_async_kernel<<<gridDim, blockDim>>>(a, b, c, m, n, k);
    }
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++)
    {
        wmma_double_buff_async_kernel<<<gridDim, blockDim>>>(a, b, c, m, n, k);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / iterations;
}

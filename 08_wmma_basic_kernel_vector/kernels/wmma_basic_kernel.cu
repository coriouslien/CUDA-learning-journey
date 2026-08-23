#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifndef __INTELLISENSE__
// When actually compiling with nvcc, use the real namespace
using namespace nvcuda;
#else
// When VS Code is parsing, give it a dummy namespace to satisfy the syntax checker
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
const int BLOCK_ROW_WARPS = 2;
const int BLOCK_COL_WARPS = 2;

// basic WMMA kernel that utilizes Tensor Cores and shared memory,
// without using cp.async or double buffering.
// Global Memory --> Shared Memory --> Registers(Fragments) --> Tensor Cores.
__global__ void wmma_basic_kernel(const half *__restrict__ a,
                                  const half *__restrict__ b,
                                  float *__restrict__ c,
                                  int m, int n, int k)
{
    __shared__ half a_s[BLOCK_ROW_WARPS * WMMA_M][WMMA_K];
    __shared__ half b_s[WMMA_K][BLOCK_COL_WARPS * WMMA_N];

    int warp_id = threadIdx.x / WARP_SIZE;
    int warp_row = warp_id / BLOCK_COL_WARPS;
    int warp_col = warp_id % BLOCK_COL_WARPS;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int global_row_base = blockIdx.y * (BLOCK_ROW_WARPS * WMMA_M);
    int global_col_base = blockIdx.x * (BLOCK_COL_WARPS * WMMA_N);

    for (int step = 0; step < k; step += WMMA_K)
    {
        // load from global memory to shared memory. hopping the step.
        int tid = threadIdx.x;
        int block_size = blockDim.x;

        // load a matrix from global to shared (BLOCK_ROW_WARPS = 2, WMMA_M = 16, WMMA_K = 16:
        // 2 x 16 x 16 = 512 elements)
        for (int i = tid; i < (BLOCK_ROW_WARPS * WMMA_M * WMMA_K); i += block_size)
        {
            int row = i / WMMA_K;
            int col = i % WMMA_K;
            if (((global_row_base + row) < m) && ((step + col) < k))
            {
                a_s[row][col] = a[(global_row_base + row) * k + (step + col)];
            }
            else
            {
                a_s[row][col] = __float2half(0.0f);
            }
        }
        for (int i = tid; i < (WMMA_K * BLOCK_COL_WARPS * WMMA_N); i += block_size)
        {
            int row = i / (BLOCK_COL_WARPS * WMMA_N);
            int col = i % (BLOCK_COL_WARPS * WMMA_N);
            if ((step + row) < k && (global_col_base + col) < n)
            {
                b_s[row][col] = b[(step + row) * n + (global_col_base + col)];
            }
            else
            {
                b_s[row][col] = __float2half(0.0f);
            }
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        // wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
        int shared_a_row = warp_row * WMMA_M;
        int shared_b_col = warp_col * WMMA_N;
        wmma::load_matrix_sync(a_frag, &a_s[shared_a_row][0], WMMA_K);
        wmma::load_matrix_sync(b_frag, &b_s[0][shared_b_col], BLOCK_COL_WARPS * WMMA_N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }
    int global_row = global_row_base + (warp_row * WMMA_M);
    int global_col = global_col_base + (warp_col * WMMA_N);

    if (global_row < m && global_col < n)
    {
        wmma::store_matrix_sync(&c[global_row * n + global_col], c_frag, n, wmma::mem_row_major);
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
        wmma_basic_kernel<<<gridDim, blockDim>>>(a, b, c, m, n, k);
    }
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++)
    {
        wmma_basic_kernel<<<gridDim, blockDim>>>(a, b, c, m, n, k);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / iterations;
}


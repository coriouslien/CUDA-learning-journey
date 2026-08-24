// wmma_double_buff_async.cu

#include "wmma_double_buff_async.h"
#include <cstdio>
#include <random>
#include <omp.h>

WMMAMultiple::WMMAMultiple(int m, int n, int k)
    : m_(m), n_(n), k_(k),
      d_a(nullptr), d_b(nullptr), d_c(nullptr),
      h_a(nullptr), h_b(nullptr), h_c(nullptr),
      h_c_ref(nullptr), h_c_gpu(nullptr)
{
    size_a = m_ * k_ * sizeof(half);
    size_b = k_ * n_ * sizeof(half);
    size_c = m_ * n_ * sizeof(float);

    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cudaMallocHost(&h_a, size_a);
    cudaMallocHost(&h_b, size_b);
    cudaMallocHost(&h_c, size_c);

    cudaMallocHost(&h_c_ref, size_c);
    cudaMallocHost(&h_c_gpu, size_c);
}
WMMAMultiple::~WMMAMultiple()
{
    if (d_a)
        cudaFree(d_a);
    if (d_b)
        cudaFree(d_b);
    if (d_c)
        cudaFree(d_c);
    if (h_a)
        cudaFreeHost(h_a);
    if (h_b)
        cudaFreeHost(h_b);
    if (h_c)
        cudaFreeHost(h_c);
    if (h_c_ref)
        cudaFreeHost(h_c_ref);
    if (h_c_gpu)
        cudaFreeHost(h_c_gpu);
}
void WMMAMultiple::initData()
{
    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < m_ * k_; i++)
    {
        h_a[i] = __float2half(dist(rng));
    }
    for (int i = 0; i < k_ * n_; i++)
    {
        h_b[i] = __float2half(dist(rng));
    }
}
void WMMAMultiple::hostToDevice()
{
    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
}
void WMMAMultiple::runWMMABasic(int iterations)
{
    float ms = launchWMMABasic(d_a, d_b, d_c, m_, n_, k_,
                               block_size, TILE_M, TILE_N, iterations);
    printf("elapsed time: %f ms\n", ms);
}
float WMMAMultiple::deviceToHost()
{
    cudaMemcpy(h_c_gpu, d_c, size_c, cudaMemcpyDeviceToHost);
    return *h_c_gpu;
}
void WMMAMultiple::hostMatmul()
{
    for (int row = 0; row < m_; row++)
    {
        for (int col = 0; col < n_; col++)
        {
            float sum = 0.0f;
            for (int step = 0; step < k_; step++)
            {
                float a_val = __half2float(h_a[row * k_ + step]);
                float b_val = __half2float(h_b[step * n_ + col]);
                sum += a_val * b_val;
            }
            h_c_ref[row * n_ + col] = sum;
        }
    }
}
bool WMMAMultiple::verify()
{
    bool success = true;
    float max_error = 0.0f;
    const float EPSILON = 1e-2f;
#pragma omp parallel for
    for (int i = 0; i < m_ * n_; i++)
    {
        float diff = std::abs(h_c_ref[i] - h_c_gpu[i]);
        if (diff > max_error)
        {
            max_error = diff;
        }
        if (max_error > EPSILON)
        {
            printf("mismatch at index %d host: %f device: %f\n",
                   i, h_c_ref[i], h_c_gpu[i]);
            success = false;
            break;
        }
    }
    if (success)
    {
        printf("SUCCESS: device result match host result, max error: %f\n", max_error);
    }
    else
    {
        printf("FAILED: device result do not match host result.\n");
    }
    return success;
}
void WMMAMultiple::verifyResult()
{

    deviceToHost();
    // Only run CPU verification for appropriately sized matrices
    if (m_ <= 16384 && n_ <= 16384 && k_ <= 16384)
    {
        printf("Running CPU Verification\n");
        hostMatmul();
        verify();
    }
    else
    {
        printf("Skipping CPU verification (Matrices are too large for single-threaded CPU).\n");
    }
}

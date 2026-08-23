// wmma_multi.cu

#include "wmma_multi.h"
#include <cstdio>
#include <cmath>

WMMAMultiple::WMMAMultiple(int m, int n, int k)
    : m_(m), n_(n), k_(k),
      d_a(nullptr),
      d_b(nullptr),
      d_c(nullptr)
{
    size_a = m_ * k_ * sizeof(half);
    size_b = k_ * n_ * sizeof(half);
    size_c = m_ * n_ * sizeof(float);
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
}
WMMAMultiple::~WMMAMultiple()
{
    if (d_a)
        cudaFree(d_a);
    if (d_b)
        cudaFree(d_b);
    if (d_c)
        cudaFree(d_c);
}
void WMMAMultiple::hostToDevice(const std::vector<half> &h_a, const std::vector<half> &h_b)
{
    cudaMemcpy(d_a, h_a.data(), size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), size_b, cudaMemcpyHostToDevice);
}
void WMMAMultiple::runWMMA(int iterations)
{
    float ms = launchWMMABasic(d_a, d_b, d_c, m_, n_, k_,
                               block_size, TILE_M, TILE_N, iterations);
    printf("elepased time: %f ms\n", ms);
}
std::vector<float> WMMAMultiple::deviceToHost()
{
    std::vector<float> h_c(m_ * n_);
    cudaMemcpy(h_c.data(), d_c, m_ * n_ * sizeof(float), cudaMemcpyDeviceToHost);
    return h_c;
}
bool WMMAMultiple::verifyResult(const std::vector<float> &host_result,
                                const std::vector<float> &device_result,
                                int m, int n)
{
    bool success = true;
    float max_error = 0.0f;
    const float EPSILON = 1e-2f;
    for (int i = 0; i < m_ * n_; i++)
    {
        float diff = std::abs(host_result[i] - device_result[i]);
        if (diff > max_error)
        {
            max_error = diff;
        }
        if (diff > EPSILON)
        {
            printf("mismatch at index %d host: %f device: %f\n",
                   i, host_result[i], device_result[i]);
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
void WMMAMultiple::hostMatmul(const std::vector<half> &h_a,
                              const std::vector<half> &h_b, std::vector<float> &h_c, int m, int n, int k)
{
    for (int row = 0; row < m; ++row)
    {
        for (int col = 0; col < n; ++col)
        {
            float sum = 0.0f;
            for (int s = 0; s < k; ++s)
            {
                // Convert half to float for CPU math
                float a_val = __half2float(h_a[row * k + s]);
                float b_val = __half2float(h_b[s * n + col]);
                sum += a_val * b_val;
            }
            h_c[row * n + col] = sum;
        }
    }
}

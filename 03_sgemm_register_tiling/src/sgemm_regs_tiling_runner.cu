
// sgemm_regs_tiling_runner.cu

#include "sgemm_config.h"
#include "sgemm_benchmarker.h"
#include "03_sgemm_register_tiling.h"
#include "check_cuda.h"
#include "nvtx3/nvtx3.hpp"

#include <cstdio>
#include <omp.h>
#include <algorithm>

using Cfg = SgemmCfg;

SgemmBenchmarker::SgemmBenchmarker(int dimension)
    : D(dimension),
      totalElemBytes(0),
      numDimElements(0),
      hostA(nullptr),
      hostB(nullptr),
      hostC(nullptr),
      hostCRef(nullptr),
      deviceA(nullptr),
      deviceB(nullptr),
      deviceC(nullptr)
{
}

SgemmBenchmarker::~SgemmBenchmarker()
{
    delete[] hostA;
    delete[] hostB;
    delete[] hostC;
    delete[] hostCRef;

    if (deviceA)
        cudaFree(deviceA);
    if (deviceB)
        cudaFree(deviceB);
    if (deviceC)
        cudaFree(deviceC);
}

void SgemmBenchmarker::runSgemmRegisterTiling(int iterators)
{
    printf("runSgemmRegisterTiling\n");

    int M = D, N = D, K = D;
    dim3 threads(Cfg::BN / Cfg::TN, Cfg::BM / Cfg::TM);
    // dim3 threads(Cfg::NUM_THREADS);
    dim3 blocks((N + Cfg::BN - 1) / Cfg::BN, (M + Cfg::BM - 1) / Cfg::BM);

    // 1. Correctness check
    {
        nvtx3::scoped_range r{"correctness_check", nvtx3::rgb{0, 255, 0}};
        CHECK_CUDA(cudaMemset(deviceC, 0, totalElemBytes));
        sgemm_register_tiling_kernel<<<blocks, threads>>>(deviceA, deviceB, deviceC, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    printf("1. Correctness check\n");
    verify("sgemm_register_tiling_kernel", 0.0f);

    // 2. Warmup
    {
        nvtx3::scoped_range r{"warmup", nvtx3::rgb{255, 165, 0}};
        CHECK_CUDA(cudaMemset(deviceC, 0, totalElemBytes));
        for (int i = 0; i < 10; i++)
            sgemm_register_tiling_kernel<<<blocks, threads>>>(deviceA, deviceB, deviceC, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // 3. Benchmark
    CHECK_CUDA(cudaMemset(deviceC, 0, totalElemBytes));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    {
        nvtx3::scoped_range r{"benchmark_100_iters", nvtx3::rgb{255, 255, 0}};
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < iterators; i++)
            sgemm_register_tiling_kernel<<<blocks, threads>>>(deviceA, deviceB, deviceC, M, N, K);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
    }

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    printf("3. Benchmark\n");
    verify("sgemm_register_tiling_kernel", ms / iterators);
}

void SgemmBenchmarker::setup()
{
    numDimElements = D * D;
    totalElemBytes = numDimElements * sizeof(float);
    printf("Allocating %.0f MB total for Matrics A, B, and C\n",
           (3 * totalElemBytes) / (1024.0 * 1024.0));
    hostA = new float[numDimElements];
    hostB = new float[numDimElements];
    hostC = new float[numDimElements];
    hostCRef = new float[numDimElements];

#pragma omp parallel for
    // init hostA, hostB and hostC
    for (int i = 0; i < numDimElements; i++)
    {
        hostA[i] = static_cast<float>(rand()) / RAND_MAX;
        hostB[i] = static_cast<float>(rand()) / RAND_MAX;
        hostC[i] = 0.0f;
    }
#pragma omp parallel for collapse(2)
    for (int row = 0; row < D; row++)
    {
        for (int col = 0; col < D; col++)
        {
            float sum = 0.0f;
            for (int k = 0; k < D; k++)
            {
                sum += hostA[D * row + k] * hostB[k * D + col];
            }
            hostCRef[row * D + col] = sum;
        }
    }

    CHECK_CUDA(cudaMalloc(&deviceA, totalElemBytes));
    CHECK_CUDA(cudaMalloc(&deviceB, totalElemBytes));
    CHECK_CUDA(cudaMalloc(&deviceC, totalElemBytes));

    CHECK_CUDA(cudaMemcpy(deviceA, hostA, totalElemBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(deviceB, hostB, totalElemBytes, cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemset(deviceC, 0, totalElemBytes));
}
void SgemmBenchmarker::verify(const char *kernelName, float ms)
{

    CHECK_CUDA(cudaMemcpy(hostC, deviceC, totalElemBytes, cudaMemcpyDeviceToHost));

    float maxError = 0.0f;
    for (int i = 0; i < numDimElements; i++)
    {
        // deviation
        maxError = std::max(maxError, std::abs(hostC[i] - hostCRef[i]));
    }
    double flops = 2.0 * D * D * D;
    double tflops = (flops / (ms / 1000.0)) / 1e12;
    // printf("\n--- %s ---\n", kernelName);
    //  %e: floating-point numbers in scientific (exponential)
    printf("Max error vs CPU: %e\n", maxError);
    printf("Execution Time: %.4f ms\n", ms);
    printf("Performance: %.2f TFLOPS\n\n", tflops);
}

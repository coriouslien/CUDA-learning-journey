
// gemm_engine.cu

#include "gemm_engine.h"
#include "check_cuda.h"
#include "check_cublas.h"
#include <nvtx_profile.h>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <iostream>
#include <cmath>

GemmEngine::GemmEngine(int m, int n, int k, size_t workspace_size)
    : m_(m), n_(n), k_(k),
      d_A_(nullptr),
      d_B_(nullptr),
      d_C_(nullptr),
      d_workspace_(nullptr),
      workspace_size_(workspace_size)
{
    CHECK_CUDA(cudaStreamCreate(&stream_));
    CHECK_CUBLAS(cublasCreate(&handle_));
    CHECK_CUBLAS(cublasSetStream(handle_, stream_));
}

GemmEngine::~GemmEngine()
{

    if (d_A_)
        cudaFree(d_A_);
    if (d_B_)
        cudaFree(d_B_);
    if (d_C_)
        cudaFree(d_C_);
    if (d_workspace_)
        cudaFree(d_workspace_);

    cudaStreamDestroy(stream_);
    cublasDestroy(handle_);
}
void GemmEngine::allocateMemory()
{
    size_A = static_cast<size_t>(m_) * k_ * sizeof(__half);
    size_B = static_cast<size_t>(k_) * n_ * sizeof(__half);
    size_C = static_cast<size_t>(m_) * n_ * sizeof(__half);

    CHECK_CUDA(cudaMalloc(&d_A_, size_A));
    CHECK_CUDA(cudaMalloc(&d_B_, size_B));
    CHECK_CUDA(cudaMalloc(&d_C_, size_C));
    if (workspace_size_ > 0)
    {
        CHECK_CUDA(cudaMalloc(&d_workspace_, workspace_size_));
    }
}
void GemmEngine::hostToDevice(const PinnedBuffer<__half> &h_A,
                              const PinnedBuffer<__half> &h_B)
{
    CHECK_CUDA(cudaMemcpyAsync(d_A_, h_A.data(), size_A,
                               cudaMemcpyHostToDevice,
                               stream_));
    CHECK_CUDA(cudaMemcpyAsync(d_B_, h_B.data(), size_B,
                               cudaMemcpyHostToDevice,
                               stream_));
    CHECK_CUDA(cudaStreamSynchronize(stream_));
}
void GemmEngine::deviceToHost(PinnedBuffer<__half> &h_C)
{
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C_, size_C, cudaMemcpyDeviceToHost));
}

void GemmEngine::executeCublasGemmEx_16F_N_16F(int iterators)
{

    int version;
    cublasGetVersion(handle_, &version);
    printf("cuBLAS version: %d\n", version);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("SM version: %d%d\n", prop.major, prop.minor);
    printf("Device: %s\n", prop.name);

    // Change alpha/beta to __half
    __half alpha_h = __float2half(1.0f);
    __half beta_h = __float2half(0.0f);
    // warmup iteration to eliminate library lazy-loading
    for (int i = 0; i < 10; i++)
    {
        cublasGemmEx(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m_, n_, k_,
                     &alpha_h, d_A_, CUDA_R_16F, m_, d_B_, CUDA_R_16F, k_,
                     &beta_h, d_C_, CUDA_R_16F, m_, CUBLAS_COMPUTE_16F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream_));
    // Add this before your benchmark loop
    cublasStatus_t status = cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH);
    printf("setMathMode status: %d\n", status); // 0 = success
    for (int i = 0; i < iterators; i++)
    {
        // On Blackwell (SM 120), this hint may not be routing to the optimal kernel.
        cublasGemmEx(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m_, n_, k_,
                     &alpha_h, d_A_, CUDA_R_16F, m_, d_B_, CUDA_R_16F, k_,
                     &beta_h, d_C_, CUDA_R_16F, m_, CUBLAS_COMPUTE_16F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        CHECK_CUDA(cudaStreamSynchronize(stream_));
    }
    queryBestAlgorithm(handle_, m_, n_, k_);
}

void GemmEngine::executeCublasGemmEx_16F_N_32F(int iterators)
{

    int version;
    cublasGetVersion(handle_, &version);
    printf("cuBLAS version: %d\n", version);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("SM version: %d%d\n", prop.major, prop.minor);
    printf("Device: %s\n", prop.name);

    float alpha = 1.0f;
    float beta = 0.0f;
    // warmup iteration to eliminate library lazy-loading
    for (int i = 0; i < 10; i++)
    {
        cublasGemmEx(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m_, n_, k_,
                     &alpha, d_A_, CUDA_R_16F, m_, d_B_, CUDA_R_16F, k_,
                     &beta, d_C_, CUDA_R_16F, m_, CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream_));
    // Add this before your benchmark loop
    cublasStatus_t status = cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH);
    printf("setMathMode status: %d\n", status); // 0 = success
    for (int i = 0; i < iterators; i++)
    {
        // On Blackwell (SM 120), this hint may not be routing to the optimal kernel.
        cublasGemmEx(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m_, n_, k_,
                     &alpha, d_A_, CUDA_R_16F, m_, d_B_, CUDA_R_16F, k_,
                     &beta, d_C_, CUDA_R_16F, m_, CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        CHECK_CUDA(cudaStreamSynchronize(stream_));
    }
    queryBestAlgorithm(handle_, m_, n_, k_);
}
void GemmEngine::executeCublasGemmEx_32F_FAST_16F(int iterators)
{

    int version;
    cublasGetVersion(handle_, &version);
    printf("cuBLAS version: %d\n", version);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("SM version: %d%d\n", prop.major, prop.minor);
    printf("Device: %s\n", prop.name);

    float alpha = 1.0f;
    float beta = 0.0f;
    // warmup iteration to eliminate library lazy-loading
    for (int i = 0; i < 10; i++)
    {
        cublasGemmEx(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m_, n_, k_,
                     &alpha, d_A_, CUDA_R_16F, m_,
                     d_B_, CUDA_R_16F, k_,
                     &beta, d_C_, CUDA_R_16F, m_,
                     CUBLAS_COMPUTE_32F_FAST_16F, // <-- change this
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream_));
    // Add this before your benchmark loop
    cublasStatus_t status = cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH);
    printf("setMathMode status: %d\n", status); // 0 = success
    for (int i = 0; i < iterators; i++)
    {
        // CUBLAS_COMPUTE_32F_FAST_16F explicitly tells cuBLAS:
        // use FP16 Tensor Cores for accumulation,
        // accept reduced precision." This is the flag that actually enables the HMMA dispatch path.
        cublasGemmEx(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m_, n_, k_,
                     &alpha, d_A_, CUDA_R_16F, m_,
                     d_B_, CUDA_R_16F, k_,
                     &beta, d_C_, CUDA_R_16F, m_,
                     CUBLAS_COMPUTE_32F_FAST_16F, // <-- change this
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        CHECK_CUDA(cudaStreamSynchronize(stream_));
    }
    queryBestAlgorithm(handle_, m_, n_, k_);
}
void GemmEngine::executeCublasLt(int iterators)
{
    // float alpha = 1.0f;
    // float beta = 0.0f;

    __half alpha_h = __float2half(1.0f);
    __half beta_h = __float2half(0.0f);

    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);

    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;

    // cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    //  Change this line in your cublasLt setup:
    cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_16F, CUDA_R_16F);
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, m_, k_, m_);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, k_, n_, k_);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, m_, n_, m_);

    cublasLtMatmulPreference_t preference;
    cublasLtMatmulPreferenceCreate(&preference);

    cublasLtMatmulHeuristicResult_t results[10];
    int returnedResults = 0;
    cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc,
                                   Adesc, Bdesc, Cdesc, Cdesc,
                                   preference, 10, results, &returnedResults);

    // warmup
    for (int i = 0; i < 10; i++)
    {
        cublasLtMatmul(ltHandle, operationDesc,
                       &alpha_h, d_A_, Adesc, d_B_, Bdesc,
                       &beta_h, d_C_, Cdesc, d_C_, Cdesc,
                       &results[4].algo, // algo[4]: lowest wavesCount
                       nullptr, 0, stream_);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream_));

    for (int i = 0; i < iterators; i++)
    {
        cublasLtMatmul(ltHandle, operationDesc,
                       &alpha_h, d_A_, Adesc, d_B_, Bdesc,
                       &beta_h, d_C_, Cdesc, d_C_, Cdesc,
                       &results[4].algo,
                       nullptr, 0, stream_);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream_));

    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatmulDescDestroy(operationDesc);
    cublasLtDestroy(ltHandle);
}
void GemmEngine::queryBestAlgorithm(cublasHandle_t handle, int m, int n, int k)
{
    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);

    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;

    cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, m, k, m);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, k, n, k);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, m, n, m);

    cublasLtMatmulHeuristicResult_t results[10];
    int returnedResults = 0;

    cublasLtMatmulPreference_t preference;
    cublasLtMatmulPreferenceCreate(&preference);

    cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc,
                                   Adesc, Bdesc, Cdesc, Cdesc,
                                   preference, 10, results, &returnedResults);

    printf("Returned %d algorithms\n", returnedResults);
    for (int i = 0; i < returnedResults; i++)
    {
        printf("  algo[%d]: wavesCount=%.2f, workspaceSize=%zu\n",
               i,
               results[i].wavesCount,
               results[i].workspaceSize);
    }

    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatmulDescDestroy(operationDesc);
    cublasLtDestroy(ltHandle);
}

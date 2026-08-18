// sgemm_engine.h
#pragma once

#include <cuda_fp16.h>
#include "pinned_buffer.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <cublasLt.h>

class GemmEngine
{
private:
    cublasHandle_t handle_;
    cudaStream_t stream_;

    // device
    __half *d_A_;
    __half *d_B_;
    __half *d_C_;
    void *d_workspace_;

    int m_, n_, k_;
    size_t workspace_size_;

    size_t size_A = 0;
    size_t size_B = 0;
    size_t size_C = 0;

public:
    GemmEngine(int m, int n, int k, size_t workspace_size = 0);
    ~GemmEngine();

    GemmEngine(const GemmEngine &) = delete;
    GemmEngine &operator=(const GemmEngine &) = delete;

    void allocateMemory();
    void hostToDevice(const PinnedBuffer<__half> &h_A, const PinnedBuffer<__half> &h_B);
    void deviceToHost(PinnedBuffer<__half> &h_C);
    void executeCublasGemmEx_16F_N_16F(int iterators);
    void executeCublasGemmEx_16F_N_32F(int iterators);
    void executeCublasGemmEx_32F_FAST_16F(int iterators);
    void executeCublasGemmEx(int iterators);
    void executeCublasLt(int iterators);

    size_t sizeA() const { return static_cast<size_t>(m_) * k_ * sizeof(__half); }
    size_t sizeB() const { return static_cast<size_t>(k_) * n_ * sizeof(__half); }
    size_t sizeC() const { return static_cast<size_t>(m_) * n_ * sizeof(__half); }

    void queryBestAlgorithm(cublasHandle_t handle, int m, int n, int k);
};



# Memory Access Patterns: Pageable vs. Pinned Memory 

## Overview
This project evaluates the performance impact of host memory allocation strategies—specifically **Pageable** vs. **Pinned (Page-Locked)** memory—on CUDA kernel execution. The experiment analyzes a simple 1D array stream processing kernel using NVIDIA Nsight Compute (NCU) to evaluate hardware utilization, DRAM throughput, and instruction latency.

## Hardware & Environment
* **GPU:** NVIDIA GeForce RTX 5080
* **OS:** Ubuntu 24.04 Linux
* **Build System:** CMake (C++17, native CUDA architectures)
* **Profiler:** Nsight Compute (NCU) / Nsight Systems (nsys)

## The Experiment
The benchmark processes a 1D array of $N = 2^{28}$ single-precision floats (approx. 1.07 GB). 
The core kernel, `bandwidth_coalesced`, performs a simple arithmetic operation (1 FLOP per 8 bytes transferred) to ensure the workload is entirely memory-bound:

```cpp
__global__ void bandwidth_coalesced(float *__restrict__ d_out, const float *__restrict__ d_in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        d_out[tid] = d_in[tid] * 2.0f;
    }
}

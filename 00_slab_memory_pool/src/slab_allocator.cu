#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <omp.h>
#include "check_cuda.h"
#include <nvtx3/nvToolsExt.h>

__global__ void process_slab(float *slab, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        slab[tid] *= 2.0f;
    }
}
bool verification(float *slab, int n)
{
    bool is_correct = true;
#pragma omp parallel for reduction(&& : is_correct)

    for (int i = 0; i < n; i++)
    {
        float expected = (float)i * 2.0f;
        if (std::abs(slab[i] - expected) > 1e-5f)
        {
            is_correct = false;
        }
    }
    if (!is_correct)
    {
        for (int i = 0; i < n; i++)
        {
            float expected = (float)i * 2.0f;
            if (std::abs(slab[i] - expected) > 1e-5f)
            {
                printf("Mismatch at index: %d\n", i);
                printf("Expected: %.6f\n", expected);
                printf("Got: %.6f\n", slab[i]);
                printf("Diff: %.6f\n", std::abs(slab[i] - expected));
                break;
            }
        }
        return false;
    }
    else
    {
        printf("Pipeline execution completed successfully. Math is correct\n");
    }
    return true;
}

void slab_memory_pool()
{
    const int N = 1 << 28;
    size_t size = N * sizeof(float);
    float *slab;
    nvtxEventAttributes_t attr = {0};
    attr.version = NVTX_VERSION;
    attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType = NVTX_COLOR_ARGB;
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    const uint32_t colors[] =
        {
            0xFFFF0000, // Red
            0xFF00FF00, // Green
            0xFF0000FF, // Blue
            0xFFFFFF00, // Yellow
            0xFFFF00FF, // Magenta
            0xFF00FFFF, // Cyan
            0xFFFFB6C1  // Pink
        };
    const char *labels[] = {"MallocManaged", "process_slab", "device sync", "cleanup",
                            "prefetch GPU", "prefetch CPU", "event sync"};

    attr.color = colors[0];
    attr.message.ascii = labels[0];
    nvtxRangePushEx(&attr);
    // allocate slab Unified Memory
    CHECK_CUDA(cudaMallocManaged(&slab, size));
    nvtxRangePop();

#pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        slab[i] = (float)i;
    }

    cudaEvent_t ker_start, ker_stop, prefetch_start, prefetch_stop;
    CHECK_CUDA(cudaEventCreate(&ker_start));
    CHECK_CUDA(cudaEventCreate(&ker_stop));
    CHECK_CUDA(cudaEventCreate(&prefetch_start));
    CHECK_CUDA(cudaEventCreate(&prefetch_stop));

    int deviceId;
    CHECK_CUDA(cudaGetDevice(&deviceId));
    attr.color = colors[4];
    attr.message.ascii = labels[4];
    nvtxRangePushEx(&attr);
    CHECK_CUDA(cudaEventRecord(prefetch_start));
    CHECK_CUDA(cudaMemPrefetchAsync(slab, size, deviceId));
    CHECK_CUDA(cudaEventRecord(prefetch_stop));
    CHECK_CUDA(cudaEventSynchronize(prefetch_stop));
    nvtxRangePop();
    float ms_pref_gpu;
    CHECK_CUDA(cudaEventElapsedTime(&ms_pref_gpu, prefetch_start, prefetch_stop));
    float gb_pref_gpu = size / (ms_pref_gpu / 1000.0f) / 1e9;
    printf("Prefetch to GPU: %6.2f ms,  %6.2f GB/s)\n", ms_pref_gpu, gb_pref_gpu);

    int minGridSize, blockSize;
    CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, process_slab));
    int gridSize = (N + blockSize - 1) / blockSize;
    printf("Occupancy: blockSize=%d, minGridSize=%d (need %d blocks for grid)\n",
           blockSize, minGridSize, gridSize);
    attr.color = colors[1];
    attr.message.ascii = labels[1];
    nvtxRangePushEx(&attr);
    CHECK_CUDA(cudaEventRecord(ker_start));
    process_slab<<<gridSize, blockSize>>>(slab, N);
    CHECK_CUDA(cudaEventRecord(ker_stop));
    CHECK_CUDA(cudaEventSynchronize(ker_stop));
    nvtxRangePop();

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, ker_start, ker_stop));
    // kernel both reads the slab and writes back to the slab
    float gb_ker = 2.0f * size / (ms / 1000.0f) / 1e9;
    printf("slab process: %6.2f ms | kernel: %6.2f GB/s)\n", ms, gb_ker);
    attr.color = colors[5];
    attr.message.ascii = labels[5];
    nvtxRangePushEx(&attr);
    CHECK_CUDA(cudaEventRecord(prefetch_start));
    CHECK_CUDA(cudaMemPrefetchAsync(slab, size, cudaCpuDeviceId));
    CHECK_CUDA(cudaEventRecord(prefetch_stop));
    CHECK_CUDA(cudaEventSynchronize(prefetch_stop));
    nvtxRangePop();
    float ms_pref_cpu;
    CHECK_CUDA(cudaEventElapsedTime(&ms_pref_cpu, prefetch_start, prefetch_stop));
    float gb_pref_cpu = size / (ms_pref_cpu / 1000.0f) / 1e9;
    printf("prefetch to CPU: %6.2f ms, %6.2f GB/s)\n", ms_pref_cpu, gb_pref_cpu);

    verification(slab, N);

    attr.color = colors[3];
    attr.message.ascii = labels[3];
    nvtxRangePushEx(&attr);
    CHECK_CUDA(cudaEventDestroy(ker_start));
    CHECK_CUDA(cudaEventDestroy(ker_stop));
    CHECK_CUDA(cudaEventDestroy(prefetch_start));
    CHECK_CUDA(cudaEventDestroy(prefetch_stop));
    CHECK_CUDA(cudaFree(slab));
    nvtxRangePop();
}

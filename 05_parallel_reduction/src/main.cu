
//  main.cu
#include "parallel_reduction.h"


int main(void)
{
    warmup();
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("Memory: %.0f MB\n", prop.totalGlobalMem / (1024.0 * 1024.0));
    printf("Memory Bus Width: %d bit\n", prop.memoryBusWidth);
    printf("Peak Bandwidth: %.0f GB/s\n\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);

    const int numElements = 1 << 28; // 268,435,456 elements, ~2G
    size_t totalBytes = numElements * sizeof(cuFloatComplex);
    double datasetSizeGb = static_cast<double>(totalBytes) / 1e9;
    const int ITERS = 100;

    atomicAddKernelNaive(numElements, ITERS);
    {
        float ms = atomicAddKernelSharedMem(numElements, ITERS);
        printf("atomic_add_kernel_shared_mem exec time: %8.2f ms\n", ms);
    }
    {
        float ms = partialStoreSharedMem(numElements, ITERS);
        printf("partial_store_shared exec time: %8.2f ms\n", ms);
    }
    {
        float ms = atomicAddPartialStoreSharedMem(numElements, ITERS);
        printf("atomic_add_partial_store_shared exec time: %8.2f ms\n", ms);
    }

    return 0;
}

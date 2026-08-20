
// parallel_reduction.cu

#include "parallel_reduction.h"
#include "parallel_reduction_kernel.h"

void verify_result(cuFloatComplex *deviceOut, int numElements, float ms)
{
    cuFloatComplex result;
    const double expected_real = (double)numElements * 1.0;
    const double expected_imag = (double)numElements * 0.5;

    cudaMemcpy(&result, deviceOut, sizeof(cuFloatComplex),
               cudaMemcpyDeviceToHost);
    printf("Result:   %.0f + %.0fi\n", cuCrealf(result), cuCimagf(result));
    printf("Expected: %.0f + %.0fi\n", expected_real, expected_imag);
    printf("Time: %.3f\n", ms);
}
void initDeviceData(cuFloatComplex *deviceIn, int numElements)
{
    NVTX_PROFILE_FUNC();

    cuFloatComplex *hostIn = new cuFloatComplex[numElements];
#pragma omp parallel for
    for (int i = 0; i < numElements; i++)
    {
        hostIn[i] = make_cuFloatComplex(1.0f, 0.5f);
    }
    // copy host data to the GPU
    NVTX_PROFILE_SCOPE("Memory Transfer, H2D");
    CHECK_CUDA(cudaMemcpy(deviceIn, hostIn,
                          numElements * sizeof(cuFloatComplex),
                          cudaMemcpyHostToDevice));
    delete[] hostIn;
    hostIn = nullptr;
}

float atomicAddKernelNaive(int numElements, int iterations)
{
    printf("atomicAddKernelNaive....\n");
    NVTX_PROFILE_FUNC();
    size_t totalBytes = numElements * sizeof(cuFloatComplex);

    int blockSize, minGrid;
    CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize,
                                                  atomic_add_kernel_naive));
    int gridSize = (numElements + blockSize - 1) / blockSize;
    if (gridSize > 32 * minGrid)
    {
        gridSize = 32 * minGrid;
    }

    printf("blockSize = %d\n", blockSize);
    printf("gridSize  = %d\n", gridSize);
    printf("total threads = %d\n", gridSize * blockSize);
    // warm up one iteration
    cuFloatComplex *deviceTmp;
    NVTX_PROFILE_SCOPE("Memory allocator");
    CHECK_CUDA(cudaMalloc(&deviceTmp, totalBytes));
    CHECK_CUDA(cudaFree(deviceTmp));

    cuFloatComplex *deviceIn, *deviceOut;
    NVTX_PROFILE_SCOPE("Memory allocator");
    CHECK_CUDA(cudaMalloc(&deviceIn, totalBytes));
    // deviceOut is not an array, it is a single cuFloatComplex value.
    CHECK_CUDA(cudaMalloc(&deviceOut, sizeof(cuFloatComplex)));

    initDeviceData(deviceIn, numElements);
    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;
    CHECK_CUDA(cudaEventRecord(start));
    NVTX_PROFILE_SCOPE("atomic_add_kernal_naive exec");
    // iterations to get a stable average
    for (int i = 0; i < iterations; i++)
    {

        CHECK_CUDA(cudaMemset(deviceOut, 0, sizeof(cuFloatComplex)));
        atomic_add_kernel_naive<<<gridSize, blockSize>>>(deviceIn, deviceOut,
                                                         numElements);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    NVTX_PROFILE_SCOPE("Memory Transfer, D2H");

    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    verify_result(deviceOut, numElements, ms / iterations);

    CHECK_CUDA(cudaFree(deviceIn));
    CHECK_CUDA(cudaFree(deviceOut));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    // return the average time
    return ms / iterations;
}
float atomicAddKernelSharedMem(int numElements, int iterations)
{
    printf("\natomicAddKernelSharedMem....\n");
    NVTX_PROFILE_FUNC();
    size_t totalBytes = numElements * sizeof(cuFloatComplex);
    int blockSize;
    int minGridSize;
    CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(&minGridSize,
                                                  &blockSize,
                                                  atomic_add_kernel_shared_mem));
    int powerOfTwoBlockSize = 1;
    while ((powerOfTwoBlockSize * 2) <= blockSize)
    {
        powerOfTwoBlockSize *= 2;
    }
    int gridSize = (numElements + powerOfTwoBlockSize - 1) / powerOfTwoBlockSize;
    // int sharedMemBytes = 2 * powerOfTwoBlockSize * sizeof(float);
    //  we can also use the following...
    int sharedMemBytes = powerOfTwoBlockSize * sizeof(cuFloatComplex);
    printf("number of thread per block: %d\n", powerOfTwoBlockSize);
    printf("blockSize = %d\n", blockSize);
    printf("gridSize  = %d\n", gridSize);
    printf("total threads = %d\n", gridSize * powerOfTwoBlockSize);
    printf("shared memory bytes: %d\n", sharedMemBytes);

    // warm up one iteration
    cuFloatComplex *deviceTmp;
    CHECK_CUDA(cudaMalloc(&deviceTmp, totalBytes));
    CHECK_CUDA(cudaFree(deviceTmp));

    cuFloatComplex *deviceIn, *deviceOut;
    NVTX_PROFILE_SCOPE("Memory Transfer");
    CHECK_CUDA(cudaMalloc(&deviceIn, totalBytes));
    // deviceOut is not an array, it is a final single cuFloatComplex value.
    CHECK_CUDA(cudaMalloc(&deviceOut, sizeof(cuFloatComplex)));

    initDeviceData(deviceIn, numElements);
    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;
    CHECK_CUDA(cudaEventRecord(start));
    NVTX_PROFILE_SCOPE("atomic_add_kernal_shared exec");

    // iterations to get a stable average
    for (int i = 0; i < iterations; i++)
    {

        CHECK_CUDA(cudaMemset(deviceOut, 0, sizeof(cuFloatComplex)));
        atomic_add_kernel_shared_mem<<<gridSize, powerOfTwoBlockSize, sharedMemBytes>>>(
            deviceIn,
            deviceOut,
            numElements);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    verify_result(deviceOut, numElements, ms / iterations);

    CHECK_CUDA(cudaFree(deviceIn));
    CHECK_CUDA(cudaFree(deviceOut));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    // return the average time
    return ms / iterations;
}

float partialStoreSharedMem(int numElements, int iterations)
{
    printf("\npartialStoreSharedMem....\n");
    NVTX_PROFILE_FUNC();
    size_t totalBytes = numElements * sizeof(cuFloatComplex);
    int blockSize;
    int minGridSize;
    CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(&minGridSize,
                                                  &blockSize,
                                                  partial_store_shared));
    int powerOfTwoBlockSize = 1;
    while ((powerOfTwoBlockSize * 2) <= blockSize)
    {
        powerOfTwoBlockSize *= 2;
    }
    int gridSize = (numElements + powerOfTwoBlockSize - 1) / powerOfTwoBlockSize;
    if (gridSize > 32 * minGridSize)
    {
        gridSize = 32 * minGridSize;
    }
    // powerOfTwoBlockSize is number of thread per block
    // int sharedMemBytes = powerOfTwoBlockSize * sizeof(float);
    // could write as the following..
    int sharedMemBytes = powerOfTwoBlockSize * sizeof(cuFloatComplex);
    printf("number of thread per block: %d\n", powerOfTwoBlockSize);
    printf("blockSize = %d\n", blockSize);
    printf("gridSize  = %d\n", gridSize);
    printf("total threads = %d\n", gridSize * powerOfTwoBlockSize);
    printf("shared memory bytes: %d\n", sharedMemBytes);

    // warm up one iteration
    cuFloatComplex *deviceTmp;
    CHECK_CUDA(cudaMalloc(&deviceTmp, totalBytes));
    CHECK_CUDA(cudaFree(deviceTmp));

    cuFloatComplex *deviceIn, *deviceOut;
    NVTX_PROFILE_SCOPE("Memory Transfer");
    CHECK_CUDA(cudaMalloc(&deviceIn, totalBytes));

    CHECK_CUDA(cudaMalloc(&deviceOut, gridSize * sizeof(cuFloatComplex)));

    initDeviceData(deviceIn, numElements);
    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;
    CHECK_CUDA(cudaEventRecord(start));
    NVTX_PROFILE_SCOPE("atomic_add_kernal_shared_part exec");
    cuFloatComplex *deviceGrandTotal;
    CHECK_CUDA(cudaMalloc(&deviceGrandTotal, sizeof(cuFloatComplex)));

    // iterations to get a stable average
    for (int i = 0; i < iterations; i++)
    {

        CHECK_CUDA(cudaMemset(deviceOut, 0, gridSize * sizeof(cuFloatComplex)));
        partial_store_shared<<<gridSize, powerOfTwoBlockSize, sharedMemBytes>>>(
            deviceIn,
            deviceOut,
            numElements);

        // powerOfTwoBlockSize: number of thread of block
        final_reduce_kernel<<<1, powerOfTwoBlockSize, sharedMemBytes>>>(deviceOut,
                                                                        deviceGrandTotal,
                                                                        gridSize);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    verify_result(deviceOut, numElements, ms / iterations);

    CHECK_CUDA(cudaFree(deviceIn));
    CHECK_CUDA(cudaFree(deviceOut));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    // return the average time
    return ms / iterations;
}

float atomicAddPartialStoreSharedMem(int numElements, int iterations)
{
    printf("\natomicAddpartialStoreSharedMem....\n");
    NVTX_PROFILE_FUNC();
    size_t totalBytes = numElements * sizeof(cuFloatComplex);
    int blockSize;
    int minGridSize;
    CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(&minGridSize,
                                                  &blockSize,
                                                  atomic_add_partial_store));
    int powerOfTwoBlockSize = 1;
    while ((powerOfTwoBlockSize * 2) <= blockSize)
    {
        powerOfTwoBlockSize *= 2;
    }
    int gridSize = (numElements + powerOfTwoBlockSize - 1) / powerOfTwoBlockSize;
    if (gridSize > 32 * minGridSize)
    {
        gridSize = 32 * minGridSize;
    }
    // powerOfTwoBlockSize is number of thread per block
    // int sharedMemBytes = powerOfTwoBlockSize * sizeof(float);
    // could write as the following..
    int sharedMemBytes = powerOfTwoBlockSize * sizeof(cuFloatComplex);
    printf("number of thread per block: %d\n", powerOfTwoBlockSize);
    printf("blockSize = %d\n", blockSize);
    printf("gridSize  = %d\n", gridSize);
    printf("total threads = %d\n", gridSize * powerOfTwoBlockSize);
    printf("shared memory bytes: %d\n", sharedMemBytes);

    // warm up one iteration
    cuFloatComplex *deviceTmp;
    CHECK_CUDA(cudaMalloc(&deviceTmp, totalBytes));
    CHECK_CUDA(cudaFree(deviceTmp));

    cuFloatComplex *deviceIn, *deviceOut;
    NVTX_PROFILE_SCOPE("Memory Transfer");
    CHECK_CUDA(cudaMalloc(&deviceIn, totalBytes));

    CHECK_CUDA(cudaMalloc(&deviceOut, sizeof(cuFloatComplex)));

    initDeviceData(deviceIn, numElements);
    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;
    CHECK_CUDA(cudaEventRecord(start));
    NVTX_PROFILE_SCOPE("atomic_add_partial_stroe exec");

    // iterations to get a stable average
    for (int i = 0; i < iterations; i++)
    {

        CHECK_CUDA(cudaMemset(deviceOut, 0, sizeof(cuFloatComplex)));
        atomic_add_partial_store<<<gridSize, powerOfTwoBlockSize, sharedMemBytes>>>(
            deviceIn,
            deviceOut,
            numElements);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    verify_result(deviceOut, numElements, ms / iterations);

    CHECK_CUDA(cudaFree(deviceIn));
    CHECK_CUDA(cudaFree(deviceOut));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    // return the average time
    return ms / iterations;
}

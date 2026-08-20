// warmup.cu
#include "warmup.h"
#include "check_cuda.h"

__global__ void warmed_up_kernel(float *d_out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (float)tid;
    for (int i = 0; i < 1000; i++)
    {
        val = val * 2.0f - 1.5f;
    }
    if (tid == 0)
    {
        *d_out = val;
    }
}
void warmup()
{
    cudaFree(0);
    float *d_warmup_out;
    CHECK_CUDA(cudaMalloc(&d_warmup_out, sizeof(float)));
    warmed_up_kernel<<<32, 256>>>(d_warmup_out);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(d_warmup_out));
}

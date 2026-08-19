#include <cuda_runtime.h>
#include <warmup.h>
#include <mem_access_patterns_kernels.h>

void warmup()
{
    cudaFree(0);
    float *d_warmup_out;

    cudaMalloc(&d_warmup_out, sizeof(float));
    warmed_up_kernel<<<32, 256>>>(d_warmup_out);
    cudaDeviceSynchronize();
    cudaFree(d_warmup_out);
}

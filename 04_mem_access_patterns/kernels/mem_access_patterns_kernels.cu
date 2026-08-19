#include <mem_access_patterns_kernels.h>


__global__ void bandwidth_coalesced(float *__restrict__ d_out,
                                    const float *__restrict__ d_in,
                                    int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        d_out[tid] = d_in[tid] * 2.0f;
    }
}
__global__ void bandwidth_strided(float *__restrict__ d_out,
                                  const float *__restrict__ d_in,
                                  int n, int stride)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int strided_idx = (tid % stride) * (n / stride) + (tid / stride);

    if (tid < n)
    {
        d_out[strided_idx] = d_in[strided_idx] * 2.0f;
    }
}
__global__ void bandwidth_random(float *__restrict__ d_out,
                                 const float *__restrict__ d_in,
                                 int *indices, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        d_out[tid] = d_in[indices[tid]] * 2.0f;
    }
}
__global__ void warmed_up_kernel(float *d_out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (float)tid;
    for (int i = 0; i < 1000; i++)
    {
        val = (val * 2.0f) - 1.5f;
    }
    if (tid == 0)
    {
        *d_out = val;
    }
}



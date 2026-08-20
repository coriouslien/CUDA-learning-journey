// parallel_reduction_kernel.cu

#include "nvtx3_utils.h"
#include "warmup.h"

#include "parallel_reduction_kernel.h"

__global__ void atomic_add_kernel_naive(const cuFloatComplex *__restrict__ d_in,
                                        cuFloatComplex *__restrict__ d_out,
                                        int num_elements)

{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float r = 0.0f;
    float im = 0.0f;
    for (int i = tid; i < num_elements; i += stride)
    {
        r += cuCrealf(d_in[i]);
        im += cuCimagf(d_in[i]);
    }
    atomicAdd((float *)d_out, r);
    atomicAdd((float *)d_out + 1, im);
}

__global__ void atomic_add_kernel_shared_mem(const cuFloatComplex *__restrict__ d_in,
                                             cuFloatComplex *__restrict__ d_out,
                                             int num_elements)
{
    extern __shared__ float s_data[];
    float *s_real = s_data;
    float *s_imag = &s_data[blockDim.x];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int tx = threadIdx.x;

    float r = 0.0f, im = 0.0f;
    for (int i = tid; i < num_elements; i += stride)
    {
        r += cuCrealf(d_in[i]);
        im += cuCimagf(d_in[i]);
    }
    s_real[tx] = r;
    s_imag[tx] = im;
    __syncthreads();

    // tree reduction, parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tx < s)
        {
            s_real[tx] += s_real[tx + s];
            s_imag[tx] += s_imag[tx + s];
        }
        __syncthreads();
    }
    if (tx == 0)
    {
        atomicAdd((float *)d_out, s_real[0]);     // real part
        atomicAdd((float *)d_out + 1, s_imag[0]); // imaginary part
    }
}

__global__ void partial_store_shared(const cuFloatComplex *__restrict__ d_in,
                                     cuFloatComplex *__restrict__ d_out,
                                     int num_elements)
{
    extern __shared__ float s_data[];
    float *s_real = s_data;
    float *s_imag = &s_data[blockDim.x];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float r = 0.0f;
    float im = 0.0f;
    for (int i = tid; i < num_elements; i += stride)
    {
        r += cuCrealf(d_in[i]);
        im += cuCimagf(d_in[i]);
    }
    s_real[tx] = r;
    s_imag[tx] = im;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tx < s)
        {
            s_real[tx] += s_real[tx + s];
            s_imag[tx] += s_imag[tx + s];
        }
        __syncthreads();
    }
    if (tx == 0)
    {
        // direct store to d_out - each block owns one slot, non-atomic
        d_out[blockIdx.x] = make_cuFloatComplex(s_real[0], s_imag[0]);
    }
}

__global__ void final_reduce_kernel(const cuFloatComplex *__restrict__ d_partial_sums,
                                    cuFloatComplex *__restrict__ d_grand_total,
                                    int num_partials)
{
    extern __shared__ float s_data[];
    float *s_real = s_data;
    float *s_imag = &s_data[blockDim.x];

    int tx = threadIdx.x;
    float r = 0.0f;
    float im = 0.0f;
    for (int i = tx; i < num_partials; i += blockDim.x)
    {
        // d_partial_sums, device in data
        r = +cuCrealf(d_partial_sums[i]);
        im = +cuCimagf(d_partial_sums[i]);
    }
    // write to shared memory
    s_real[tx] = r;
    s_imag[tx] = im;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tx < s)
        {
            s_real[tx] += s_real[tx + s];
            s_imag[tx] += s_imag[tx + s];
        }
        __syncthreads();
    }
    if (tx == 0)
    {
        // d_grand_total, device out
        d_grand_total[0] = make_cuFloatComplex(s_real[0], s_imag[0]);
    }
}

// d_partial: device in
__global__ void atomic_add_partial_store(const cuFloatComplex *__restrict__ d_partial,
                                         cuFloatComplex *__restrict__ d_out,
                                         int total_elements)
{
    extern __shared__ float s_data[];
    float *s_real = s_data;
    float *s_imag = &s_data[blockDim.x];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int tx = threadIdx.x;

    float r = 0.0f, im = 0.0f;
    for (int i = tid; i < total_elements; i += stride)
    {
        r += cuCrealf(d_partial[i]);
        im += cuCimagf(d_partial[i]);
    }
    s_real[tx] = r;
    s_imag[tx] = im;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tx < 0)
        {
            s_real[tx] += s_real[tx + s];
            s_imag[tx] += s_imag[tx + s];
        }
        __syncthreads();
    }
    if (tx == 0)
    {
        atomicAdd((float *)d_partial, s_real[0]);
        atomicAdd((float *)d_partial + 1, s_imag[0]);
    }
}

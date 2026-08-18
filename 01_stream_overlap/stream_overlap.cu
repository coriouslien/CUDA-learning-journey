
#include <cstdio>
#include <cmath> // std::sin, std::cos, std::fabs
#include <iostream>
#include <cuda_runtime.h>
#include <omp.h>
#include <nvtx3/nvToolsExt.h>

#define CHECK_CUDA(call)                                                              \
    {                                                                                 \
        cudaError_t err = call;                                                       \
        if (err != cudaSuccess)                                                       \
        {                                                                             \
            printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(EXIT_FAILURE);                                                       \
        }                                                                             \
    }
__global__ void process_array(float *__restrict__ d_out,
                              float *__restrict__ d_in, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        float val = d_in[tid] * 2.0f;
        for (int i = 0; i < 50; i++)
        {
            val = sinf(val) + cosf(val);
        }
        d_out[tid] = val;
    }
}
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

// int stride = 0)
bool verification(const float *output, const float *input, int N)
{
    int is_correct = 1;
    const float epsilon = 1e-4f;

#pragma omp parallel for reduction(& : is_correct)
    for (int i = 0; i < N; i++)
    {
        float expected = input[i] * 2.0f;
        for (int j = 0; j < 50; j++)
        {
            expected = std::sin(expected) + std::cos(expected);
        }
        if (std::fabs(output[i] - expected) > epsilon)
        {
            is_correct = 0;
        }
    }

    if (!is_correct)
    {
        for (int i = 0; i < N; i++)
        {
            float expected = input[i] * 2.0f;
            for (int j = 0; j < 50; j++)
            {
                expected = std::sin(expected) + std::cos(expected);
            }
            if (std::fabs(output[i] - expected) > epsilon)
            {
                printf("Mismatch at index %d\n", i);
                printf("Expected: %.6f\n", expected);
                printf("Got     : %.6f\n", output[i]);
                printf("Diff    : %.6f\n", std::fabs(output[i] - expected));
                break;
            }
        }
        return false;
    }
    else
    {
        printf("Pipeline execution completed successfully. Math is CORRECT.\n");
    }
    return true;
}
void stream_overlap(int N)
{
    size_t size = N * sizeof(float);
    float *h_in, *h_out;

    // const int NUM_STREAMS = 4;
    const int NUM_STREAMS = 8;
    int chunk_elements = N / NUM_STREAMS;
    cudaStream_t streams[NUM_STREAMS];

    CHECK_CUDA(cudaMallocHost(&h_in, size));
    CHECK_CUDA(cudaMallocHost(&h_out, size));

#pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        h_in[i] = (float)i;
    }
    // cuda timers
    cudaEvent_t start_h2d[NUM_STREAMS], stop_h2d[NUM_STREAMS];
    cudaEvent_t start_d2h[NUM_STREAMS], stop_d2h[NUM_STREAMS];
    cudaEvent_t start_ker[NUM_STREAMS], stop_ker[NUM_STREAMS];

    // Define colors, nvtx
    nvtxEventAttributes_t attr = {0};
    attr.version = NVTX_VERSION;
    attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType = NVTX_COLOR_ARGB;
    const uint32_t colors[] = {
        0xFFFF0000, // Red
        0xFF00FF00, // Green
        0xFF0000FF, // Blue
        0xFFFFFF00, // Yellow
        0xFFFF00FF, // Magenta
        0xFF00FFFF  // Cyan
    };

    const char *labels[] = {"Malloc Async", "H2D", "process_array", "D2H", "Free Async", "Device Synch", "Destroy"};
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;

    for (int i = 0; i < NUM_STREAMS; i++)
    {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
        CHECK_CUDA(cudaEventCreate(&start_h2d[i]));
        CHECK_CUDA(cudaEventCreate(&stop_h2d[i]));
        CHECK_CUDA(cudaEventCreate(&start_d2h[i]));
        CHECK_CUDA(cudaEventCreate(&stop_d2h[i]));
        CHECK_CUDA(cudaEventCreate(&start_ker[i]));
        CHECK_CUDA(cudaEventCreate(&stop_ker[i]));
    }
    float *d_in[NUM_STREAMS], *d_out[NUM_STREAMS];

    attr.color = colors[0]; // red
    attr.message.ascii = labels[0];
    nvtxRangePushEx(&attr);
    for (int i = 0; i < NUM_STREAMS; i++)
    {
        int this_chunk = (i == NUM_STREAMS - 1)
                             ? chunk_elements + (N % NUM_STREAMS)
                             : chunk_elements;
        size_t this_bytes = this_chunk * sizeof(float);
        CHECK_CUDA(cudaMallocAsync(&d_in[i], this_bytes, streams[i]));
        CHECK_CUDA(cudaMallocAsync(&d_out[i], this_bytes, streams[i]));
    }
    nvtxRangePop();

    // the following is an experiment of changing the theadsPerBlock

    for (int i = 0; i < NUM_STREAMS; i++)
    {
        int this_chunk = (i == NUM_STREAMS - 1)
                             ? chunk_elements + (N % NUM_STREAMS)
                             : chunk_elements;
        size_t this_bytes = this_chunk * sizeof(float);

        int offset = i * chunk_elements;
        attr.color = colors[1]; // green
        attr.message.ascii = labels[1];
        nvtxRangePushEx(&attr);
        CHECK_CUDA(cudaEventRecord(start_h2d[i], streams[i]));
        CHECK_CUDA(cudaMemcpyAsync(d_in[i], &h_in[offset], this_bytes,
                                   cudaMemcpyHostToDevice, streams[i]));
        CHECK_CUDA(cudaEventRecord(stop_h2d[i], streams[i]));
        nvtxRangePop();
        // experiment 1, the bad occupancy (too low)
        // int threads = 64;

        // experiment 2, the standard occupancy
        // int threads = 256;

        // experiment 3, the max occupancy
        int threads = 1024;
        int blocks = (this_chunk + threads - 1) / threads;
        CHECK_CUDA(cudaEventRecord(start_ker[i], streams[i]));
        attr.color = colors[2]; // blue
        attr.message.ascii = labels[2];
        nvtxRangePushEx(&attr);
        process_array<<<blocks, threads, 0, streams[i]>>>(d_out[i], d_in[i], this_chunk);
        nvtxRangePop();
        CHECK_CUDA(cudaEventRecord(stop_ker[i], streams[i]));
        CHECK_CUDA(cudaEventRecord(start_d2h[i], streams[i]));
        attr.color = colors[3]; // yellow
        attr.message.ascii = labels[3];
        nvtxRangePushEx(&attr);
        CHECK_CUDA(cudaMemcpyAsync(&h_out[offset], d_out[i], this_bytes,
                                   cudaMemcpyDeviceToHost, streams[i]));
        CHECK_CUDA(cudaEventRecord(stop_d2h[i], streams[i]));
        nvtxRangePop();
    }
    
    attr.color = colors[5]; // cyan
    attr.message.ascii = labels[5];
    nvtxRangePushEx(&attr);
    CHECK_CUDA(cudaDeviceSynchronize());
    nvtxRangePop();

    attr.color = colors[4];
    attr.message.ascii = labels[4];
    nvtxRangePushEx(&attr);
    for (int i = 0; i < NUM_STREAMS; i++)
    {
        cudaFreeAsync(d_in[i], streams[i]);
        cudaFreeAsync(d_out[i], streams[i]);
    }
    nvtxRangePop();

    attr.color = colors[5];
    attr.message.ascii = labels[5];
    nvtxRangePushEx(&attr);
    CHECK_CUDA(cudaDeviceSynchronize());
    nvtxRangePop();

    for (int i = 0; i < NUM_STREAMS; i++)
    {
        int this_chunk = (i == NUM_STREAMS - 1)
                             ? chunk_elements + (N % NUM_STREAMS)
                             : chunk_elements;
        size_t this_bytes = this_chunk * sizeof(float);
        float ms_h2d, ms_ker, ms_d2h;
        CHECK_CUDA(cudaEventElapsedTime(&ms_h2d, start_h2d[i], stop_h2d[i]));
        CHECK_CUDA(cudaEventElapsedTime(&ms_ker, start_ker[i], stop_ker[i]));
        CHECK_CUDA(cudaEventElapsedTime(&ms_d2h, start_d2h[i], stop_d2h[i]));

        float gb_h2d = this_bytes / (ms_h2d / 1000.0f) / 1e9;
        float gb_d2h = this_bytes / (ms_d2h / 1000.0f) / 1e9;
        // 50 iterations * (sinf + cosf + add) = 150 FLOPs per element
        float gflops = (150.0f * this_chunk) / (ms_ker / 1000.0f) / 1e9;

        printf("Stream %d -> H2D: %6.2f ms (%5.1f GB/s) | Kernel: %6.2f ms | (%.1f GFLOP) | D2H: %6.2f ms (%5.1f GB/s)\n",
               i, ms_h2d, gb_h2d, ms_ker, gflops, ms_d2h, gb_d2h);
    }
    verification(h_out, h_in, N);

    // cleanup
    for (int i = 0; i < NUM_STREAMS; i++)
    {
        cudaEventDestroy(start_h2d[i]);
        cudaEventDestroy(stop_h2d[i]);
        cudaEventDestroy(start_d2h[i]);
        cudaEventDestroy(stop_d2h[i]);
        cudaEventDestroy(start_ker[i]);
        cudaEventDestroy(stop_ker[i]);
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }
    CHECK_CUDA(cudaFreeHost(h_in));
    CHECK_CUDA(cudaFreeHost(h_out));
}

void warmup()
{
    cudaFree(0);
    float *d_warmup_out;

    cudaMalloc(&d_warmup_out, sizeof(float));
    warmed_up_kernel<<<32, 256>>>(d_warmup_out);
    cudaDeviceSynchronize();
    cudaFree(d_warmup_out);
}
int main()
{
    warmup();
    const int N = 1 << 28; // N = 2^{28} (~1.07 GB)
    stream_overlap(N);
    return 0;
}

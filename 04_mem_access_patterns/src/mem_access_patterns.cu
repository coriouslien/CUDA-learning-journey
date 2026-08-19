#include <mem_access_patterns.h>
#include <mem_access_patterns_kernels.h>
#include <handle_errors.h>



bool verification(const float *output, const float *expected_in,
                  int N, const char *kernel_name,
                  int *indices = nullptr, // for random
                  int stride = 0)         // for strided
{
    int is_correct = 1;
#pragma omp parallel for reduction(& : is_correct)
    for (int i = 0; i < N; i++)
    {
        float expected;
        int idx;
        if (indices)
        {
            // random: output[i] = input[indices[i]] * 2
            expected = expected_in[indices[i]] * 2.0f;
            idx = i;
        }
        else if (stride > 0)
        {
            // strided: both read and write at strided_idx
            idx = (i % stride) * (N / stride) + (i / stride);
            expected = expected_in[idx] * 2.0f;
        }
        else
        {
            // coalesced: output[i] = input[i] * 2
            expected = expected_in[i] * 2.0f;
            idx = i;
        }

        if (output[idx] != expected)
            is_correct = 0;
    }

    if (!is_correct)
    {
        for (int i = 0; i < N; i++)
        {
            float expected;
            int idx;
            if (indices)
            {
                expected = expected_in[indices[i]] * 2.0f;
                idx = i;
            }
            else if (stride > 0)
            {
                idx = (i % stride) * (N / stride) + (i / stride);
                expected = expected_in[idx] * 2.0f;
            }
            else
            {
                expected = expected_in[i] * 2.0f;
                idx = i;
            }

            if (output[idx] != expected)
            {
                printf("%s mismatch at i=%d: got %f, expected %f\n",
                       kernel_name, i, output[idx], expected);
                break;
            }
        }
        return false;
    }
    return true;
}

void run_benchmark(int N, int *strides, int num_strides, bool use_pinned)
{
    size_t size = N * sizeof(float);
    float *h_in, *h_out_coal, *h_out_strided, *h_out_rand;
    int *h_idx;

    float *d_in;
    float *d_out_coal;
    float *d_out_strided;
    float *d_out_rand;
    int *d_idx;

    printf("\n\n%s Memory Benchmark\n\n", use_pinned ? "Pinned" : "Pageable");
    if (use_pinned)
    {
        HANDLE_ERROR(cudaMallocHost(&h_in, size));
        HANDLE_ERROR(cudaMallocHost(&h_out_coal, size));    // coalesced
        HANDLE_ERROR(cudaMallocHost(&h_out_strided, size)); // stride
        HANDLE_ERROR(cudaMallocHost(&h_out_rand, size));    // random
        HANDLE_ERROR(cudaMallocHost(&h_idx, N * sizeof(int)));
    }
    else
    {
        h_in = new float[N];
        h_out_coal = new float[N];    // coalesced
        h_out_strided = new float[N]; // stride
        h_out_rand = new float[N];    // random
        h_idx = new int[N];
    }

    HANDLE_ERROR(cudaMalloc(&d_in, size));
    HANDLE_ERROR(cudaMalloc(&d_out_coal, size));
    HANDLE_ERROR(cudaMalloc(&d_out_strided, size));
    HANDLE_ERROR(cudaMalloc(&d_out_rand, size));
    HANDLE_ERROR(cudaMalloc(&d_idx, N * sizeof(int)));

#pragma omp parallel
    {
        unsigned int seed = omp_get_thread_num();
#pragma omp for
        for (int i = 0; i < N; i++)
        {
            h_in[i] = (float)i;
            h_idx[i] = rand_r(&seed) % N;
        }
    }
    // cuda timers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    const int NUM_RUNS = 5;
    cudaEvent_t starts[NUM_RUNS], stops[NUM_RUNS];
    // Define colors
    nvtxEventAttributes_t attr = {0};
    attr.version = NVTX_VERSION;
    attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType = NVTX_COLOR_ARGB;

    attr.color = 0xFFFFA500; // Orange
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    // dynamically label based on memory type
    attr.message.ascii = use_pinned ? "H2D_Pinned" : "H2D_Pageable";
    nvtxRangePushEx(&attr);
    cudaEventRecord(start);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx, h_idx, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    nvtxRangePop();
    float ms_h2d;
    cudaEventElapsedTime(&ms_h2d, start, stop);
    float gb_h2d = (size + N * sizeof(int)) / (ms_h2d / 1000.0f) / 1e9;
    printf("H2D Transfer: %.2f ms (%.1f GB/s)\n", ms_h2d, gb_h2d);
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // coalesced
    // Coalesced - green
    attr.color = 0xFF00FF00;
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = "Coalesced";
    // coalesced warm up, no timing
    bandwidth_coalesced<<<blocks, threads>>>(d_out_coal, d_in, N);
    cudaDeviceSynchronize();
    nvtxRangePushEx(&attr);

    for (int r = 0; r < NUM_RUNS; r++)
    {
        cudaEventCreate(&starts[r]);
        cudaEventCreate(&stops[r]);
        cudaEventRecord(starts[r]);
        bandwidth_coalesced<<<blocks, threads>>>(d_out_coal, d_in, N);
        cudaEventRecord(stops[r]);
    }
    cudaDeviceSynchronize();
    float total_ms = 0.0f;
    for (int r = 0; r < NUM_RUNS; r++)
    {
        float ms;
        cudaEventElapsedTime(&ms, starts[r], stops[r]);
        total_ms += ms;
        cudaEventDestroy(starts[r]);
        cudaEventDestroy(stops[r]);
    }
    float ms_coal = total_ms / NUM_RUNS;
    nvtxRangePop();
    // strided
    const uint32_t colors[] = {
        0xFFFF0000, // Red
        0xFF00FF00, // Green
        0xFF0000FF, // Blue
        0xFFFFFF00, // Yellow
        0xFFFF00FF, // Magenta
        0xFF00FFFF  // Cyan
    };
    int current_stride = 0;
    // warm up strided, no timing.
    cudaMemcpy(h_out_strided, d_out_strided, size, cudaMemcpyDeviceToHost);
    for (int s = 0; s < num_strides; s++)
    {
        char label[64];
        snprintf(label, sizeof(label), "Strided_%d", strides[s]);
        attr.color = colors[s % 6]; // Cycle through colors
        attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
        attr.message.ascii = label;
        attr.payloadType = NVTX_PAYLOAD_TYPE_INT64;
        // attach the stride number 2,4,8,16...256
        attr.payload.llValue = strides[s];
        nvtxRangePushEx(&attr);

        current_stride = strides[s];
        cudaEvent_t starts[NUM_RUNS], stops[NUM_RUNS];
        for (int r = 0; r < NUM_RUNS; r++)
        {
            cudaEventCreate(&starts[r]);
            cudaEventCreate(&stops[r]);
            cudaEventRecord(starts[r]);
            bandwidth_strided<<<blocks, threads>>>(d_out_strided, d_in, N, current_stride);
            cudaEventRecord(stops[r]);
        }
        cudaDeviceSynchronize();
        float total_ms_strided = 0.0f;
        for (int r = 0; r < NUM_RUNS; r++)
        {
            float ms_str;
            cudaEventElapsedTime(&ms_str, starts[r], stops[r]);
            total_ms_strided += ms_str;
            cudaEventDestroy(starts[r]);
            cudaEventDestroy(stops[r]);
        }
        float ms_strided = total_ms_strided / NUM_RUNS;
        nvtxRangePop();

        attr.color = 0xFFFFA500; // Orange
        attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
        attr.message.ascii = use_pinned ? "D2H_Pinned_Transfers" : "D2H_Pageable_Transfers";
        nvtxRangePushEx(&attr);
        float ms_d2h, gb_d2h;

        cudaEventRecord(start);
        cudaMemcpy(h_out_strided, d_out_strided, size, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms_d2h, start, stop);
        gb_d2h = size / (ms_d2h / 1000.0f) / 1e9;

        nvtxRangePop();
        // verify specific stride using OpenMP

        float gb_strided = (2.0f * size) / (ms_strided / 1000.0f) / 1e9;
        nvtxRangePushA("CPU Verification-stride");

        bool correct = verification(h_out_strided, h_in, N, "Strided", nullptr, current_stride);
        printf("Stride %3d: %8.3f ms (%.1f GB/s)\n   D2H: %.2f ms (%.1f GB/s)\n   result: %s\n",
               current_stride, ms_strided, gb_strided, ms_d2h, gb_d2h,
               (correct ? "CORRECT" : "WRONG"));
        nvtxRangePop();
    }
    printf("\n");
    // random
    // Random - blue
    attr.color = 0xFF0000FF;
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = "Random";
    nvtxRangePushEx(&attr);

    for (int r = 0; r < NUM_RUNS; r++)
    {
        cudaEventCreate(&starts[r]);
        cudaEventCreate(&stops[r]);
        cudaEventRecord(starts[r]);
        bandwidth_random<<<blocks, threads>>>(d_out_rand, d_in, d_idx, N);
        cudaEventRecord(stops[r]);
    }
    cudaDeviceSynchronize();
    float total_ms_rand = 0.0f;
    for (int r = 0; r < NUM_RUNS; r++)
    {
        float ms_rand;
        cudaEventElapsedTime(&ms_rand, starts[r], stops[r]);
        total_ms_rand += ms_rand;
        cudaEventDestroy(starts[r]);
        cudaEventDestroy(stops[r]);
    }
    float ms_random = total_ms_rand / NUM_RUNS;
    nvtxRangePop();

    float gb_coal = (2.0f * size) / (ms_coal / 1000.0f) / 1e9;
    float gb_rand = (2.0f * size) / (ms_random / 1000.0f) / 1e9;

    printf("Coalesced:  %.2f ms  (%.1f GB/s)\n", ms_coal, gb_coal);
    printf("Random:     %.2f ms  (%.1f GB/s)\n", ms_random, gb_rand);

    // warm up. no timing and nvtx.
    cudaMemcpy(h_out_coal, d_out_coal, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_rand, d_out_rand, size, cudaMemcpyDeviceToHost);

    attr.color = 0xFFFFA500;
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = use_pinned ? "D2H_Pinned_Transfers" : "D2H_Pageable_Transfers";

    nvtxRangePushEx(&attr);
    cudaEventRecord(start);
    cudaMemcpy(h_out_coal, d_out_coal, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_rand, d_out_rand, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    nvtxRangePop();

    float ms_d2h;
    cudaEventElapsedTime(&ms_d2h, start, stop);
    float gb_d2h = (2.0 * size) / (ms_d2h / 1000.0f) / 1e9;
    printf("\n%s D2H Transfer: %.2f ms (%.1f GB/s)\n",
           use_pinned ? "Pinned" : "Pageable", ms_d2h, gb_d2h);

    // coalesced verification (OpenMP)
    nvtxRangePushA("CPU Verifaction-coal");
    printf("Coalesced result: %s\n",
           verification(h_out_coal, h_in, N, "Coalesced") ? "CORRECT" : "WRONG");

    nvtxRangePop();
    // Random verification (OpenMP)
    nvtxRangePushA("CPU Verification-random");
    printf("Random result: %s\n",
           verification(h_out_rand, h_in, N, "Random", h_idx) ? "CORRECT" : "WRONG");
    nvtxRangePop();

    // cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out_coal);
    cudaFree(d_out_strided);
    cudaFree(d_out_rand);
    cudaFree(d_idx);

    if (use_pinned)
    {
        cudaFreeHost(h_in);
        cudaFreeHost(h_out_coal);
        cudaFreeHost(h_out_strided);
        cudaFreeHost(h_out_rand);
        cudaFreeHost(h_idx);
    }
    else
    {
        delete[] h_in;
        delete[] h_out_coal;
        delete[] h_out_strided;
        delete[] h_out_rand;
        delete[] h_idx;
    }
}

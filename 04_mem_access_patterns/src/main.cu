#include <mem_access_patterns.h>
#include <warmup.h>


int main()
{
    warmup();

    const int N = 1 << 28; // N = 2^{28} (~1.07 GB)

    // strides testing
    int strides[] = {2, 4, 8, 16, 32, 64, 128, 256};
    int num_strides = sizeof(strides) / sizeof(strides[0]);

    // use pageable
    run_benchmark(N, strides, num_strides, false);
    // use pinned
    run_benchmark(N, strides, num_strides, true);
    return 0;
}

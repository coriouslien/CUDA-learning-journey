#include "warmup.h"
#include "wmma_multi.h"
#include <random>

int main()
{
    warmup();
    // int m = 256;
    // int n = 256;
    // int k = 256;
     int m = 2048;
     int n = 2048;
     int k = 2048;
    // int m = 4096;
    // int n = 4096;
    // int k = 4096;
//    int m = 8192;
//    int n = 8192;
//    int k = 8192;
    // int m = 16384;
    // int n = 16384;
    // int k = 16384;
    const int ITERS = 100;
    std::vector<half> h_a(m * k);
    std::vector<half> h_b(k * n);
    std::vector<float> h_c(m * n);
    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < m * k; i++)
    {
        h_a[i] = __float2half(dist(rng));
    }
    for (int i = 0; i < k * n; ++i)
    {
        h_b[i] = __float2half(dist(rng));
    }
    WMMAMultiple wmmaMultiple(m, n, k);
    wmmaMultiple.hostToDevice(h_a, h_b);
    wmmaMultiple.runWMMA(ITERS);
    // ?? 
    std::vector<float> d_c = wmmaMultiple.deviceToHost();
    // Only run CPU verification for appropriately sized matrices
    if (m <= 4096 && n <= 4094 && k <= 4096)
    {
        printf("Running CPU Verification\n");
        wmmaMultiple.hostMatmul(h_a, h_b, h_c, m, n, k);
        wmmaMultiple.verifyResult(h_c, d_c, m, n);
    }
    else
    {
        printf("Skipping CPU verification (Matrices are too large for single-threaded CPU).\n");
    }

    // wmmaMultiple.hostMatmul(h_a, h_b, h_c, m, n, k);
    // wmmaMultiple.verifyResult(h_c, d_c, m, n);
    return 0;
}


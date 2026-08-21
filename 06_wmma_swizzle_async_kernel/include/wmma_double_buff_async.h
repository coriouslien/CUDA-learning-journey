// wmma_double_buff_async.h
#pragma once
#include "wmma_basic.h"

class WMMAMultiple
{
private:
    int m_, n_, k_;

    half *d_a, *d_b;
    float *d_c;

    half *h_a, *h_b;
    float *h_c;

    float *h_c_ref; // CPU matmul result
    float *h_c_gpu; // copy back GPU result

    size_t size_a, size_b, size_c;

    const int block_size = 256;
    const int TILE_M = 128;
    const int TILE_N = 128;

public:
    explicit WMMAMultiple(int m, int n, int k);
    ~WMMAMultiple() noexcept;
    WMMAMultiple(const WMMAMultiple &) = delete; // shallow copy
    WMMAMultiple &operator=(const WMMAMultiple &) = delete;
    void initData();
    void hostToDevice();
    void runWMMABasic(int iterations);
    float deviceToHost();
    void hostMatmul();
    bool verify();
    void verifyResult();
};

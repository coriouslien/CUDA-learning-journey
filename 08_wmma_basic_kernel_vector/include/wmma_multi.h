// wmma_multi.h
#pragma once

#include "wmma_basic.h"

#include <vector>

class WMMAMultiple
{
private:
    int m_, n_, k_;
    half *d_a;
    half *d_b;
    float *d_c;
    size_t size_a;
    size_t size_b;
    size_t size_c;

    // dimension 2x2, 4 warp, 128 threads
    const int block_size = 128;
    const int TILE_M = 32;
    const int TILE_N = 32;

public:
    WMMAMultiple(int m, int n, int k);
    ~WMMAMultiple();
    void hostToDevice(const std::vector<half> &h_a, const std::vector<half> &h_b);
    void runWMMA(int iterations = 100);
    std::vector<float> deviceToHost();
    bool verifyResult(const std::vector<float> &hostResult,
                      const std::vector<float> &deviceResult, int m, int n);
    void hostMatmul(const std::vector<half> &h_a,
                    const std::vector<half> &h_b, std::vector<float> &h_c, int m, int n, int k);
};

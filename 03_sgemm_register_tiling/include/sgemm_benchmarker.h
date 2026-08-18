// sgemm_benchmarker.h
#pragma once

#include <cstddef>

class SgemmBenchmarker
{
private:
    int D; // matrix dimenstion (D * D) dim * dim
    size_t totalElemBytes;
    int numDimElements;

    float *hostA;
    float *hostB;
    float *hostC;
    float *hostCRef;

    float *deviceA;
    float *deviceB;
    float *deviceC;

    void verify(const char *kernelName, float ms);

public:
    SgemmBenchmarker(int dimension);
    ~SgemmBenchmarker();

    void setup();
    void runSgemmRegisterTiling(int iterators);
};

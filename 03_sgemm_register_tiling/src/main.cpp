/*
cmake --build build
[ 20%] Building CXX object CMakeFiles/sgemm_benchmarker.dir/src/main.cpp.o
[ 40%] Building CUDA object CMakeFiles/sgemm_benchmarker.dir/src/warmup.cu.o
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z16warmed_up_kernelPf' for 'sm_120'
ptxas info    : Function properties for _Z16warmed_up_kernelPf
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 8 registers, used 0 barriers
ptxas info    : Compile time = 10.896 ms
[ 60%] Building CUDA object CMakeFiles/sgemm_benchmarker.dir/src/sgemm_regs_tiling_runner.cu.o
ptxas info    : 0 bytes gmem
[ 80%] Building CUDA object CMakeFiles/sgemm_benchmarker.dir/kernels/03_sgemm_register_tiling.cu.o
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z28sgemm_register_tiling_kernelPKfS0_Pfiii' for 'sm_120'
ptxas info    : Function properties for _Z28sgemm_register_tiling_kernelPKfS0_Pfiii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 128 registers, used 1 barriers, 8192 bytes smem
ptxas info    : Compile time = 26.746 ms
[100%] Linking CXX executable sgemm_benchmarker
[100%] Built target sgemm_benchmarker

output:
Allocating 192 MB total for Matrics A, B, and C
runSgemmRegisterTiling
1. Correctness check
Max error vs CPU: 1.094961e+03
Execution Time: 0.0000 ms
Performance: inf TFLOPS

3. Benchmark
Max error vs CPU: 1.094961e+03
Execution Time: 4.8790 ms
Performance: 28.17 TFLOPS


  */
#include "warmup.h"
#include "sgemm_benchmarker.h"

#include <cstdio>

int main()
{
    warmup();
    const int ITERS = 100;
    const int dimension = 4096;

    SgemmBenchmarker sgemm(dimension);
    sgemm.setup();
    sgemm.runSgemmRegisterTiling(ITERS);

    return 0;
}

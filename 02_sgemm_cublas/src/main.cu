/*
cmake --build build
-- Configuring done (0.0s)
-- Generating done (0.0s)
-- Build files have been written to: /home/chialan/cuda-learning/a_new_cuda/aug_17/sgemm/sgemm_cublas/build
[ 20%] Building CUDA object CMakeFiles/gemm_engine.dir/src/main.cu.o
ptxas info    : 0 bytes gmem
[ 40%] Building CUDA object CMakeFiles/gemm_engine.dir/src/warmup.cu.o
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z16warmed_up_kernelPf' for 'sm_120'
ptxas info    : Function properties for _Z16warmed_up_kernelPf
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 8 registers, used 0 barriers
ptxas info    : Compile time = 10.249 ms
[ 60%] Building CUDA object CMakeFiles/gemm_engine.dir/src/gemm_engine.cu.o
ptxas info    : 0 bytes gmem
[ 80%] Building CUDA object CMakeFiles/gemm_engine.dir/src/pinned_buffer.cu.o
ptxas info    : 0 bytes gmem
[100%] Linking CUDA executable gemm_engine
[100%] Built target gemm_enginecuBLAS version: 120902
SM version: 120
Device: NVIDIA GeForce RTX 5080
setMathMode status: 0
Returned 8 algorithms
  algo[0]: wavesCount=390.10, workspaceSize=0
  algo[1]: wavesCount=780.19, workspaceSize=0
  algo[2]: wavesCount=195.05, workspaceSize=0
  algo[3]: wavesCount=390.10, workspaceSize=0
  algo[4]: wavesCount=97.52, workspaceSize=0
  algo[5]: wavesCount=624.15, workspaceSize=0
  algo[6]: wavesCount=1560.38, workspaceSize=0
  algo[7]: wavesCount=390.10, workspaceSize=0
16384x16384x16384, Execution Complete. C[0] = 32768.000000


output:
cuBLAS version: 120902
SM version: 120
Device: NVIDIA GeForce RTX 5080
setMathMode status: 0
Returned 8 algorithms
  algo[0]: wavesCount=390.10, workspaceSize=0
  algo[1]: wavesCount=780.19, workspaceSize=0
  algo[2]: wavesCount=195.05, workspaceSize=0
  algo[3]: wavesCount=390.10, workspaceSize=0
  algo[4]: wavesCount=97.52, workspaceSize=0
  algo[5]: wavesCount=624.15, workspaceSize=0
  algo[6]: wavesCount=1560.38, workspaceSize=0
  algo[7]: wavesCount=390.10, workspaceSize=0
16384x16384x16384, Execution Complete. C[0] = 32768.000000

*/

// main.cu
#include "gemm_engine.h"
#include "pinned_buffer.h"
#include "warmup.h"

int main()
{
    warmup();
    const int ITERS = 100;

    // testing
    // int m = 8192;
    // int n = 8192;
    // int k = 8192;
    int m = 16384;
    int n = 16384;
    int k = 16384;
    // int m = 32768;
    // int n = 32768;
    // int k = 32768;

    PinnedBuffer<__half> h_A(static_cast<size_t>(m) * static_cast<size_t>(k));
    PinnedBuffer<__half> h_B(static_cast<size_t>(k) * static_cast<size_t>(n));
    PinnedBuffer<__half> h_C(static_cast<size_t>(m) * static_cast<size_t>(n));

    h_A.initHostData(1.0f);
    h_B.initHostData(2.0f);

    GemmEngine engine(m, n, k, 0);

    // 5. Execute the Pipeline
    engine.allocateMemory();

    // Hand the initialized host buffers over to the engine
    engine.hostToDevice(h_A, h_B);

    engine.executeCublasGemmEx_16F_N_16F(ITERS);

    // engine.executeCublasLt(ITERS);

    // Pull the results back into our h_C buffer
    engine.deviceToHost(h_C);

    // 6. Verify the Math
    // If A is all 1s, B is all 2s, and K is 8192, C[0] should be 16384.
    float first_element = __half2float(h_C.data()[0]);
    printf("%dx%dx%d, Execution Complete. C[0] = %f\n", m, n, k, first_element);

    return 0;
}


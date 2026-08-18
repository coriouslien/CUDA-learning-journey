# CUDA Learning Journey

Early-stage CUDA programs written while building foundational GPU 
programming skills. Code quality and optimization level intentionally 
reflect the learning stage at the time of writing.

These exercises preceded the more advanced work in 
[cuda-gemm-kernels](https://github.com/coriouslien/cuda-gemm-kernels).

## Contents
| Folder | Topic |
|---|---|
| 01_vector_add | Basic kernel launch, grid/block configuration |
| 02_memory_access_patterns | Coalescing, stride access benchmarks |
| 03_parallel_reduction | Shared memory, warp shuffle reduction |
| 04_naive_sgemm | Tiled SGEMM without CuTe |

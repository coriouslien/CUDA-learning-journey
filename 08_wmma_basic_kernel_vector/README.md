<pre>
The kernel dictates data flow through the standard CUDA memory hierarchy: Global Memory → Shared Memory → Registers (Fragments) → Tensor Cores.

Key performance characteristics and optimizations derived from Nsight Compute (NCU) profiling:

Tensor Core Offloading: All matrix math operations are successfully mapped to the Tensor Pipeline (hmma instructions), bypassing standard FMA CUDA cores.

Shared Memory Bottlenecks: Naive 2D shared memory allocations ([32][16] and [16][32]) create severe 12.2-way bank conflicts and uncoalesced wmma::load_matrix_sync accesses, leading to excessive "Long Scoreboard" stalls.

Padding Resolution: Applying a padding offset (PAD = 8) to the leading dimension of the __shared__ arrays breaks the stride alignment. This eliminates the bank conflicts, allowing the kernel to leverage its high arithmetic intensity and reach compute-bound performance.

Prerequisites
Hardware: NVIDIA GeForce RTX 5080 (or equivalent GPU with dedicated Tensor Cores).

Environment: A modern C++ compiler (C++17 or newer) and the CUDA Toolkit.

Build Automation: CMake (version 3.18+ recommended).

Profiling: NVIDIA Nsight Compute (to verify pipeline utilization and memory roofs).

Build Instructions
This project utilizes CMake for straightforward compilation. Execute the following commands from the project root:

cmake -S . -B build
cmake --build build 

  
</pre>

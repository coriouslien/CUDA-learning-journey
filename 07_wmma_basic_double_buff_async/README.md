<pre>
Speed of Light (SOL) Throughput:
Compute (SM) Throughput:  36.68%
Memory Throughput:        88.20%  <-- BOTTLENECK
L1/TEX Cache Throughput:  90.08%  <-- BOTTLENECK
L2 Cache Throughput:      58.54%
DRAM Throughput:          26.48%

- **Compute SM Utilization:** Only **36.68%** of peak compute capacity is achieved. Tensor Core pipe utilization sits at **37.5%**, indicating the execution units are severely starved.
- **Memory Pipeline Saturation:** Memory subsystem reports **88.20%** utilization, driven almost entirely by the **L1/TEX Cache subsystem (90.08%)**.

---

### Memory & Subsystem Breakdown

- **L1 Cache Hit Rate:** 0.76% (L1 is predominantly used as Shared Memory).
- **L2 Cache Hit Rate:** 86.10% (Global matrix tiles exhibit high L2 temporal reuse).
- **Shared Memory Requests:** **100.66 Million requests** executed.
- **Shared Memory Load Bank Conflicts:**
  - **Average Conflict Factor:** **24.1-way conflict** across all shared load requests.
  - **Conflict Ratio:** **83.43%** of wavefronts suffer from bank conflicts (202,795,991 bank conflicts generated).
  - **Estimated Potential Speedup (NCU):** **75.16%**
- **Global Memory Access Efficiency:**
  - **Uncoalesced Excessive Sectors:** **268,435,456 excessive 32B sectors** (33% of 813.7M total sectors).
  - **Estimated Potential Speedup (NCU):** **32.97%**

---

### Warp Stall Statistics

Warp state sampling identifies where warps spend clock cycles:
Warp Stall States (Cycles per Issued Instruction = 17.72 cycles):
Stall Short Scoreboard:    5.80 cycles (32.9%)
Stall Wait:                2.80 cycles
Stall Math Pipe Throttle:  2.40 cycles
Stall Barrier:             2.10 cycles
Stall MIO Throttle:        1.60 cycles
Stall Long Scoreboard:     1.60 cycles
Selected / Executing:      1.20 cycles

- **Stall Short Scoreboard (5.8 cycles / 32.9%):** The dominant stall reason. Warps are stalled waiting for MIO (Memory Input/Output) shared memory load instructions to return data to registers.
- **Branch Efficiency:** **100.0%** (0 divergent branches across 218M branch instructions).

---

### Occupancy & Resource Constraints
Occupancy Metrics:
Theoretical Occupancy:    33.33% (16 active warps / SM out of 48 max)
Achieved Occupancy:       33.13% (15.90 active warps / SM)
Active Blocks per SM:     2 blocks (Hardware limit: 24 blocks)

- **Limiting Factor:** **Register Pressure**. Threads require ~96–100 registers each, restricting the SM to 2 active thread blocks ($2 \times 256 = 512$ threads / 16 warps).
- **Shared Memory Headroom:** Current shared memory usage per block (~22 KB) is well below the hardware capacity threshold (~51 KB for 4 blocks/SM). Shared memory capacity is **not** the occupancy bottleneck.


## Identified Bottlenecks & Root Causes
┌─────────────────────────────────────────────────────────────────────────┐
│                      PRIMARY PERFORMANCE ROADBLOCKS                     │
├─────────────────────────────────────────────────────────────────────────┤
│ 1. Massive Shared Memory Serialization (24.1-way Bank Conflict)         │
│    -> Consecutive threads load matrix rows along identical bank lines.  │
│    -> Causes L1/TEX pipeline to hit 90% load and triggers Short         │
│       Scoreboard stalls.                                                │
│                                                                         │
│ 2. Uncoalesced Global Memory Transactions (33% Extra Traffic)           │
│    -> Asynchronous staging strides cause non-contiguous 32B cache sector│
│       fetches from L2.                                                  │
│                                                                         │
│ 3. Register-Bounded Occupancy (33.33% Theoretical Cap)                  │
│    -> High register count per thread prevents warp latency hiding.      │
└─────────────────────────────────────────────────────────────────────────┘

---

## Optimization Roadmap & Actionable Fixes

### 1. Resolving Shared Memory Bank Conflicts (Swizzling / Padding)
Shared memory comprises 32 banks (4-byte width per bank). When warps load sub-matrices for WMMA fragments, 2D indexing without padding causes stride collisions.

**Solution: Swizzle Indexing / Padding**
```cpp
// 1. Padding approach: Add stride offset to shared memory pitch
constexpr int TILE_K = 64;
constexpr int PAD = 8; // Offset bank alignment
__shared__ half s_A[2][BLOCK_M][TILE_K + PAD];

// 2. Swizzling approach (XOR Pattern)
__device__ __forceinline__ int swizzle(int row, int col) {
    return col ^ (row & 0x7);
}
2. Coalescing Global Memory Transfers (128-bit Vectorized Loads)
Ensure global loads utilize aligned 128-bit memory instructions (cp.async.cg.shared.global [dst], [src], 16):
#include <cuda_pipeline_primitives.h>

// Vectorized 128-bit async copy per thread
int4* smem_ptr = reinterpret_cast<int4*>(&s_A[stage][row][col]);
const int4* gmem_ptr = reinterpret_cast<const int4*>(&g_A[global_idx]);

__pipeline_memcpy_async(smem_ptr, gmem_ptr, sizeof(int4));
__pipeline_commit();

3. Alleviating Register Pressure to Boost OccupancyTarget register allocation $\le 80$ registers/thread to increase active warps per SM from 16 to 24 (boosting occupancy from 33.3% to 50%):Add compiler launch bounds directive:
__global__ void __launch_bounds__(256, 3) wmma_double_buff_async_kernel(...)
Or enforce --maxrregcount=80 in nvcc flags.



</pre>

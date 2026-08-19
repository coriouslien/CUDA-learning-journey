## Register-Tiled SGEMM — FP32 Kernel Progression ##

Hardware: NVIDIA RTX 5080 (SM120, Blackwell) · CUDA 12.8 · Ubuntu 24.04  
Matrix size: 4096 × 4096 × 4096 (FP32)  
Block tile: BM=128, BN=128, BK=8, TM=8, TN=8 · Threads: dim3(16, 16) = 256/block


**Results:**

| Kernel | Time   | TFLOPS | Registers | Blocks/SM | Occupancy | NCU Primary Stall |
|--------|------|--------|-----------|-----------|-----------|-------------------|
| `03_sgemm_register_tiling.cu` | 5.69 ms | ~28 | 128 | 2 | 33.3% | MIO Throttle |
| `03_sgemm_register_tiling_float4.cu` | ~8.0 ms | ~17 | 142 | 1 | 16.7% | Long Scoreboard |
| `03_sgemm_register_tiling_swizzing.cu` | ~9.3 ms | ~17 | 128–168 | 1 | 16.7% | Long Scoreboard |

The baseline kernel wins. The two attempted improvements both regress — and understanding
precisely *why* is the point of keeping all three files.

### Project Structure ###
src/main.cpp: Entry point initializing the benchmark environment.

src/warmup.cu: Executes a dummy computational kernel to spin up the GPU from idle power states.

src/sgemm_regs_tiling_runner.cu: The primary benchmarking harness. Handles host/device memory allocation, cuBLAS reference checks, and iterative kernel launches

### Kernel 1 — `03_sgemm_register_tiling.cu` (Baseline, Best Result) ###

Foundational register tiling implementation utilizing 1D thread mapping and vectorized
global loads.
Register-tiled SGEMM with 2D thread mapping (`dim3(16,16)`). Each thread owns an
8×8 register tile (`c_regs[TM][TN]`). The outer K-loop loads BM×BK and BK×BN tiles
into shared memory (`s_A`, `s_B`), then each thread accumulates its 8×8 output via
an outer-product loop over BK=8 k-slices. `s_B` is read via two `float4` loads per
thread per k-iteration.

### NCU findings (profiled, `sgemm_benchmarker.ncu-rep`) ###


Compute (SM) Throughput:   58.01%
Memory Throughput:         72.94%
L1/TEX Cache Throughput:   77.09%   ← bottleneck
L2 Cache Throughput:        1.44%
DRAM Throughput:            0.31%   ← near zero; everything served from L1/L2

Theoretical Occupancy:     33.33%   ← register-limited: 128 regs × 256 threads = 32768
Block Limit Registers:     2        ← exactly at the 2-block threshold
Active Warps/Scheduler:    3.87
No Eligible [%]:           38.71%

Top stalls: MIO Throttle, Short Scoreboard, Stall Not Selected
Uncoalesced Shared Accesses — Est. Speedup: 36.04%
  → 268M excessive wavefronts (38% of total) from s_B read-path bank conflicts


### What the profile means ###

The kernel is register-limited at exactly 2 blocks/SM — 128 registers × 256 threads
= 32,768 = half of SM120's 65,536-register file. This is the maximum occupancy
achievable for a 256-thread block that needs 128 registers. The MIO Throttle and
Short Scoreboard stalls are the signature of shared memory bank conflicts on the
`s_B` read path: threads in the same warp read `s_B[k][tx*8]` at stride-8, which
aliases banks under the original scalar addressing.

Despite the bank conflicts, this kernel outperforms all variants because **33%
occupancy gives the scheduler 3.87 active warps per scheduler** — enough to hide a
significant portion of the stall latency by switching warps. Reducing occupancy
below this level, even to eliminate the bank conflicts entirely, results in a net loss.

Best FP32 wall-clock result for this tile configuration. The NCU report provides a
clean baseline for comparison against every subsequent variant.


### Kernel 2 — `03_sgemm_register_tiling_float4.cu` (Float4 Attempt) ###

Enhances the baseline by vectorizing the inner-loop shared memory loads into registers via float4 instructions.
Identical to Kernel 1 except `compute_slice` reads `s_B` using `float4` reinterpret
casts instead of scalar indexing. The intent was to replace 8 scalar `LDS`
instructions per k-iteration with 2 `LDS.128` instructions, eliminating the bank
conflicts identified in the Kernel 1 NCU report.

### Why it does not help — static analysis ###

With `dim3(16,16)` and `TN=8`, the two float4 reads per thread per k-iteration are:

b0 = s_B[k][tx * 8]       — column tx*8,   bank = (tx*8) % 32
b1 = s_B[k][tx * 8 + 4]   — column tx*8+4, bank = (tx*8+4) % 32

For tx ∈ [0,15], the b0 banks are: 0, 8, 16, 24, 0, 8, 16, 24, ... The two
halves of the warp (ty=0 and ty=1 share the same tx range) issue identical
addresses → **hardware broadcast, single transaction, zero conflict**. The float4
read path in Kernel 1 was *already conflict-free*. There was no bank conflict to fix
on the read side.

### The register trap ###

ptxas: Used 142 registers, 8192 bytes smem

142 regs × 256 threads = 36,352 > 32,768 (half of 65,536)
→ Block Limit Registers drops from 2 → 1
→ Theoretical Occupancy: 16.67% (half of Kernel 1)
→ Active Warps/Scheduler: 2.00 (vs 3.87)
→ No Eligible [%]: ~58% (vs 38.71%)

The extra 14 registers — from float4 temporaries (`b0`, `b1`) and associated
address scalars kept live across the unrolled k-loop — push the block over the
2-block threshold. Occupancy halves. Long Scoreboard becomes the dominant stall
because with only 2 warps/scheduler, there is nothing to issue while warps wait for
L2/DRAM to service the strided `load_s_A` global reads.

Demonstrates the register-occupancy catch-22 precisely: an optimization that is
correct in isolation (float4 reads, zero bank conflicts) regresses overall
performance because the hardware resource it consumes (registers) costs more than
the stalls it eliminates. This is the most important lesson from this kernel family.

### Kernel 3 — `03_sgemm_register_tiling_swizzing.cu` (Swizzle Attempt) ###

Explores XOR layout permutations to mitigate shared memory bank conflicts during matrix accumulation.
Attempted to eliminate the shared memory bank conflicts from Kernel 1 by applying
an XOR swizzle to the `s_B` write addresses in `load_s_B` and mirroring the same
transform on reads in `compute_slice`.

Two implementations were tried:

**v1 (original):** Column-derived XOR: `swizzled_col = s_col ^ ((s_col >> 5) << 2)`.
Scalar reads in `compute_slice` (8 individual `LDS` per k-iteration).

**v2 (corrected attempt):** Row-index XOR: `swizzled_chunk = chunk ^ s_row`.
Float4 global loads in `load_s_B`. Float4 reads in `compute_slice` with
`phys = (chunk ^ k) * 4` unswizzle.

Both versions profiled at ~9.1–9.3ms — slower than the baseline.

### Why both versions fail ###

**v1 — Wrong swizzle formula.**
`s_col ^ ((s_col >> 5) << 2)` applies a mask of {0, 4, 8, 12} to columns in groups
of 32. For columns 0–31 the mask is 0 (identity — no permutation). The bank
conflicts from stride-8 reads occur *within* a 32-column window; the formula acts
only *between* windows. Bank analysis shows the formula creates new conflicts for
the upper half of the column range while fixing nothing in the lower half.

Replacing float4 reads with scalar address computation also added ~26 registers
(168 total), dropping to 1 block/SM.

**v2 — Swizzle correct, but solving a non-problem.**
The bank conflict analysis proved (as with Kernel 2) that the float4 read path is
already conflict-free. Applying `chunk ^ k` XOR correctly distributes bank
assignments — but there were no conflicts to distribute away. The float4 global
loads in `load_s_B` changed from consecutive (Kernel 1: coalesced) to strided
(v2: chunk-based), **collapsing L1/TEX hit rate from 87.69% to 1.52%** and making
Long Scoreboard the dominant stall.

### NCU signature shared by both swizzle versions ###

L1/TEX Hit Rate:         1.52%   (vs 87.69% in Kernel 1)
Block Limit Registers:   1        (co-limited with shared mem)
Theoretical Occupancy:   16.67%
Active Warps/Scheduler:  2.00
No Eligible [%]:         ~58–59%
Dominant stall:          Long Scoreboard (global load latency exposure)
Uncoalesced Shared:      not flagged (swizzle did eliminate smem conflicts)

The swizzle achieved its stated goal — shared memory bank conflicts gone — but the
cost (occupancy, L1 hit rate) was far higher than the benefit.

Shows the full diagnostic arc: hypothesis → implementation → NCU measurement →
root cause identification → corrected implementation → NCU measurement again →
conclusion. The fact that the fix made things worse twice, for different reasons
each time, is the most technically detailed section of this repository. A reviewer
who reads the commit history and the NCU reports sees the methodology, not just
the outcome.

### The Core Lesson: The Register-Occupancy Catch-22 ###

Every optimization attempt in Kernels 2 and 3 hit the same wall:

Eliminating shared memory bank conflicts requires:
  → float4 reads (need float4 temporaries in registers)
  → swizzle address computation (needs integer temporaries in registers)

Extra registers push beyond the 2-block/SM threshold:
  → 128 regs × 256 threads = 32,768 = exactly 2 blocks ✓
  → 142 regs × 256 threads = 36,352 = only 1 block   ✗

Halved occupancy costs more than eliminated bank conflicts save.

This catch-22 is not fixable within the FP32 register-tiled SGEMM paradigm at this
tile size. The path out is offloading the accumulator to hardware — which is exactly
what WMMA (Tensor Core) kernels do. `HMMA.16816.F32` instructions maintain the 8×8
output tile in hardware registers at a fraction of the software register cost,
breaking the occupancy constraint entirely.

This register-tiled FP32 series exists in this repository as the documented reason
*why* WMMA was the necessary next step, not just the obvious one.


### Hardware Context ###

| Resource | SM120 (RTX 5080) | Constraint for this kernel |
|----------|-----------------|---------------------------|
| Register file | 65,536 regs/SM | 128 regs × 256 threads × 2 blocks = 65,536 (exact fit) |
| Shared memory | 100 KB/SM (configurable) | 8,192 bytes/block — not the binding constraint |
| Max warps/SM | 48 | At 33% occupancy: 16 active warps |
| Schedulers/SM | 4 | 4 warps/scheduler at 33% occupancy |
| SM frequency | 2.29 GHz | |
| DRAM frequency | 14.79 GHz | |


### Build ###

<pre>
cmake --build build
cmake -S . -B build

ncu \
    -f \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section WarpStateStats \
    --section SourceCounters \
    --section SchedulerStats \
    --section Occupancy \
    --import-source yes \
    --export ./build/sgemm_benchmarker \
    --verbose \
    .build/sgemm_benchmarker
</pre>





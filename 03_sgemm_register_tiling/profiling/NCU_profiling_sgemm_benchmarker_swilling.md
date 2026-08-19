This is sgemm_benchmarker_swizzle — the same register-tiled SGEMM kernel with a swizzle applied to eliminate shared memory bank conflicts.

GPU Speed Of Light Throughput
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/72162ef8-b234-44c1-a20c-2e66c6d71f6d" />
The first thing that jumps out: the swizzle version is significantly slower, not faster. 9.07ms vs 5.69ms — 
a 59% regression. All the L1/memory pressure numbers dropped dramatically, but L2 and DRAM increased 
substantially. NCU's top-level diagnosis changed from "High Memory Throughput" to "Latency Issue" — compute 
and memory are both below 60% of peak, which NCU interprets as the kernel being fundamentally latency-bound 
rather than throughput-bound.

This means the swizzle successfully reduced shared memory bank conflict traffic through L1/TEX, but something 
else went badly wrong in the process.
______________________________________________________________________________________________________
Memory Workload Analysis & Scheduler Statistics
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/6325170b-84dd-4645-9433-f79a06c27bcc" />
<pre>
This is the critical clue. L1/TEX hit rate collapsed from 87.69% to 1.52%. This is not a minor degradation — 
it's essentially total L1 cache invalidation. Memory throughput jumped from 2.94 GB/s to 109.94 GB/s because 
now nearly every access misses L1 and goes to L2 or DRAM. Because the L1 cache is being missed 98.5% of the
time, the GPU is forced to fetch data from the L2 cache (which shows a 77.99% hit rate) or directly from DRAM.
This floods the memory subsystem with requests and severely throttles the bandwidth.

The swizzle transformation broke the spatial locality that the float4 vectorized global loads were 
exploiting. A swizzle that reorders how data is laid out in shared memory — or how threads index into it — can 
inadvertently destroy the access pattern that the L1 prefetcher or coalescing logic depended on for the global 
loads, depending on how the swizzle was applied. The L2 hit rate at 77.99% suggests most misses are served 
from L2 rather than DRAM, but L2 latency (~200 cycles) vs L1 (~32 cycles) is still devastating.
</pre>
______________________________________________________________________________________________________
Warps Per Scheduler Chart
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/cc124c3a-f4a1-4cec-9b10-23c324b68c17" />
The waterfall got worse in every dimension. Theoretical warps: ~2 (vs. ~4 before). Active: 2.00 (vs. 3.87). 
Eligible: 0.74 (vs. 1.85). Issued: 0.42 (vs. 0.61). The occupancy got cut roughly in half compared to the 
already-low baseline.
______________________________________________________________________________________________________
Warp State Statistics
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/5cde74c8-19aa-4185-aee0-8f49f89b21a6" />
Warp Cycles Per Issued Instruction: 4.76 (vs. 6.31 baseline). On its face this looks better, but it's 
misleading — the absolute number of instructions issued per unit time actually fell because there are far 
fewer warps and the no-eligible rate skyrocketed. Fewer cycles per issued instruction with worse throughput
means the instructions that do issue are faster, but the scheduler is idle so much more of the time that 
overall throughput collapses.


______________________________________________________________________________________________________
Warp State
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/65d9a3e9-f36e-4dfb-871a-e34a77e9f3ec" />
<pre>
Stall Long Scoreboard is now dominant. This stall type means warps are waiting for the result of a long-
latency instruction — specifically global memory loads (L2/DRAM access, ~200–800 cycles). Stall Long
Scoreboard dominates, the primary reason the warps are sitting idle is Stall Long Scoreboard. 
A "Long Scoreboard" stall occurs when a warp requests data from global memory (L2 cache or DRAM) and has to
wait hundreds of clock cycles for that data to arrive before it can execute the next math instruction. 
Because the L1 cache hit rate is virtually zero, almost every memory fetch triggers a long-latency lookup.
The Compute (SM) pipelines are starving because the data isn't arriving fast enough, which is why the 
overall Compute SOL has dropped to 46.27%

In the baseline, Long Scoreboard was minor because L1 hit rate was 87.69% — loads completed quickly. 
In the swizzle version, L1 hit rate is 1.52%, so nearly every load goes to L2 or DRAM, and warps stall 
for hundreds of cycles waiting for results. This is textbook global memory latency exposure.

Stall MIO Throttle decreased somewhat (bank conflicts partially reduced), and Stall Short Scoreboard (which 
was prominent in the baseline from shared memory replay) largely disappeared — confirming the swizzle did 
achieve its intended effect on shared memory. But the global memory access pattern broke, and the cure was 
worse than the disease.

Stall Wait appearing prominently (new in this profile) indicates warps stalling on __syncthreads() barriers 
while other warps are still stalled on L2 loads — the pipeline is stretched so long that barrier 
synchronization costs became visible.
</pre>
_______________________________________________________________________________________________________
Occupancy
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/1a1696e5-a668-4227-a8a4-2e4320378202" />
Occupancy halved again: 16.67% vs 33.33%. And crucially, both registers and shared memory now limit to 1 
block/SM, where before only registers were the constraint (at 2 blocks). The NCU note says: "limited by the 
number of required registers, and the required amount of shared memory." The swizzle implementation consumed
significantly more shared memory per block — enough to drop the shared memory block limit from 7 to 1, 
matching the register limit. This is the occupancy killer.

The register count also increased (the dot in Image 7 is at ~168 registers/thread vs. ~128 before). The 
swizzle implementation added register pressure on top of consuming more shared memory.
________________________________________________________________________________________________________
Impact of Varying Register Court Per Thread
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/cb5695b4-0ffb-4e91-b7af-941f2f9e3bf2" />
The curve is completely flat at ~16% across all register counts from 1 to 256 (dot at x≈168). This is telling:
reducing registers alone cannot improve occupancy at all, because shared memory is now the co-binding 
constraint at 1 block/SM. No matter how few registers it use, it is still limited to 1 block by shared 
memory. The baseline had a step-down curve with clear improvement potential; the swizzle version has no 
headroom from the register axis whatsoever.
_______________________________________________________________________________________________________
Impact of Varying Block Size
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/592d826e-28b1-4d4f-8b1b-f1da930922af" />
Current block 256 (dot at x=256, ~16%). The curve slopes upward slightly as block size grows toward 384, 
then falls to near-zero above 416. This is the shared memory constraint — larger blocks would need more shared 
memory, but the total smem budget is already consumed by 1 block. The block size axis offers essentially no 
improvement path.
_______________________________________________________________________________________________________
Impact of Varying Shared Memory Usage Per Block
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/704ad065-4b8e-4618-8118-1f37109d651f" />
Current smem (dot at ~9.5KB, ~16%). The curve is completely flat from the minimum all the way to ~98KB, then
falls to zero. This means reducing shared memory per block does not improve occupancy — the register limit is 
equally binding at 1 block/SM. Both limits are simultaneously at 1, so relaxing either one alone is useless;
it would have to reduce both. This is the true double-bind.
_______________________________________________________________________________________________________
Impact Of Varying Block Barriers
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/befce936-088b-4541-b975-9f5042f3a1dc" />
Flat at ~16% up to ~24 barriers then drops to near-zero. No change from baseline structure; barriers are not 
a constraint.
_______________________________________________________________________________________________________
Impact of Varying Cluster Size
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/4f741e7f-e408-4c31-845f-a3d3f22e7291" />

Branch Efficiency: 100%     Avg Divergent Branches: 0
Uncoalesced Global Accesses — Est. Speedup: 0.97%
2,097,152 excessive sectors (2% of total 138,412,032 sectors)

Compare to baseline: Uncoalesced Shared Accesses — Est. Speedup: 36.04% with 268M excessive wavefronts (38%). 
The shared memory bank conflict problem is gone — this is the intended effect of the swizzle. Uncoalesced 
global accesses are only 2%, essentially negligible.

But this confirms the regression is entirely from the occupancy and L1 cache effects, not from new memory 
access pattern problems. The swizzle worked correctly for its stated purpose; the implementation just had 
unacceptable side effects.

Summary:<br>
The swizzle implementation is correct in intent but regressed on two axes simultaneously:

1. Shared memory consumption increased enough to drop Block Limit Shared Mem from 7 → 1. The swizzle likely
2. added padding or extra storage to implement the permuted layout (e.g., a full extra row or column of the
3. tile). This made smem the co-binding occupancy constraint alongside registers, cutting theoretical
4. occupancy from 33.3% to 16.7%.

5. The swizzle disrupted the global load access pattern, collapsing L1/TEX hit rate from 87.69% to 1.52%. This
6. is the more damaging effect. The swizzle likely changed which threads access which addresses in global
7. memory (or changed the shared memory write pattern that feeds from global loads), breaking the coalescing
8. or spatial locality that the L1 prefetcher was exploiting. Nearly every global load now misses to L2, and
9. Long Scoreboard stalls dominate.

The net result: it traded a 36% bank conflict penalty for a near-total L1 eviction, and the L1 miss penalty 
is far more expensive.

What to Fix
-The swizzle as implemented is applied in a way that affects the global→shared load path, not just the 
shared→register read path. A correct swizzle for a register-tiled SGEMM should:

-Apply the permutation only to the shared memory indexing on the read side (thread→smem address mapping for 
the smem[row][col] reads into registers)

-Leave the global→shared write path (the LDG / float4 loads and the STS stores into smem) completely
untouched, so those still land in contiguous smem addresses for good write-side coalescing
Not increase total shared memory capacity — swizzle should be an index remapping, not additional allocation

-The register count increase (128→168) also suggests the swizzle index computation itself may be adding scalar
integer operations that the compiler is keeping live in registers. That extra ~40 registers/thread × 256 
threads × 2 blocks = real register file pressure that dropped Block Limit Registers from 2→1.

-The target outcome: Uncoalesced Shared Accesses near 0% (already achieved), L1 hit rate restored to ~87%, 
-smem per block unchanged, and registers back near 128. That should recover the 33% occupancy and eliminate 
-both the Long Scoreboard and the MIO Throttle stalls.





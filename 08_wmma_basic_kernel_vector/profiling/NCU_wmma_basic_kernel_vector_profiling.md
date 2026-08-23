<pre>
his NCU profile for wmma_basic_kernel, grid (64,64,1) × block (128,1,1), running 1.35 ms / 3,088,088 cycles 
on RTX 5080 at 2.28 GHz.
  
The kernel is currently heavily latency-bound and suffering from severe memory access inefficiencies.
GPU Speed Of Light Throughput
</pre>
<img width="1024" height="590" alt="image" src="https://github.com/user-attachments/assets/9eb273ad-a46b-4188-9f82-bcb94dd8717e" />
<pre>
Latency-bound, not bandwidth-bound.

Both compute and memory are co-limited at ~45% — neither is the clear bottleneck in the traditional 
roofline sense. DRAM at 1.32% tells it the working set is almost entirely cache-resident (L2 serving 
traffic). The "Latency Issue" flag from NCU is correct: the kernel is neither compute-saturated nor bandwidth-
saturated; it's stalling waiting for data that takes too long to arrive, even from L1/L2.
  
The kernel is significantly undervperforming its theoretical limits. Both Compute and Memory Speed Of Light
(SOL) throughputs are stalled at approximately 45.48%. When compute and memory are both this low, it typically
indicates that the GPU is struggling to hide latency and is spending too much time waiting for data rather
than executing instructions.
</pre>
________________________________________________________________________________________________
Tensor Core Operations Roofline
<img width="1024" height="590" alt="image" src="https://github.com/user-attachments/assets/7a400a13-9c3c-49d8-84f5-2d8e9865a967" />
<pre>
The Roofline chart visually plots the kernel's achieved performance against the theoretical hardware limits
of the GPU.

X-Axis (Arithmetic Intensity): Measures how many operations (math) the kernel performs per byte of data
fetched from DRAM. Higher is generally better, indicating less reliance on slow global memory.

Y-Axis (Performance): Measures the actual computational throughput in Operations Per Second (OP/s).

The "Roofs": The sloped diagonal lines represent memory bandwidth bottlenecks. The flat horizontal lines
represent the peak theoretical compute limits of the hardware's math units.

The kernel's data point (the purple dot on the far right) reveals two critical pieces of information:
excellent Memory Reuse: the Arithmetic Intensity is exceptionally high at 1,019.35 OP/byte. This means the
blocking and tiling strategy is highly effective at reusing data once it is brought on-chip. It is firmly
positioned to the right of the ridge point, putting it in "compute-bound" territory.Severe Compute
Underutilization: Despite being in the compute-bound region, the achieved performance is roughly 12.7 
Tera-Operations per second ($12.7 \times 10^{12}$ OP/s). This sits drastically below the horizontal "roof"
line, which represents the peak theoretical limit of the RTX 5080's Tensor Cores.

Starved Math Units
This Roofline chart perfectly visualizes the symptom of the memory access issues we diagnosed earlier.

The high Arithmetic Intensity confirms that the kernel is successfully relying on the tiles stored in 
shared memory rather than fetching repeatedly from global DRAM.

However, the massive gap between the data point and the peak compute roofline means the Tensor Cores 
are sitting idle.

They are not performing matrix multiplication because they are trapped in "Long Scoreboard" stalls, 
waiting for the SM to resolve the massive shared memory bank conflicts caused by the wmma::load_matrix_sync
instructions.
</pre>
________________________________________________________________________________________________
Floating Point Operations Roofline (Half Precision)
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/bcc31618-f664-4a5f-8c26-adcd32e392e1" />
<pre>
The absence of data points on this chart confirms that the kernel is executing its math operations entirely
on the Tensor Cores rather than the standard CUDA cores.
  
</pre>
________________________________________________________________________________________________
Compute Workload Analysis
<img width="1024" height="590" alt="image" src="https://github.com/user-attachments/assets/cc58b72e-e66c-4702-b245-ea699ec2b896" />
<pre>
Pipe Utilization (% of active cycles):

ALU: ~20% — highest-utilized, marked "Balanced"
Tensor (All): ~7–8%
Tensor (FP): ~5–6%
FMA: ~7%

Pipe Utilization (% of peak instructions executed):

LSU: ~45% — by far the highest
ADU: ~25%
ALU: ~20%
Tensor (FP): ~5%

The critical observation: LSU (Load/Store Unit) dominates peak-instruction utilization at ~45%, while Tensor
(FP) sits at ~5%. This is a WMMA kernel spending most of its SM time doing memory operations, not tensor math.
ALU being the highest on the active-cycles chart (integer address computation) is consistent with lots of 
indexing overhead per WMMA tile.
The workload profile confirms that this kernel is currently bottlenecked by memory traffic, not compute math.

The Load/Store Unit (LSU) is the highest-utilized pipeline at roughly 50%, while actual arithmetic units 
(ALU, FMA, Tensor) sit at or below 25%.

Furthermore, the L1/TEX cache hit rate is exceptionally poor at 9.54%. Because data isn't found in L1, 
the GPU must fetch it from L2 or device memory, heavily contributing to the Long Scoreboard stalls.

To fix these issues, it will likely need to pad the shared memory allocations (e.g., adding an offset
column) to break up the stride causing the 12-way bank conflicts, and review the global memory access
patterns to ensure they are fully coalesced to improve L1 hit rates.
  
</pre>

________________________________________________________________________________________________
Memory Workload Analysis
<img width="1024" height="590" alt="image" src="https://github.com/user-attachments/assets/155eaa3e-1dc2-49e4-8366-8c52b0143cbd" />
<img width="1024" height="590" alt="image" src="https://github.com/user-attachments/assets/46990ac3-7bf5-4fdd-8e01-83b10a2b341d" />
<img width="1024" height="590" alt="image" src="https://github.com/user-attachments/assets/7d650ebc-0f32-4742-87d8-14a1ef464d84" />

Critical warning: Shared Load Bank Conflicts

12.2-way average bank conflict
344,496,367 bank conflicts total
Affects 67.28% of the 51,269,592 wavefronts for shared loads
Est. Speedup if fixed: 32.59%

The memory flow (Images 6–7):

16.84M global load instructions → 16.78M requests → L1/TEX (9.54% hit) → L2 (96.93% hit) → only 16.79 MB
reaches Device L2 from DRAM 20.97M shared memory instructions → 4.19M requests + 16.78M requests (the large 
number is bank-conflict replays)
GCC (Global Cache) hit rate: 92.96%

Interpretation: The L1 hit rate of 9.54% is very low — the kernel's global access pattern doesn't reuse L1 
lines well. However, L2 at 96.93% means almost nothing goes to DRAM, which explains DRAM throughput of 1.32%.
The shared memory bank conflicts are severe: with a 12.2-way conflict, each shared load effectively serializes
into ~12 sequential accesses, costing 11/12 of the potential bandwidth. This is the single largest fixable
bottleneck.


________________________________________________________________________________________________
Scheduler Statistics
<img width="1024" height="590" alt="image" src="https://github.com/user-attachments/assets/120bac71-ddbe-4e6f-9a7f-1a80dd837eb0" />

Issue Slot Utilization warning: Est. Speedup 53.73%

The scheduler can issue 1 instruction/cycle but only does so every 2.2 cycles. With 11.27 active warps but
only 0.84 eligible, ~93% of active warps are stalled at any given cycle. The schedulers spend 53.73% of cycles
with nothing to issue — this is the clearest symptom of latency not being hidden by warp parallelism. Even
though theoretical occupancy is 100% (48 warps/SM, 12 blocks × 4 warps per block with 128-thread blocks), 
the actual eligible warp count is catastrophically low.
________________________________________________________________________________________________
Warps Per Scheduler
<img width="1024" height="590" alt="image" src="https://github.com/user-attachments/assets/54ad6f67-ddcd-4de5-b6a2-b3d76d0efc71" />


________________________________________________________________________________________________
Warp State Statistics
<img width="1024" height="590" alt="image" src="https://github.com/user-attachments/assets/703e931f-3a97-42c2-8efd-121def05958b" />


________________________________________________________________________________________________
Warp State (All Cycles)
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/98387f20-4426-433c-8e24-be3da969e894" />
|Stall Type|	~Cycles|
|----------|---------|
|Stall Long Scoreboard|	~15 (dominant)|
|Stall Wait|	~3.5|
|Stall Short Scoreboard|	~3.5|
|Stall Barrier|	~2.5|
|Selected	|~1.5|
|Stall Not Selected|	~1.5|
|Stall MIO Throttle	|~0.5|
<pre>

Long Scoreboard = 62.4% of 24.4 average cycles per instruction. This means warps spend ~15 cycles per
instruction waiting for an L1TEX (global/shared/surface/texture) load to return. NCU confirms: this is a
memory latency stall on L1TEX operations. With only 0.84 eligible warps, there isn't enough warp parallelism
to pipeline over this 15-cycle latency.

Stall Barrier at ~2.5 cycles is secondary — __syncthreads() between WMMA phases.
The Streaming Multiprocessors (SMs) are struggling to keep warps active.

Schedulers report having no eligible warps to issue 53.73% of the time.

The primary cause for this is "Stall Long Scoreboard," which delays warps by roughly 15 cycles per instruction.

This stall state means the threads are locked up waiting on L1TEX (global, local, texture, or surface) memory
operations to return data.
</pre>
________________________________________________________________________________________________
Occupancy
<img width="1024" height="590" alt="image" src="https://github.com/user-attachments/assets/5c994ab1-7807-432a-bfc5-284ac6273914" />
Occupancy is excellent on paper — 12 blocks × 4 warps = 48 warps/SM = 100% theoretical. But Image 9 showed only 0.84 eligible warps despite 11.27 active warps. High occupancy is not helping because warps are stalled in Long Scoreboard, not executing.

The "Impact of Varying Shared Memory" chart (Image 14) shows current allocation sits at the very beginning of the curve (100% occupancy at low smem), then crashes steeply. This is because the kernel uses substantial shared memory (the WMMA accumulator tiles). If shared memory increases even slightly, occupancy collapses — confirming the kernel is near the shared memory limit per SM.

Block barrier sensitivity (Image 15): occupancy drops sharply after ~2 barriers/block, confirming __syncthreads() usage is close to the hardware limit.
_________________________________________________________________________________________________
Impact of Varying Block Size
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/646b46ac-a8f6-4341-a1d7-4c29f54c372d" />


________________________________________________________________________________________________
Impact of Varying Shared Memory Usage Per Block
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/f8c12921-4f9e-473f-bb59-b91a2e1582a5" />

________________________________________________________________________________________________
Impact of Varying Block Barriers & Impact of Varying Cluster Size
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/17eeeb47-4a1b-4dfb-baa0-02ba2e45f496" />



________________________________________________________________________________________________
GPU and Memory Workload Distribution 
<img width="1024" height="590" alt="image" src="https://github.com/user-attachments/assets/ca36c27c-0226-4b0b-86aa-4304483e555e" />
|Metric|	Avg|	Min|	Max|
|------|-----|-----|-----|
|SM Active Cycles|	2,891,719|	2,729,989|	3,062,223|

<pre>
SM Workload Imbalance: Est. Speedup 5.23% — some SMs finish 5.57% earlier than average, some 5.59% later. 
The grid (64,64,1) = 4,096 blocks / ~84 SMs on RTX 5080 ≈ ~48 waves, so tail-effect imbalance is mild but 
present.

SMSPs Workload Imbalance: Est. Speedup 6.42% — sub-SM partitions (4 SMSPs per SM) show slightly worse
imbalance, consistent with the bank conflict pattern disturbing SMSP-level load balance.
</pre>

________________________________________________________________________________________________
Source Counters
<img width="1024" height="590" alt="image" src="https://github.com/user-attachments/assets/b9fe5fba-67da-4950-8a6d-e42f1d4eb0c1" />
<pre>
Uncoalesced Shared Accesses: Est. Speedup 46.94%

33,554,432 excessive wavefronts (50% of 67,108,864 total shared wavefronts)
Primary sources:
mma.hpp:99 — 25,165,824 excessive wavefronts (75%)
mma.hpp:91 — 8,388,608 excessive wavefronts (25%)

These lines in mma.hpp are the WMMA fragment store/load operations. The fragment layout for WMMA (especially
the accumulator in fp32) does not naturally map to conflict-free shared memory banks, causing 75% of the
excessive traffic to come from one store/load pattern.

Warp Stall Sampling top locations: four distinct kernel lines each contributing ~8k–17k stall samples (8–15% 
each) — the stalls are spread across the main compute loop, not isolated.
  
NCU flags a critical bottleneck under "Source Counters" with "Uncoalesced Shared Accesses," highlighting a
massive 46.94% estimated speedup opportunity.
  
</pre>


________________________________________________________________________________________________


________________________________________________________________________________________________



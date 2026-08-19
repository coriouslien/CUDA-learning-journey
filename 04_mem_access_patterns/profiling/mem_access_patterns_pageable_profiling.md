
GPU Speed Of Light Throughout
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/d2b61dfe-4efa-49e2-86d4-8bd020a5dc52" />
Memory throughput equals DRAM throughput at 86.92% — and they are identical, which is the key insight. 
The working set is 2 × 1 GiB (read d_in, write d_out), and every cache level is bypassed: L1 hit rate is 0, 
L2 hit rate is 0 (Image 2). This is expected and correct for a streaming kernel at this scale — 2²⁸ floats × 4
bytes × 2 arrays = 2 GiB blows past both L1 (128 KB per SM on Blackwell) and L2 (well under 2 GiB). 
Every access goes straight to DRAM.

The Compute throughput at 16.87% tells you this is solidly memory-bound. The kernel does one multiply per
thread (* 2.0f), which is essentially free compared to the DRAM latency. SM is mostly idle waiting for data.
This is the correct and expected behavior for a memory bandwidth benchmark.

The NCU "High Throughput" notice — >80% on memory — confirms the kernel is doing exactly what you designed it
to do.
______________________________________________________________________________________________________
Memory Workload Analysis & Scheduler Statistics
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/edd42cff-357c-4b18-b2a9-120819f2b5c5" />
822 GB/s on your RTX 5080 is a strong result. The GB102/GB203 Blackwell memory subsystem has ~960 GB/s theoretical peak on the 5080; 822 GB/s is ~85.6% of theoretical, which tracks precisely with the 86.92% SOL. That is excellent coalesced throughput — you are within ~14% of the hardware ceiling.

L1=0, L2=0 is not a problem here; it's the streaming access pattern working as intended. The working set is too large to benefit from cache reuse.

Schedule Statistics:
This looks alarming at first glance but is actually the correct signature of a DRAM-latency-saturated, bandwidth-optimal kernel. Parse it carefully:

You have 9.51 active warps per scheduler — nearly at theoretical (12 per scheduler, visible in Image 3).
But only 0.08 of those are eligible (ready to issue) per cycle.
The scheduler finds no eligible warp 92.74% of cycles and idles.

This happens because every warp is stalled waiting on its global load to come back from DRAM. DRAM latency is ~400–600 cycles on Blackwell. With 9.51 warps in flight, each warp is covering only a fraction of that latency. This is the DRAM-bandwidth regime: you've pushed enough warps to saturate the memory bus (86.92%), but not enough total in-flight requests to keep schedulers busy. The scheduler idle rate is the price of a bandwidth-bound kernel at this occupancy level.

The issue slot utilization note ("issues once every 13.8 cycles") is a direct consequence — each instruction the scheduler issues is a memory op, then it waits ~120+ cycles for the result.
______________________________________________________________________________________________________
Warp Per Scheduler
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/bf3c9bae-66b6-4a7b-8c2f-37fe2d2d76f4" />
The chart visually confirms:

GPU Maximum Warps Per Scheduler: ~12 (Blackwell SM120 capacity)
Theoretical Warps Per Scheduler: ~12 (your block size of 256 = 8 warps, × 6 blocks per SM = 48 warps / 4 schedulers = 12)
Active Warps Per Scheduler: ~9.5 (80% of theoretical)
Eligible: nearly zero (thin sliver)
Issued: nearly zero

The gap between Active (~9.5) and Eligible (~0.08) is the DRAM stall visualized. All those warps are alive; none are ready to run. They're all waiting for L2→DRAM traffic to return. This is the canonical picture of a memory-latency-limited kernel that is nonetheless achieving near-peak bandwidth, because the aggregate in-flight transaction volume is what saturates the bus, not scheduler throughput.
______________________________________________________________________________________________________
Warp State Statistics
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/e829a95a-bb88-4bd2-9bb1-21aa8ad01f41" />
The dominant stall: Long Scoreboard — 120.2 cycles per instruction, 91.8% of all stall cycles.

Long Scoreboard means the warp issued a load and is waiting for data to come back from a long-latency unit — in this case L1TEX, which is serving global memory loads straight from DRAM. 120 cycles sitting on a DRAM fetch is precisely the expected Blackwell global load latency when L1 and L2 are both missing.

The "Not Predicated Off Threads" of 30.12 vs. 32 active threads is the tail-end boundary effect: the very last block has slightly fewer than 256 threads for N = 2²⁸ = 268,435,456, which is divisible by 256 evenly (268,435,456 / 256 = 1,048,576 blocks exactly). So actually the 30.12 average hints at a slight warp divergence from the if (tid < n) guard — the last block has 256 threads all valid, but some warps in some blocks are partially-predicated. At N = 2²⁸ this is negligible.

The 13.08% estimated speedup NCU shows for both "Issue Slot Utilization" and "Long Scoreboard Stalls" is the same recommendation: better latency hiding. Practically speaking, for a pure streaming kernel at 86.92% of DRAM bandwidth, 13% headroom is acceptable — you would need either persistent kernels or a different access pattern to close it.
_____________________________________________________________________________________________________
Warp State (All Cycles)
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/391ed432-0cdc-4eee-8f79-c375e5049387" />
Stall breakdown (cycles per issued instruction, x-axis to 150):

Stall Long Scoreboard: ~120 cycles — dominates completely (~91%)
Stall Short Scoreboard: ~4–5 cycles — minor (WAR hazards on registers)
Stall Wait: ~4 cycles — minor (structural hazard, e.g., MIO throttle)
Stall Drain, Selected, Branch Resolving, No Instruction: all negligible

This is exactly the expected profile. A pure load-multiply-store kernel has essentially one stall type: waiting for DRAM. Short scoreboard is a distant second from the * 2.0f write-back into the same register. Everything else is noise.
_____________________________________________________________________________________________________
Occupancy
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/9da1f818-509f-4e5e-97ad-b57c67ca4ba3" />
The binding limiter is Block Limit Warps = 6. Work through why:

Block size = 256 threads = 8 warps per block.
Blackwell SM120 supports 48 warps max per SM.
48 / 8 = 6 blocks max per SM → this is the limit.
6 blocks × 8 warps = 48 warps = 100% theoretical occupancy.

So theoretically you're at 100%. But achieved is 80.4%, meaning you're landing ~38.6 warps per SM in practice. The delta is from tail effect — the last few wave-fronts of blocks don't fill all SMs evenly. With 1,048,576 blocks across the GPU, the tail imbalance is minimal in absolute terms, but NCU measures average over the whole execution timeline including the startup/drain phases.

The "Achieved Occupancy Est. Speedup: 13.08%" is NCU saying: "if you closed the 20% occupancy gap, you might get 13% faster." For this kernel, that gain is mostly theoretical — you're already at 86.92% memory SOL, so more warps would only help if additional in-flight transactions could push DRAM utilization from 87% to 100%.
_____________________________________________________________________________________________________
Impact Of Varying Register Count Per Thread
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/8348065c-42d9-4373-bc57-eef9772b0d41" />
The dot at 16 registers per thread = 100% occupancy. Your kernel uses very few registers (the coalesced kernel is tid = blockIdx.x * blockDim.x + threadIdx.x; d_out[tid] = d_in[tid] * 2.0f; — 3–4 registers total). The cliff at 64 registers per thread dropping to ~30% shows the general register file pressure curve for Blackwell SM120. You are nowhere near this cliff.
_____________________________________________________________________________________________________
Impact of Varying Block Size
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/fe2292b4-8093-4a88-ba52-7a1350119694" />
Current block size = 256 (marked with the dot at 100%). The curve is relatively flat from 64–512, meaning this kernel is not sensitive to block size — any multiple of 32 from 64 to 512 gives near-100% occupancy. The drop after 768 is because large blocks reduce the number of resident blocks per SM below the warp-limit.
_____________________________________________________________________________________________________
Impact of Varying Shared Memory Usage Per Block
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/ff1b8aba-287d-4e4f-acb5-6143a89dc62f" />
Dot at 0 shared memory = 100% occupancy. Since bandwidth_coalesced uses zero shared memory, you have maximum flexibility. The steep cliff shows that adding shared memory would immediately hurt occupancy because Blackwell SM120's shared memory budget limits concurrent blocks. This confirms your kernel design is optimal — no shared memory needed for a pure streaming copy.

_____________________________________________________________________________________________________
Impact of Varying Block Barriers
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/a9ba698c-d15d-4829-8638-619277320960" />
Dot at 0 barriers = 100%. No __syncthreads() in your kernel, as expected. Barriers reduce occupancy similarly to shared memory.
____________________________________________________________________________________________________

Impact of Varying Cluster Size
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/4d6d4e8b-2b0f-4663-a96f-b019c99373b4" />


Flat zero active clusters — you're not using Thread Block Clusters (a Hopper+ feature). The chart shows 0 active clusters regardless of potential cluster size. This is correct; you didn't annotate your kernel with \__cluster_dims\__ or use cooperative groups clusters. For a simple streaming kernel this is fine, but it's a feature worth knowing for future work on Blackwell.
Branch Instructions: 16,777,216 — that's exactly 2²⁸ / 16 = N / 32 (one branch per warp for the if (tid < n) guard). With N perfectly divisible by block size, Branch Efficiency would be 100% (no divergence). NCU shows 0 for Branch Efficiency — this is a display artifact when efficiency is undefined because there are no divergent branches.


Warp Stall Sampling table:

Top location: 222,596 samples (92%) → one source line dominates
Second location: 5,393 samples (2%)

The 92% concentration at one source line is the global load d_in[tid] in bandwidth_coalesced. This is exactly where the 120-cycle Long Scoreboard stall lives. NCU's sampler caught nearly every stalled warp at that one instruction.

___________________________________________________________________________________________________
Summary:
For bandwidth_coalesced specifically:

822 GB/s / 86.92% DRAM SOL — excellent. You are achieving ~86% of peak DRAM bandwidth with a simple streaming kernel. This is the right baseline.
L1=0, L2=0 hit rate — correct for a 2 GiB working set. Not a problem.
Long Scoreboard stalls at 120 cycles, 91.8% — the entire execution is DRAM-fetch time. There is no compute work to expose. This is the expected shape of a bandwidth-bound kernel.
80.4% achieved occupancy vs. 100% theoretical — the ~20% gap is tail-effect from block scheduling, not a kernel design flaw. You can't improve it without reformulating the problem.
Scheduler is idle 92.74% of cycles — this is the dual face of #3. The GPU is not being lazy; it's correctly saturating DRAM. The scheduler has nothing to issue because every warp is waiting on DRAM.
Occupancy sensitivity charts — your kernel is at the optimal point on every axis (registers, shared mem, block size, barriers). There is no low-hanging occupancy fruit.

Post the pinned memory screenshots when ready — the interesting comparison will be in the H2D/D2H transfer timing (which NCU won't profile, but your cudaEventElapsedTime will show), and whether the kernel-side metrics change (they should not, since once data is on-device the access pattern is identical regardless of original host memory type).









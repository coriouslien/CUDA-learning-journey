<pre>
Grid: (5376, 1, 1)   Block: (768, 1, 1)
Total threads: 5376 × 768 = 4,128,768
Duration: 12.19 ms
SM Frequency: 2.29 GHz

cudaOccupancyMaxPotentialBlockSize chose 768 threads/block, capped to 32 * minGrid blocks.
</pre>
GPU Speed Of Light Throughput
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/cde1b0ba-09ba-4bab-aa17-dfcf74eda562" />
<pre>
177.92 GB/s achieved out of ~960 GB/s peak — roughly 18.5% of peak.

NCU flags this as a Latency Issue: both compute and memory are below 60% of peak, meaning the GPU is spending
most of its time stalled rather than doing useful work. This is the central finding for this kernel.
</pre>
________________________________________________________________________________________________________
Memory Workload Analysis & Scheduler Statistics
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/68b9d17c-bec0-4e55-bc44-55d7b027d831" />
<pre>
L1 hit rate of 0% means every load goes to L2 or DRAM. This makes sense — atomic_add_kernel_naive strides 
through a 268M element array with no locality and no shared memory staging. Each thread's load is essentially 
uncacheable relative to neighboring threads' loads.

This is the most damning number in the entire profile. The scheduler finds zero eligible warps to issue 99.46% 
of cycles. The GPU is essentially idle almost all the time — active warps exist but are all stalled 
simultaneously. The scheduler issues one instruction every 185.9 cycles instead of every cycle.
</pre>
________________________________________________________________________________________________________
Warps Per Scheduler
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/97f2e790-14d3-43cc-99fd-17d8eeae9de3" />

________________________________________________________________________________________________________
Warp State Statistics
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/38e9f272-07ae-4dba-9a74-bdc3a023d31b" />
Warp Cycles Per Issued Instruction: 1,630 cycles

That is catastrophic — ideally this is close to 1. Every issued instruction costs 1,630 cycles of waiting.

Dominant stall reasons:

Stall Type	Cycles	Est. Speedup if Fixed
Long Scoreboard	~955 cycles (58.6%)	58.62%
LG Throttle	~546 cycles (33.5%)	33.53%

Long Scoreboard means warps are stalled waiting for a global memory load to return data. The warp issued a
load, the data is not in L1 or L2, it goes all the way to DRAM, and the warp sits idle for ~955 cycles 
waiting. This is the direct consequence of 0% L1 hit rate.

LG Throttle means the L1 instruction queue for local/global memory operations is full. Threads are hammering
global memory so aggressively that the memory pipeline is backed up — new memory instructions cannot even be
issued because the queue is saturated. This is a secondary effect of the same root cause: too many in-flight
global memory requests.

Together these two stalls account for 92% of all stall cycles.
_______________________________________________________________________________________________________
Warp State (All Cycles)
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/7119932f-803a-4b03-966c-4ef1fcccfafe" />
Visually confirms Long Scoreboard dominates (~970 cycles), LG Throttle is second (~550 cycles), Drain and MIO
Throttle are minor. The bar chart makes the relative magnitude immediately obvious.
_______________________________________________________________________________________________________
Occupancy
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/a1c01214-293d-4a05-8204-9544f690f5b0" />
Block Limit Warps = 2 is the binding constraint — with 768 threads/block = 24 warps/block, and the SM
supporting 48 warps maximum, only 2 blocks fit per SM. This is what caps theoretical occupancy despite the
register limit saying 3 blocks could fit.

Achieved (73%) is lower than theoretical (100%) due to workload imbalance — the grid cap (32 * minGrid = 5376
blocks) means later waves have fewer blocks to schedule as SMs drain. NCU estimates a 27% speedup from fixing
occupancy, but this is misleading — the real bottleneck is the stalls, not occupancy itself. Higher occupancy
helps hide latency only if you have more independent memory requests to issue; here the memory system is
already throttled.
_______________________________________________________________________________________________________
Impact of Varying Register Count Per Thread
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/1ae43922-af63-43ec-9d4b-0f194aa103a3" />
Current is at 16 registers/thread — occupancy stays at 100% theoretical up to 40 registers, then drops 
sharply at 41+
Your kernel is register-efficient; this is not the constraint
_______________________________________________________________________________________________________
Impact of Varying Block Size
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/1050ef79-afb4-46b8-bcb5-861d5a98d2a9" />
Current block size 768 hits 100% theoretical occupancy (the dot)
Block sizes above 768 drop to ~50% then near 0% — correct choice by cudaOccupancyMaxPotentialBlockSize
_______________________________________________________________________________________________________
Impact of Varying Shared Memory Usage Per Block
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/d1e77d96-01c2-427d-9fde-51b9ee1eba14" />
This kernel uses no shared memory, so the dot is at 0 bytes = 100% occupancy
Occupancy drops immediately with any shared memory allocation — relevant context for comparing against 
atomic_add_kernel_shared_mem which does use shared memory
_______________________________________________________________________________________________________
Impact of Varying Block Barriers
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/a30176d3-9c21-4997-aa88-196addefffc1" />
Current kernel has no __syncthreads() so 0 barriers = 100% occupancy
Drops at 12+ barriers — not relevant here but shows the cost of synchronization
_______________________________________________________________________________________________________
Impact of Varying Cluster Size
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/c46f6bbf-6306-4434-87d1-663272043dcc" />

Branch Efficiency: 100%, Avg Divergent Branches: 0

The stride loop for (int i = tid; i < num_elements; i += stride) has no divergence — all threads in a warp
exit the loop at the same iteration. Clean.

Warp Stall Sampling:

Top stall location: parallel_reduction_ker... with 383,122 samples (34%) and 110,863 samples (10%)
These point directly to the global load line inside the for loop — the cuCrealf(d_in[i]) / cuCimagf(d_in[i])
loads.
_____________________________________________________________________________________________
Summary:
The naive kernel's performance is entirely limited by one design decision: every thread loads directly from
global memory with no caching, no shared memory staging, and two global atomicAdds at the end.

|Problem|	Evidence|	Root Cause|
|-------|---------|-----------|
|18.8% memory bandwidth|	SOL, DRAM throughput|	Not enough concurrent independent requests|
|0% L1 hit rate|	Memory Workload|	No data reuse, no shared memory|
|99.46% no-eligible cycles|	Scheduler Statistics|	All warps stalled on memory simultaneously|
|1,630 cycles per instruction	|Warp State	|Long scoreboard stalls dominate|
|LG Throttle 33.5%|	Warp State|	Global memory instruction queue saturated|

The two atomicAdd calls at the end also serialize across all 4M threads to a single output location, but the 
load-loop latency dominates so heavily that the atomic contention is secondary here.

The shared memory kernels that follow in your project address the L1/L2 miss problem directly — that is 
exactly what the NCU is telling you to do.






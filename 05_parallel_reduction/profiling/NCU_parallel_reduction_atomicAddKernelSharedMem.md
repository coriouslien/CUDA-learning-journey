<pre>
Grid: (524288, 1, 1)   Block: (512, 1, 1)
Total threads: 524288 × 512 = 268,435,456
Duration: 3.42 ms  (vs 12.19 ms naive — 3.6× faster)
SM Frequency: 2.29 GHz

The grid is dramatically larger — 524,288 blocks vs 5,376. This is because powerOfTwoBlockSize = 512, 
and there is no 32 * minGrid cap applied here, so gridSize = (268M + 511) / 512 = 524,288. Every element 
gets exactly one thread, no stride loop needed in the ideal case — though with this many blocks, waves of 
blocks execute sequentially on the SM pool.

</pre>

GPU Speed Of Light Throughput
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/95e37a32-de89-49cf-8999-1b9c56eface2" />
<pre>
NCU now flags High Memory Throughput instead of Latency Issue. This is a fundamentally different diagnosis — 
the kernel is no longer wasting cycles doing nothing, it is actually saturating memory. Memory is more 
utilized than compute, which is the correct regime for a reduction kernel. The bottleneck shifted from latency
stalls to bandwidth.

629 GB/s achieved out of ~960 GB/s peak — roughly 65.5% of peak, compared to 177 GB/s (18.5%) for the naive kernel.
</pre>
_________________________________________________________________________________________________________
Memory Workload Analysis & Scheduler Statistics
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/f1b49d47-019a-49a6-b967-82964ebb4854" />
<pre>
L1 hit rate is still 0% — this is expected and correct. The input array is 2 GB, far larger than L1 or L2 can
cache. Each element is read exactly once by one thread and never reused from global memory. The shared memory
kernel's improvement comes not from cache hits but from doing the reduction in shared memory so threads don't
issue redundant global atomics.

Mem Pipes Busy jumped from 0.75% to 41.30% — the memory pipeline is now being kept busy.

Scheduler Statistics:
This is the most dramatic improvement in the entire profile. The scheduler went from finding eligible warps
only 0.54% of cycles to 54.71% of cycles — a 100× improvement in scheduler utilization. The kernel is now
issuing an instruction every 1.8 cycles instead of every 185.9 cycles. Shared memory staging broke the
dependency chain that was stalling every warp in the naive kernel.
</pre>

_________________________________________________________________________________________________________
Warps Per Scheduler
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/dc8215d2-f361-4cb6-b064-bb8ed62f69e6" />

_________________________________________________________________________________________________________
Warp State Statistics
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/98fa56de-74ca-4a72-9af0-0ec344cf059b" />
<pre>
Warp Cycles Per Issued Instruction: 19.37 (vs 1,630 naive — 84× reduction)

This single number summarizes the entire improvement. Each issued instruction now costs 19 cycles of overhead
instead of 1,630.

Dominant stall — Barrier Stalls (31.32% Est. Speedup):

Each __syncthreads() in the tree reduction loop causes warps to stall waiting for sibling warps to reach the
barrier. With 512 threads/block = 16 warps/block, and log2(512) = 9 reduction steps, each block executes 9
barriers. Warps that finish their work early stall waiting for slower warps — this is barrier imbalance.

NCU's suggestion to split blocks ≥512 threads into smaller groups is worth noting — it would reduce the 
number of warps waiting at each barrier, but would also require more blocks and more final atomicAdds.

Long Scoreboard stalls still present (~4 cycles, reduced from ~955) — these come from the global load loop at
the start. The per-thread grid-stride loop still hits DRAM for the input load, but there are far fewer stall
cycles because the work is more evenly distributed and there is no LG Throttle.

LG Throttle: completely gone (was 33.5% of stalls in naive). The global memory instruction queue is no longer
saturated — only thread 0 per block issues atomicAdd, so the queue pressure dropped from N-threads to 
1-thread-per-block.
</pre>
________________________________________________________________________________________________________
Warp State (All Cycles)
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/3b1b907a-7a95-44ea-9d0c-0777f2170061" />
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/6caeb792-ecb2-4984-b645-aae8ec9a72e8" />
<pre>
The chart axis is now 0–10 cycles, compared to 0–1000 in the naive kernel — the scale alone tells the story.

Stall Barrier dominates at ~6.5 cycles — this is the __syncthreads() cost in the tree reduction. Expected and
largely unavoidable with this algorithm.

Stall Long Scoreboard ~4 cycles — global loads from d_in in the accumulation loop.

Stall Wait ~3 cycles — warp is ready to issue but another warp was selected instead. This is healthy 
scheduler competition, not a problem.

Stall LG Throttle: near zero — confirmed eliminated.

The profile is now dominated by synchronization overhead from a correct algorithmic design, not by memory
system saturation from poor access patterns.
</pre>
________________________________________________________________________________________________________
Occupancy
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/6e141bcf-459f-4b5d-92ea-c947f2ee5134" />

<pre>
Block Limit Warps went from 2 → 3. With 512 threads/block = 16 warps/block, 3 blocks × 16 warps = 48 warps 
per SM — exactly the SM maximum. The smaller block size is more warp-efficient.

Block Limit Shared Mem = 6 is now a real constraint. Each block uses 512 × sizeof(cuFloatComplex) = 4096 
bytes of shared memory. The RTX 5080 SM has ~100KB shared memory, so 100KB / 4KB ≈ 25 blocks could fit from 
a shared memory perspective alone — but the warp limit of 3 is the binding constraint here.

Achieved occupancy improved from 73% → 88.8% — the occupancy gap from theoretical is smaller, reflecting more
uniform workload distribution across the larger grid.

The estimated speedup from fixing occupancy dropped from 27% to 11.2% — the kernel is healthier, so occupancy
matters less as a lever.
</pre>
________________________________________________________________________________________________________
Impact of Varying Register Count Per Thread
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/a6b12702-0360-4e7a-acfd-7b592db58d97" />

<pre>
Current kernel uses ~20 registers/thread (dot at ~20). Occupancy stays at 100% up to 40 registers, then drops
sharply at 64+. The shared memory kernel uses more registers than naive due to the reduction loop variables s,
tx, r, im — still comfortably within the 100% occupancy zone.
</pre>
________________________________________________________________________________________________________
Impact of Varying Block Size
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/66fdc56f-2f52-42d7-9464-96172dd2569f" />
<pre>
Current block size 512 hits 100% theoretical occupancy (dot at 512). Note 512 is not the unique optimal — 
several other block sizes also hit 100%. The choice of power-of-two for the tree reduction is the real
constraint, not the occupancy optimization.
</pre>
________________________________________________________________________________________________________
Impact of Varying Shared Memory Usage Per Block

<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/1827facf-f1e6-4662-9db7-71520f4d3017" />
<pre>
The dot is at ~4,096 bytes (512 × 8 bytes per cuFloatComplex). Occupancy stays 100% up to ~6,368 bytes, then
drops to 75%, then 31%, then near zero. This confirms shared memory is a real occupancy limiter for this 
kernel — if you increase block size to 1024, shared memory doubles to 8KB and occupancy would drop noticeably.
</pre>

________________________________________________________________________________________________________
Impact of Varying Block Barriers

<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/e8139019-cdd6-477b-aae0-6d64695e3fde" />
<pre>
Current kernel has 1 barrier per reduction step × 9 steps = 9 barriers (dot at ~1 shown at leftmost).
Occupancy stays at 100% up to 9 barriers, drops to ~67% at 12+, then ~31% at 24+. This kernel sits right at
the safe edge of the barrier budget.
</pre>

________________________________________________________________________________________________________
Impact of Varying Cluster Size
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/0d7c7377-a019-40a7-9bbe-423a1ecbfdc6" />

<pre>
Branch Efficiency: 98.64% (vs 100% naive)

Avg Divergent Branches: 7,801.90 — this is significant and expected. The tree reduction if (tx < s) creates warp divergence in every reduction step. At each step, half the threads go inactive. With 9 steps and 16 warps/block, divergence accumulates. Branch efficiency is still 98.64% because the vast majority of executed instructions are in the non-divergent load loop.

Warp Stall Sampling: Top stall location has 112,669 samples (35%) and 52,802 samples (16%) — these point to
the __syncthreads() locations in the tree reduction loop, consistent with Barrier Stalls being the dominant
stall type.
</pre>

___________________________________________________________________________________________________
Summary:
The shared memory kernel's improvement is real and correctly diagnosed. The remaining bottleneck — Barrier
Stalls from __syncthreads() — is inherent to the tree reduction algorithm. The partial_store_shared kernel
eliminates the final atomicAdd contention, which should push throughput closer to peak.








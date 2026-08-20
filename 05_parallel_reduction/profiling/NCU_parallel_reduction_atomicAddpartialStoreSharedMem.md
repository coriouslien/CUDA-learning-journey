<pre>
Grid: (5376, 1, 1)   Block: (512, 1, 1)
Total threads: 5376 × 512 = 2,752,512
Duration: 2.39 ms
SM Frequency: 2.29 GHz

Same grid cap pattern as the naive kernel (32 * minGrid), but with 512 threads/block instead of
768. Each thread processes a stride-loop over 268M elements.
</pre>
GPU Speed Of Light Throughput
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/7429a879-7443-4dba-8741-1258bde2345c" />
<pre>
NCU now flags "High Throughput" — >80% of memory performance. This sounds impressive — 
899.95 GB/s achieved out of ~960 GB/s peak, roughly 93.7% — but this is a broken kernel 
producing wrong results. The high DRAM throughput is real but misleading: because the tree 
reduction is dead (tx < 0), every thread's accumulated local sum in registers is correct, 
but the final atomicAdd writes to d_partial (the input buffer) instead of d_out, corrupting 
the input data. The kernel is fast precisely because it skips the shared memory reduction
entirely and goes straight to the (wrong) atomic write.
</pre>

______________________________________________________________________________________________________
Memory Workload Analysis & Scheduler Statistics
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/ab289276-9db2-4a44-b86c-29435d6eeded" />
<pre>
L2 Hit Rate is essentially 0% — even lower than the SharedMem kernel's 1.56%. The input array 
is accessed in a pure streaming fashion with no reuse whatsoever.

Mem Pipes Busy is only 4.05% despite 95% DRAM throughput — this is a key contradiction. 
It means the memory pipe is not the bottleneck in the sense of instruction issue rate; rather,
the DRAM bus itself is nearly saturated from the sheer volume of in-flight requests even with 
few pipe-busy cycles. The kernel is issuing very few instructions but each one triggers 
enormous DRAM traffic.

No Eligible cycles jumped back to 96.93% — nearly as bad as the naive kernel's 99.46%. 
Despite having the most active warps per scheduler (11.91), almost none are eligible. 
This tells the warps are stalled, not doing work. The scheduler issues one instruction 
every 32.6 cycles — far worse than SharedMem's 1.8, though better than naive's 185.9.

The paradox: 95% DRAM throughput with 97% no-eligible cycles. Both are true simultaneously — 
the GPU is keeping DRAM busy via in-flight load requests from the stride loop, but the warps
issuing those loads immediately stall waiting for the data to return. The memory system is
saturated but the compute pipeline is almost idle.

</pre>


______________________________________________________________________________________________________
Warps Per Scheduler

<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/56f10ea0-71e2-4471-b2dc-9aee8d3f345d" />


______________________________________________________________________________________________________
Warp State Statistics
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/e12168ed-04af-403d-af65-1280c4b5fa95" />

______________________________________________________________________________________________________
Warp State (All Cycles)
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/0e7cbf49-4457-4e66-80c7-e9acf395aac6" />

Warp Cycles Per Issued Instruction: 387.86 (vs naive 1,630, SharedMem 19.37)

Significantly worse than SharedMem but much better than naive. This intermediate value makes sense: 
the stride loop generates massive DRAM traffic like the naive kernel, but the grid is much smaller 
(5,376 blocks vs naive's same grid but with 512 threads/block handling more elements per thread).

Dominant stall — Long Scoreboard: 90.5% of stall cycles (351 cycles)

This is the same root cause as the naive kernel: threads are stalled waiting for DRAM loads to return from 
the stride loop for (int i = tid; i < total_elements; i += stride). With only 5,376 blocks covering 268M 
elements, each thread must process 268M / 2,752,512 ≈ 97 elements — far more than the SharedMem kernel's 1 
element per thread. That means each thread issues ~97 global loads in sequence, each stalling for DRAM.


Stall Barrier: ~60 cycles — present but small. This is the __syncthreads() inside the tree reduction loop. 
Even though tx < 0 makes the reduction body dead code, the __syncthreads() calls themselves still execute and
still synchronize warps — they just synchronize over no-op iterations. This barrier cost is real overhead from
a reduction that does no work.

LG Throttle: near zero — the memory instruction queue is not saturated despite high DRAM usage. With only 
5,376 blocks and 512 threads/block, there are far fewer concurrent threads than the SharedMem kernel's 
524,288 blocks — so the queue doesn't fill up.

Warp State chart scale is 0–400 cycles — between the naive (0–1000) and SharedMem (0–10) scales, visually 
confirming this kernel's intermediate pathology.
______________________________________________________________________________________________________
Occupancy & Impact of Varying Register Count Per Thread
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/28bd7a06-22f0-4c5a-b1a0-085365f56f56" />

Achieved occupancy 98.71% — the highest of all three kernels, essentially at theoretical maximum. This is the
only genuinely good number in this profile, but again it reflects that the kernel is doing less work per warp
(the reduction is dead, so warps finish the load loop and immediately proceed to the broken atomicAdd), which
counterintuitively keeps occupancy high.

Block limits are identical to SharedMem: 5 register blocks, 6 shared memory blocks, 3 warp blocks — same 
512-thread configuration.
______________________________________________________________________________________________________
Impact of Varying Block Size
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/1edc29b1-f121-42b9-8f88-558f6df6ba71" />



______________________________________________________________________________________________________
Impact of Varying Shared Memory Usage Per Block

<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/e927faf2-885b-4bba-910d-3d752fe60cf2" />


______________________________________________________________________________________________________
Impact of Varying Block Barriers
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/c45cd9a2-c37c-4855-891e-243dad672745" />



______________________________________________________________________________________________________
Impact of Varying Cluster Size
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/4bb6b023-a190-469d-aeb7-ce5b8e3cd132" />


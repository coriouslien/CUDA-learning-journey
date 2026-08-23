<pre>
15.22 ms / 34.9M cycles on RTX 5080, Grid: (64, 64, 1), Block: (256, 1, 1) (8 warps/block).
</pre>
GPU Speed Of Light Throughput
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/caebdf8a-0e58-4e5f-98e4-d09d5a58a09f" />
<pre>
Memory: 88.20% / Compute: 36.68%, Kernel is heavily bounded by the memory subsystem, specifically 
L1/Shared Memory.
It is memory-bound, not compute-bound. Memory throughput at 88.20% and L1/TEX at 90.08% are both above the 80%
"High Throughput" warning threshold.
L1/TEX Cache Throughput: 90.08%, L1/TEX throughput is near maximum, driven entirely by shared memory traffic.
</pre>
_____________________________________________________________________________________________________
Floating Point Operations Roofline (Half Precision)
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/91de6d8a-3cc5-4255-bdd4-3b8f8fd16969" />
Compute SM sits at only 36.68%, which means the arithmetic units are starved — they have data problems, 
not instruction-count problems. The roofline shows this clearly: the kernel's achieved FLOPS/s plots as a 
flat horizontal line far below the half-precision tensor-core roof, meaning you are not limited by the 
number of operations you need to do, but by how slowly those operations are being fed.

_____________________________________________________________________________________________________
Compute Workload Analysis
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/c9d49c18-d369-4ae2-98da-40149ec26191" />

The good news: Tensor (FP) is the dominant pipe at 37.5% of active cycles, which confirms that wmma::mma_sync
is actually issuing to tensor cores and not silently falling back to CUDA cores. ALU and FMA are tiny slivers,
which is expected.

The bad news: 37.5% pipe utilization on the tensor core means it is sitting idle 62.5% of active cycles. 
The tensor core pipeline wants to be fed continuously; instead it is being starved by the shared memory 
stalls described below. The overall IPC of 0.90 is very low — roughly one instruction every 4.5 scheduler
cycles instead of one per cycle.
_____________________________________________________________________________________________________
Memory Workload Analysis
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/1227b465-30df-4a8a-9727-2ed36d5da0c6" />
<pre>
Severe Shared Memory Bank Conflicts (75.16% Estimated Speedup)

The Problem: The kernel executes 100.66M shared memory requests, but due to poor stride/access patterns, 
warps suffer an average 24.1-way bank conflict out of the 32 banks.

Impact: Every single shared memory load is serialized up to 24 times by the hardware scheduler, choking 
the L1/Shared Memory pipeline (pushing L1 throughput to 90.08%) while compute tensor cores sit idle 
(only 36.68% utilization).
</pre>
_____________________________________________________________________________________________________
Memory Chart
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/be49e134-2d60-480b-9a54-b05ad9461d1c" />
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/f74ff8a0-73bf-4249-96a1-fc944c3eba5b" />
The dominant traffic path is shared memory: 100.66 M instructions, 100.66 M requests — essentially all compute-side memory traffic is hitting shared memory. The L2 is healthy (86.1% hit rate), and DRAM is not the bottleneck (26.48% throughput). The memory chart in image 6 confirms the pipeline path: global → L1/TEX (low 0.76% hit) → L2 (de-compressed 25.77 GB) → device (267 MB). Very little global data needs to go to DRAM.

The critical finding is the bank conflict warning: 24.1-way average conflict, 83.43% of all shared-memory wavefronts conflicted. This means for every intended single-cycle shared load, the hardware is serializing it into an average of 24 sub-transactions. This is why MIO (memory I/O) is saturated and why Short Scoreboard stalls dominate warp state time. Your PAD=8 is not solving the problem because the WMMA intrinsic's internal thread-to-element mapping across a 16×16 fragment generates a strided access pattern that hits the same banks regardless of simple row padding.
____________________________________________________________________________________________________
Scheduler Statistics
<img width="1456" height="839" alt="image" src="https://github.com/user-attachments/assets/aea2f6c6-5eea-4c6d-9217-8be50aa8a8db" />
Warps Per Scheduler
<img width="1456" height="839" alt="image" src="https://github.com/user-attachments/assets/0059955c-1bcd-4181-96f9-cc56c60cb93f" />
With 3.98 active warps per scheduler but only 0.31 eligible warps per scheduler, 77.56% of all cycles have no eligible warp to issue. The scheduler is essentially idle three quarters of the time. This is the direct downstream effect of the bank conflicts: all 4 active warps are simultaneously stalled waiting for their shared memory requests to complete through the serialized conflict replays, leaving the scheduler with nothing to do. The issue slot utilization warning (11.80% estimated speedup) represents the scheduling efficiency loss on top of the actual stall time — even when a warp becomes eligible, there often aren't enough of them to hide the latency.

_____________________________________________________________________________________________________
Warp State Statistics
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/72a77cc4-2581-46f8-b035-2f87f811ab87" />
Warp State (All Cycles)
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/5a8101ce-f5dc-4e53-98d8-220609f21d2e" />

The warp state breakdown tells you exactly where cycles are being spent. Short Scoreboard at 5.8 cycles (32.9% of total) is the dominant stall and is entirely the bank conflict manifesting as MIO backpressure — each conflicted load holds a scoreboard entry for 24× longer than it should. Stall Wait (~2.5 cy) is warps that are ready but not selected by the scheduler — with so few eligible warps this barely matters. Math Pipe Throttle (~2.4 cy) is the tensor core pipeline being full or backed up, which is a secondary consequence of infrequent data delivery. Stall Barrier (~2.0 cy) is the cost of __syncthreads() inside your double-buffering loop — unavoidable, but inflated because the conflicted loads extend the window between barrier issue and barrier clear. Long Scoreboard (~1.3 cy) represents global memory latency — relatively small, confirming DRAM is not the primary issue.

_____________________________________________________________________________________________________
Occupancy
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/c10e6a9e-bf50-4338-80e9-1bf12fe05b67" />


_____________________________________________________________________________________________________
Impact of Varying Register Count Per Thread
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/deeab33d-ff12-4951-9a9f-691843f6f474" />


_____________________________________________________________________________________________________
Impact of Varying Block Size 
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/e2b546b6-7fd1-41bc-84d7-1ee98a89b567" />


_____________________________________________________________________________________________________
Impact of Varying Shared Memory Usage Per Block
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/f19824fe-8664-433a-8381-0d5676668b79" />


_____________________________________________________________________________________________________
Impact of Varying Block Barriers & Impact of Varying Cluster Size
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/06c28e12-d677-4a2a-818f-a2f4eb643d73" />


_____________________________________________________________________________________________________
GPU and Memory Workload Distribution
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/871da54c-af40-4f15-83bb-3c5464b43e9f" />


_____________________________________________________________________________________________________
<pre>
There is a major memory pipeline bottleneck caused by severe shared memory bank conflicts and low occupancy.
  
</pre>

_____________________________________________________________________________________________________


_____________________________________________________________________________________________________


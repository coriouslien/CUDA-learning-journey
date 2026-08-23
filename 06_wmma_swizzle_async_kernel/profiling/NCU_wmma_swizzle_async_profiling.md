<pre>
The profiled kernel (wmma_swizzle_... on an NVIDIA RTX 5080) is heavily bottlenecked by shared memory bank
conflicts and uncoalesced shared memory accesses, rather than compute capacity or DRAM bandwidth.

Although L1/TEX throughput is near maximum at 96.60%, the SM Compute throughput is only 15.36%. Schedulers
spend 90.13% of their cycles with zero eligible warps to issue because warps are stuck waiting on serialized
shared memory operations.
  
</pre>

GPU Speed Of Light Through
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/d2841edd-faca-4bb2-baee-333559a23deb" />
<pre>
The headline numbers:

Compute (SM): 15.36% — Tensor Cores are almost idle. Arithmetic and Tensor Core pipelines are heavily 
underutilized.
Memory: 95.93% — completely memory-bound
L1/TEX: 96.60% — L1 is saturated. The L1 cache/shared memory data pipe is running at near maximum capacity.
L2: 22.60%, DRAM: 22.24% — both low. Off-chip memory bandwidth is not the bottleneck. The issue is strictly localized inside the SM's shared memory / L1 data path.

The critical insight here: L1 is at 96.60% but DRAM is only 22.24%. Traffic is not reaching DRAM — 
it is being recycled inside L1/TEX. This is the bank conflict signature: the same data is being 
re-fetched repeatedly from shared memory due to conflicts, hammering L1 without producing useful compute work.
</pre>

_________________________________________________________________________________________________
Memory Workload Analysis & Scheduler Statistics
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/107bfd5d-8723-490d-ae9e-fd871fec31e5" />
<pre>
Memory numbers:

Memory Throughput: 210.50 GB/s
L1/TEX Hit Rate: 0.56% — catastrophically low for global loads
L2 Hit Rate: 78.43%
Mem Busy: 95.93%, Max Bandwidth: 22.47%, Mem Pipes Busy: 14.44%

The 0.56% L1 hit rate on global loads tells it cp.async / LDGSTS is bypassing L1 as intended (.BYPASS flag),
going straight to L2/DRAM. That is correct behavior. The memory pressure is entirely on the shared memory 
read side, not global loads.

Scheduler statistics:

Active Warps Per Scheduler: 5.93 (out of theoretical 6)
Eligible Warps Per Scheduler: 0.15 — nearly zero
No Eligible: 90.13% — 90% of cycles the scheduler has nothing to issue. Schedulers issue an instruction 
only once every ~10 cycles because warps are perpetually stalled waiting on data.
Issue Slot Utilization: one instruction every 10.1 cycles

Active warps are present but almost none are eligible. They are all stalled. The double-buffering is not 
hiding the latency — warps are waiting on smem reads that are backed up due to conflicts.
</pre>
_________________________________________________________________________________________________
Warps Per Scheduler
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/ea5e3ac1-0202-4765-ba18-8a5f16045c0e" />
<pre>
Visually confirms image 2: Active warps ≈ Theoretical warps (both around 6), but Eligible and Issued warps are
near zero — a flat sliver on the chart. It has warps, but they cannot issue because they are all stalled on
the same bottleneck.
  
</pre>
_________________________________________________________________________________________________
Warp State Statistics
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/88ee5349-a84e-40fb-9ebf-01f0efb5929d" />
<pre>
Warp Cycles Per Issued Instruction: 60.07 cycles
Short Scoreboard Stalls(~48.8% of latency / 29.3 cycles): dominant — 29.3 cycles average, representing 48.8%
of all 60.1 cycles between instructions. The single largest stall reason. Short scoreboard stalls occur when
instructions depend on pending shared memory accesses (LDS/STS) or special function unit operations.
NCU message: "primary reason is memory operations to shared memory... reduce bank conflicts"

This is the direct confirmation. Short Scoreboard stalls in the context of smem = warps are waiting for LDSM
(shared memory load for WMMA) to complete. The LDSM.16.M88.4 instructions are stalling because bank conflicts
are serializing the accesses. Each conflict multiplies the cycles needed, and since 60 cycles separate
instructions on average, the pipeline is essentially stalled almost the entire time.


</pre>
_________________________________________________________________________________________________
Warp State (All Cyclyes)
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/da862672-394d-4017-95d2-c67736bc7267" />
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/07ee4d3f-5f6a-4f94-9876-f7206a920431" />

<pre>
 Stall breakdown in order of magnitude:

Stall Short Scoreboard — dominant, extends past the 20-cycle mark, close to 30
Stall MIO Throttle — roughly half of Short Scoreboard. The second largest stall contributor, indicating 
the internal Memory Input/Output instruction queue is completely backed up.
Stall Barrier — moderate, from __syncthreads() in the pipeline
Stall Long Scoreboard — smaller, from global memory latency
Stall Wait, Math Pipe Throttle, Selected — small

Short Scoreboard + MIO Throttle together account for the majority of all stall cycles. MIO Throttle means the
MIO pipeline queue (which handles smem and special instructions) is full — the bank conflicts are flooding 
the MIO queue faster than it can drain, so new instructions cannot even enter the queue. 
The remaining stall reasons (Math Pipe Throttle, Selected, Not Selected, Branch Resolving, Dispatch, No
Instruction, Drain, LG Throttle, Misc) are all negligible slivers. This confirms the bottleneck is entirely
Short Scoreboard + MIO Throttle. Math Pipe Throttle being tiny is consistent with 15% compute SOL — the Tensor
Cores are barely being fed.
</pre>
_________________________________________________________________________________________________
Occupancy
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/0288fa5f-6d60-4310-8c13-3c7962bebe1f" />
<pre>
Theoretical Occupancy: 50% — 24 warps per SM. Limited by Shared Memory allocation per block (~34 KB per 
block, allowing at most 3 blocks of 256 threads per SM). 
Achieved Occupancy: 49.39% — matches theoretical closely, so the kernel is launching correctly. Matches 
theoretical occupancy well; register allocation (~52 registers/thread) is not currently an occupancy limiter.
Block Limit Shared Memory: 3 blocks — this is the binding limiter
Block Limit Registers: 4 blocks
Block Limit Warps: 6 blocks

The binding occupancy limiter is shared memory, not registers. The a_smem[2][BLOCK_M * BLOCK_K] + b_smem[2]
[BLOCK_K * BLOCK_N] = 2×(64×64×2) + 2×(64×64×2) = 32KB per block. SM120 has ~100KB smem per SM, so only 3 
blocks fit → 3×8 warps = 24 warps → 50% occupancy. This is the hard ceiling from the double-buffer smem
allocation.

NCU estimate: fixing occupancy gives 4.07% speedup — meaning occupancy is not the problem. Even if it got 
to 100% occupancy it wouldn't matter much. The bottleneck is the bank conflicts.
</pre>
_________________________________________________________________________________________________
Impact of Varying Register Count Per Thread & Impact of Varying Block Size
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/d318a8e7-b40f-413b-a45b-c43e1500f51f" />
<pre>
Current operating point (dot) is at ~52 registers, 50% occupancy. The curve stays flat at 50% until ~80
registers, then drops. It has headroom to use more registers without losing occupancy. This is relevant
context: the compiler is not over-pressuring registers here.

Block size curve shows 50% at block size 256 (the current config: 256 threads = 8 warps). Larger block sizes
can reach ~75% occupancy, but that would require restructuring the warp tiling.

Block size graph: the current 256-thread config hits 50% occupancy. Block sizes around 384–512 threads
achieve ~75% — but those configs would require it to change how many warps handle the tile, and the smem 
per block would change too.

</pre>
_________________________________________________________________________________________________
Impact of Varying Shared Memory Usage Per Block & Impact of Varying Block Barriers
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/b25f587e-1c0b-4b1a-910b-4b984031464b" />

<pre>
Shared memory graph: current usage is ~32KB (shown by the dot at ~32KB on the x-axis). At that point 
occupancy drops from the ~67% flat region to ~50%. If it could reduce smem usage below ~25KB, occupancy 
would jump. That's not practical with double-buffering at BLOCK_M=BLOCK_N=BLOCK_K=64 unless it reduces 
tile size.

The dot is at ~32KB smem, 50% occupancy. The graph shows the cliff is around 25KB → if it halved the smem
(e.g., single buffer instead of double, or smaller tiles) it would get to 67% occupancy. But the double-buffer 
is necessary for latency hiding, so this is a genuine tension — reducing smem to gain occupancy would
eliminate the pipeline.

Current barrier count keeps occupancy at 50%. Adding more barriers (from more __syncthreads() calls) 
crushes occupancy rapidly. At 9+ barriers occupancy drops to ~33%. This tells it that the existing
__syncthreads() pattern (2 per pipeline iteration: one after wait, the implicit one from pipeline) is 
already at a reasonable count. Don't add more sync points.
</pre>
_________________________________________________________________________________________________
Impact of Varying Cluster Size
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/6d1dfdc1-4c99-4a57-8aaa-1238c1149c08" />
<pre>
Uncoalesced Shared Accesses: 1,409,286,144 excessive wavefronts
That is 84% of the total 1,677,721,600 wavefronts
NCU estimated speedup from fixing this: 83.42%

This is the definitive answer. 84% of all shared memory wavefronts are excess replays from bank conflicts.
Only 16% are doing real work. NCU is telling it that fixing smem access patterns would give an 83% speedup —
roughly a 6× improvement on this kernel alone.

This is exactly the unfixed swizzle/read mismatch: it XOR-swizzle on the write path but 
wmma::load_matrix_sync generates linear LDSM reads, so every LDSM instruction hits conflicts because the 
data layout in smem doesn't match what LDSM expects.
</pre>
________________________________________________________________________________________________
Summary:
|Metric|	Value|	Implication|
|------|-------|-------------|
|Compute SOL|	15.36%|	Tensor Cores starved|
|L1/TEX SOL|	96.60%|	Smem conflicts flooding L1|
|Short Scoreboard stalls|	48.8%| of cycles	LDSM waiting on conflicted smem|
|MIO Throttle|	2nd largest stall|	MIO queue saturated by conflict replays|
|Uncoalesced smem wavefronts|	84% excess|	The direct conflict count|
|NCU estimated speedup|	83.42%|	Fix the smem layout|

<pre>
The profiler reports 1.41 billion excessive wavefronts (84% of total 1.68B wavefronts) due to uncoalesced
shared memory accesses.

A single shared memory load/store instruction that should execute in 1 wavefront is taking multiple 
serialized replay phases due to severe bank conflicts, saturating the L1 pipeline and causing the 
estimated 83.42% performance loss.
</pre>





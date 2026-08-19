GPU Speed Of Light Throughput
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/a29fe78c-ed3c-4864-b775-1bda5bb7dd0b" />
<pre>
The kernel is 0.02 ms faster and hits 87.18% DRAM SOL vs 86.92%. The delta is real but small — 
about 0.26 percentage points, well within run-to-run variance territory. The critical observation is: 
the kernel-side metrics are essentially identical. Both profiles are the same bandwidth_coalesced kernel 
executing on data that already lives in device memory. Pinned vs. pageable only affects the H2D/D2H transfer
path; once the data is resident on the GPU, it makes no difference to the kernel.

The actual benefit of pinned memory shows up in your cudaEventElapsedTime output for H2D transfers — 
which NCU doesn't profile. That's where you'd see the real gap.
</pre>
|Metric|	Pinned	|Pageable	|Delta|
|------|----------|---------|-----|
|Duration	|2.56 ms	|2.58 ms	|−0.02 ms|
|Elapsed Cycles|	5,870,526|	5,929,855|	−59,329|
|SM Active Cycles	|5,840,120	|5,829,072	|+11,048|
|Compute (SM) Throughput|	17.04%|	16.87%|	+0.17%|
|Memory Throughput|	87.18%|	86.92%|	+0.26%|
|DRAM Throughput	|87.18%	|86.92%	|+0.26%|
|L1/TEX Cache Throughput|	15.49%	|15.34%|	+0.15%|
|L2 Cache Throughput|	23.13%	|22.92%	|+0.21%|
|SM Frequency	|2.29 GHz|	2.29 GHz|	identical|
|DRAM Frequency	|14.79 GHz|	14.79 GHz	|identical|


____________________________________________________________________________________________________
Memory Workload Analysis & Scheduler Statistics
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/7b3a5b2f-1212-4e62-ae4e-be329f0849d3" />
<pre>

825 GB/s vs. 822 GB/s — a 2.5 GB/s difference. On a ~960 GB/s theoretical peak device this is 0.26%. 
This is measurement noise, not a meaningful kernel difference.

The L2 hit rate going from 0.00% to 0.01% is also noise — one or two extra L2 hits across 2²⁸ accesses is
statistically zero.

The scheduler numbers are bit-for-bit identical. Active warps, eligible warps, issued warps, no-eligible percentage — all match pageable to two decimal places. This is the definitive confirmation: pinned memory has zero effect on kernel execution. The scheduler does not know or care how the host memory backing was allocated.

  Metric	Pinned	Pageable	Delta
Memory Throughput	825.02 GB/s	822.54 GB/s	+2.48 GB/s
L1/TEX Hit Rate	0%	0%	none
L2 Hit Rate	0.01%	0.00%	negligible
Mem Busy	19.90%	19.68%	+0.22%
Max Bandwidth	87.18%	86.92%	+0.26%
Mem Pipes Busy	17.04%	16.87%	+0.17%
Active Warps/Scheduler	9.51	9.51	identical
Eligible Warps/Scheduler	0.08	0.08	identical
Issued Warp/Scheduler	0.07	0.07	identical
No Eligible [%]	92.75%	92.74%	+0.01%
One or More Eligible [%]	7.25%	7.26%	−0.01%
</pre>

____________________________________________________________________________________________________
Warp Per Scheduler
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/5622907e-72fb-4d41-83b1-f30c9e9141a8" />
<pre>
Visually identical to pageable. Active ~9.5, Eligible and Issued both near zero. Same enormous gap between active and eligible — same DRAM-stall signature.


</pre>

____________________________________________________________________________________________________
Warp State Statistics
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/45eb9fe2-c2bb-4776-8f83-88f614d447f1" />

<pre>

131.17 vs. 130.98 cycles per issued instruction — a 0.19-cycle difference at 2.29 GHz is 0.08 ns. Noise.

Long Scoreboard is identical at 120.2 cycles. The DRAM round-trip latency is a hardware constant; it doesn't care what memory type was on the host before the H2D transfer.

The Est. Speedup dropping from 13.08% to 12.82% is interesting — pinned is marginally closer to optimal in NCU's model, consistent with the slightly higher throughput.


Metric	Pinned	Pageable	Delta
Warp Cycles Per Issued Instruction	131.17	130.98	+0.19
Avg. Active Threads Per Warp	32	32	identical
Avg. Not Predicated Off Threads	30.12	30.12	identical
Long Scoreboard Stalls	120.2 cycles	120.2 cycles	identical
Long Scoreboard % of total	91.6%	91.8%	−0.2%
Est. Speedup	12.82%	13.08%	−0.26%
</pre>
____________________________________________________________________________________________________
Warp State (All Cycles)
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/a880b968-d5c0-4051-863f-c469ea8fc05d" />

<pre>

Visually indistinguishable from pageable. Long Scoreboard bar at ~120 cycles, Short Scoreboard, Wait, Drain all identically small.
</pre>
_____________________________________________________________________________________________________
Occupancy
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/006352ce-d170-4087-a6fc-e95875478de6" />
<pre>
Achieved occupancy is 80.23% vs 80.40% — a 0.17% difference. This is within the tail-effect variance from block wave scheduling. All the hard limits (registers, shared mem, warps, SM) are identical, as expected — these are compile-time properties of the kernel PTX, completely independent of host memory allocation.

  Metric	Pinned	Pageable	Delta
Theoretical Occupancy	100%	100%	identical
Achieved Occupancy	80.23%	80.40%	−0.17%
Achieved Active Warps/SM	38.51	38.59	−0.08
Block Limit Registers	16	16	identical
Block Limit Shared Mem	16	16	identical
Block Limit Warps	6	6	identical
Block Limit SM	24	24	identical
Est. Speedup	12.82%	13.08%	−0.26%
  
</pre>
_____________________________________________________________________________________________________
Impact of Varying Register Count Per Thread
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/1bcad814-b4f8-46cf-9f36-ad64ddc6f469" />



_____________________________________________________________________________________________________
Impact of Varying Block Size
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/40f29101-e04b-4797-b797-8f598101076e" />

_____________________________________________________________________________________________________
Impact of Varying Shared Memory Usage Per Block
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/575736ec-da24-4a8a-b670-c3bb0fa2dde2" />


_____________________________________________________________________________________________________
Impact of Varying Block Barriers
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/75ac7712-e760-42f5-b510-5c883c391b53" />

<pre>
All four charts (Register Count, Block Size, Shared Memory, Block Barriers) are pixel-for-pixel identical to the pageable versions. This is mathematically guaranteed — these are analytical curves computed from the kernel's resource usage, which is a compile-time constant. They will never differ between pageable and pinned runs of the same kernel.
  
</pre>
_____________________________________________________________________________________________________
Impact of Varying Cluster Size
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/05adc04e-0c67-4ccd-93e7-638701d204a2" />

<pre>
Identical. Same instruction count, same branch behavior. The Warp Stall Sampling table would again show 92%+ samples concentrated at the global load line.

  Metric	Pinned	Pageable
Branch Instructions	16,777,216	16,777,216
Branch Instructions Ratio	0.12%	0.12%
Branch Efficiency	0	0
Avg. Divergent Branches	0	0

  
</pre>

<pre>
Summary:
What Pinned Memory Actually Does

Here is the precise mental model, which your data proves empirically:

Pinned memory (cudaMallocHost) affects only the PCIe transfer path. When you call cudaMemcpy(d_in, h_in, size, H2D):

With pageable host memory: the CUDA runtime must first shadow-copy the data to a pinned staging buffer (OS-managed, invisible to you), then DMA from that buffer to the device. This adds latency and reduces effective H2D bandwidth — you're paying for an extra memcpy on the CPU side.
With pinned host memory: the DMA engine can access the host buffer directly. No staging copy. Maximum PCIe bandwidth is achievable.

Once cudaMemcpy returns and the data is in device DRAM, the GPU has no record of how it got there. The DRAM cells holding d_in[tid] are identical bits regardless of whether they came from pinned or pageable host memory. Your NCU data makes this structurally visible — every single kernel-side metric is statistically identical.

Where to look for the real difference: your cudaEventElapsedTime printout for H2D. On an RTX 5080 connected via PCIe 5.0 x16 you should see something like:

Pageable H2D: ~10–14 GB/s effective (throttled by staging copy)
Pinned H2D: ~25–50+ GB/s effective (direct DMA, near PCIe 5.0 bandwidth)

That's a 2–4× difference on the transfer, completely invisible to NCU's kernel profiler.

Complete Delta Table: Everything That Matters
What changed	Pageable	Pinned	Verdict
Kernel duration	2.58 ms	2.56 ms	Noise (0.8%)
DRAM throughput	822.54 GB/s	825.02 GB/s	Noise (0.3%)
DRAM SOL %	86.92%	87.18%	Noise (0.3%)
Long Scoreboard stall	120.2 cycles	120.2 cycles	Identical
Scheduler eligible warps	0.08	0.08	Identical
Achieved occupancy	80.40%	80.23%	Noise (0.2%)
All sensitivity charts	—	—	Identical
H2D transfer bandwidth	your printout	your printout	Large difference

The NCU kernel profile tells you the same story both times: you have a DRAM-bandwidth-saturated, Long-Scoreboard-dominated streaming kernel achieving ~87% of hardware peak, and pinned vs. pageable host allocation is orthogonal to all of it.

  
</pre>




GPU Speed Of Light Throughput
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/a29fe78c-ed3c-4864-b775-1bda5bb7dd0b" />
<pre>
The kernel is 0.02 ms faster and hits 87.18% DRAM SOL vs 86.92%. The delta is real but small — about 0.26 percentage points, well within run-to-run variance territory. The critical observation is: the kernel-side metrics are essentially identical. Both profiles are the same bandwidth_coalesced kernel executing on data that already lives in device memory. Pinned vs. pageable only affects the H2D/D2H transfer path; once the data is resident on the GPU, it makes no difference to the kernel.

The actual benefit of pinned memory shows up in your cudaEventElapsedTime output for H2D transfers — which NCU doesn't profile. That's where you'd see the real gap.

Metric	Pinned	Pageable	Delta
Duration	2.56 ms	2.58 ms	−0.02 ms
Elapsed Cycles	5,870,526	5,929,855	−59,329
SM Active Cycles	5,840,120	5,829,072	+11,048
Compute (SM) Throughput	17.04%	16.87%	+0.17%
Memory Throughput	87.18%	86.92%	+0.26%
DRAM Throughput	87.18%	86.92%	+0.26%
L1/TEX Cache Throughput	15.49%	15.34%	+0.15%
L2 Cache Throughput	23.13%	22.92%	+0.21%
SM Frequency	2.29 GHz	2.29 GHz	identical
DRAM Frequency	14.79 GHz	14.79 GHz	identical
</pre>

____________________________________________________________________________________________________
Memory Workload Analysis & Scheduler Statistics
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/7b3a5b2f-1212-4e62-ae4e-be329f0849d3" />
<pre>

825 GB/s vs. 822 GB/s — a 2.5 GB/s difference. On a ~960 GB/s theoretical peak device this is 0.26%. This is measurement noise, not a meaningful kernel difference.

The L2 hit rate going from 0.00% to 0.01% is also noise — one or two extra L2 hits across 2²⁸ accesses is statistically zero.

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


_____________________________________________________________________________________________________
Impact of Varying Cluster Size
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/05adc04e-0c67-4ccd-93e7-638701d204a2" />








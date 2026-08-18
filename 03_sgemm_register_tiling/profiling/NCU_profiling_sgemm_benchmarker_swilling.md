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
This is the critical clue. L1/TEX hit rate collapsed from 87.69% to 1.52%. This is not a minor degradation — 
it's essentially total L1 cache invalidation. Memory throughput jumped from 2.94 GB/s to 109.94 GB/s because 
now nearly every access misses L1 and goes to L2 or DRAM.

The swizzle transformation broke the spatial locality that your float4 vectorized global loads were 
exploiting. A swizzle that reorders how data is laid out in shared memory — or how threads index into it — can 
inadvertently destroy the access pattern that the L1 prefetcher or coalescing logic depended on for the global 
loads, depending on how the swizzle was applied. The L2 hit rate at 77.99% suggests most misses are served 
from L2 rather than DRAM, but L2 latency (~200 cycles) vs L1 (~32 cycles) is still devastating.
______________________________________________________________________________________________________
Warps Per Scheduler
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/cc124c3a-f4a1-4cec-9b10-23c324b68c17" />

______________________________________________________________________________________________________
Warp State Statistics
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/5cde74c8-19aa-4185-aee0-8f49f89b21a6" />

______________________________________________________________________________________________________
Warp State
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/65d9a3e9-f36e-4dfb-871a-e34a77e9f3ec" />

_______________________________________________________________________________________________________
Occupancy
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/1a1696e5-a668-4227-a8a4-2e4320378202" />

________________________________________________________________________________________________________
Impact of Varying Register Court Per Thread
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/cb5695b4-0ffb-4e91-b7af-941f2f9e3bf2" />

_______________________________________________________________________________________________________
Impact of Varying Block Size
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/592d826e-28b1-4d4f-8b1b-f1da930922af" />

_______________________________________________________________________________________________________
Impact of Varying Shared Memory Usage Per Block
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/704ad065-4b8e-4618-8118-1f37109d651f" />

_______________________________________________________________________________________________________
Impact Of Varying Block Barriers
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/befce936-088b-4541-b975-9f5042f3a1dc" />

_______________________________________________________________________________________________________
Impact of Varying Cluster Size
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/4f741e7f-e408-4c31-845f-a3d3f22e7291" />







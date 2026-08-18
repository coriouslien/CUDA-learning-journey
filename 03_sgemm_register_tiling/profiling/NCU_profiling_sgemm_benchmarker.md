
**Kernel: 568 - sgemm_registe..., grid (32,32,1), block (16,16,1) = 256 threads/block. RTX 5080, 5.69ms, 13,040,292 elapsed cycles. This is the baseline register-tiled SGEMM**
03_sgemm_register_tiling
GPU Speed Of Light Throughput
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/8a9a6097-3825-43c3-addd-20af583adbe7" />
The bottleneck is L1/TEX at 77%, not DRAM (0.31%) or L2 (1.44%). NCU flags "High Memory Throughput: memory
more heavily utilized than compute." The near-zero L2/DRAM numbers confirm data is largely recycled within L1
— this is the shared memory / register file pressure picture, not a bandwidth wall against HBM. 
The 58% compute SOL tells you the SMs are not fully pipelined.

Memory Workload Analysis & Scheduler Statics
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/41c4aaab-dc9e-4445-bb18-7428f710ad98" />
L1 hit rate of 87.69% and L2 of 91.93% are both very high — almost nothing reaches DRAM. This is consistent 
with your memory note: you achieved this after applying float4 vectorized global loads. The 2.94 GB/s 
effective memory throughput is extremely low for an 8192³ problem, confirming essentially everything is served
from cache/shared memory, but the Mem Busy at 72.94% with only Max Bandwidth at 40.27% reveals the memory unit
is being kept busy not by large transfers but by high-frequency small accesses — i.e., shared memory bank 
conflict replays inflating transaction counts without moving useful bytes.

The Scheduler Statistics show Active Warps Per Scheduler: 3.87, Eligible Warps: 1.85, Issued Warp Per 
Scheduler: 0.61, and No Eligible: 38.71%. The 38.71% no-eligible rate is a hard signal that warps are 
regularly stalled and the scheduler has nothing to issue — this is the latency hiding failure caused by low 
occupancy (only ~4 warps/scheduler vs. the 12 maximum).


Warp Per Scheduler
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/f58fedb5-4e7d-48cf-b7c6-410868585cbb" />

Warp State Ststistics
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/7dfba5b8-a5a7-4d9f-8684-82690a9cc69a" />

Warp State
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/c0dae752-18ae-46b3-b721-a6414e1b0465" />

Warp State
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/2f31b5a2-a2be-4084-80fe-dc1cb0213a11" />

Occupancy
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/290cdd34-a3a1-4e09-a5a2-62ccdac5a093" />

<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/f677e2bb-3a65-4762-b2d8-979bf389267b" />

<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/4dfb8164-b283-4cfe-b165-e9977fffc77e" />

<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/2a381f38-1ee7-4bfe-8f8f-0f0b28b24871" />

<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/e8174cbd-7f66-4a5a-87d7-f1e6400dafcc" />

<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/ef240166-bd8c-4639-92c6-0dd41e3f9209" />


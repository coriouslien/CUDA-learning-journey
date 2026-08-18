
**Kernel: 568 - sgemm_registe..., grid (32,32,1), block (16,16,1) = 256 threads/block. RTX 5080, 5.69ms, 13,040,292 elapsed cycles. This is the baseline register-tiled SGEMM**<br>
03_sgemm_register_tiling
GPU Speed Of Light Throughput
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/8a9a6097-3825-43c3-addd-20af583adbe7" />
The bottleneck is L1/TEX at 77%, not DRAM (0.31%) or L2 (1.44%). NCU flags "High Memory Throughput: memory
more heavily utilized than compute." The near-zero L2/DRAM numbers confirm data is largely recycled within L1
— this is the shared memory / register file pressure picture, not a bandwidth wall against HBM. 
The 58% compute SOL tells you the SMs are not fully pipelined.
___________________________________________________________________________________________________
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

_____________________________________________________________________________________________________
Warp Per Scheduler
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/f58fedb5-4e7d-48cf-b7c6-410868585cbb" />
GPU Maximum Warps Per Scheduler: ~12. Theoretical: ~4. Active: ~3.87. Eligible: ~1.85. Issued: 0.61.

The issued/active ratio of 0.61/3.87 ≈ 15.8% means on most cycles, the scheduler has active warps but none are
ready to issue. This is the classic register-pressure / low-occupancy trap: you have 4 warps but most are
stalled, so the scheduler sits idle ~38% of cycles. The gap between Eligible (1.85) and Issued (0.61) further
tells you that even when warps are eligible, the issue rate is far below 1 instruction/cycle — there are 
structural stalls (dispatch, scoreboard, barriers) preventing back-to-back issuance.

______________________________________________________________________________________________________
Warp State Ststistics
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/7dfba5b8-a5a7-4d9f-8684-82690a9cc69a" />
Warp Cycles Per Issued Instruction: 6.31. All threads active (Avg Active Threads Per Warp = 32, Not Predicated Off = 32 — no divergence).

The dominant flagged stall is Stall Not Selected (Est. Local Speedup: 32.07%) at ~2.0 cycles/instruction.
NCU's interpretation here is nuanced: "Not Selected" means the warp was eligible but the scheduler picked a
different warp. High Not-Selected stalls typically indicate you actually have enough warps to hide latency but
are accumulating priority/scheduling overhead — it suggests reducing active warps to improve cache coherence.
However, this must be read in context: with only ~4 warps/scheduler, "Not Selected" isn't evidence of healthy
occupancy; it's evidence that the few warps you have are competing for the issue slot against each other while
all simultaneously stalled on the same resources (shared memory replay, barriers).
_________________________________________________________________________________________________
Warp State
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/c0dae752-18ae-46b3-b721-a6414e1b0465" />
Warp State
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/2f31b5a2-a2be-4084-80fe-dc1cb0213a11" />
The stall breakdown in rough visual order:

Stall Not Selected (~2.0 cycles) — largest bar
Selected (~1.3 cycles) — actual issued cycles
Stall Dispatch Stall (~1.2 cycles) — warp ready but dispatch unit busy; structural pipeline pressure
Stall Short Scoreboard (~0.8 cycles) — waiting for a register written by a recent instruction 
(typically ~20-cycle latency ops like shared memory loads or integer arithmetic)
Stall MIO Throttle (~0.75 cycles) — MIO queue (the memory instruction issue pipeline) is saturated; this is
the bank conflict symptom
Stall Barrier (~0.7 cycles) — __syncthreads() induced wait
Stall Long Scoreboard (~0.6 cycles) — waiting for a long-latency result (global memory, though very minor here
given L1 hit rate)
Stall Wait / No Instruction / Math Pipe Throttle — minor

The co-presence of MIO Throttle + Short Scoreboard + Barrier is the canonical register-tiled shared-memory
GEMM stall signature: bank conflicts inflate MIO queue depth → Short Scoreboard waits for the replayed load 
result → Barrier stalls accumulate because all warps converge at __syncthreads() before the next tile.

Your prior analysis in memory already identified the 5-way bank conflicts (271M conflicts) and the catch-22 
between tile size, registers, and occupancy. These stall bars confirm it quantitatively.
_________________________________________________________________________________________________________
Occupancy
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/290cdd34-a3a1-4e09-a5a2-62ccdac5a093" />
Register pressure is the sole occupancy limiter: 2 blocks/SM. Block limit from registers is 2, everything else is far higher (7 for smem, 6 for warps). At 256 threads/block, 2 blocks = 512 threads = 16 warps out of a theoretical 48 (SM120 max) → 33.33%. NCU's estimated speedup from resolving this is 66.67% — the single largest lever in this profile.

The theoretical and achieved occupancy are very close (33.33% vs. 32.24%), meaning the kernel actually fills the SM to its register-limited ceiling with near-perfect efficiency. The problem isn't launch configuration or workload imbalance — it's that 128 registers/thread × 256 threads × 2 blocks = the full SM120 register file.
________________________________________________________________________________________________________
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/f677e2bb-3a65-4762-b2d8-979bf389267b" />
The curve shows occupancy at 100% for ≤40 registers/thread, then cascading down in steps. Your kernel is at ~128 registers/thread (dot at x≈128), sitting at ~33%. To get to 50% you'd need ~80 registers; to reach 66% roughly ~64 registers. These are aggressive reductions for a register-tiled kernel that explicitly accumulates in registers — the tiling strategy itself is what's burning registers.
______________________________________________________________________________________________________
<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/4dfb8164-b283-4cfe-b165-e9977fffc77e" />

<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/2a381f38-1ee7-4bfe-8f8f-0f0b28b24871" />

<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/e8174cbd-7f66-4a5a-87d7-f1e6400dafcc" />

<img width="1826" height="1024" alt="image" src="https://github.com/user-attachments/assets/ef240166-bd8c-4639-92c6-0dd41e3f9209" />


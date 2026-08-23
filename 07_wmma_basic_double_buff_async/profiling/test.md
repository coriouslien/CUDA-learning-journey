15.22 ms / 34.9M cycles on RTX 5080, Grid: (64, 64, 1), Block: (256, 1, 1) (8 warps/block).
GPU Speed Of Light Throughput
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/caebdf8a-0e58-4e5f-98e4-d09d5a58a09f" />

_____________________________________________________________________________________________________
Floating Point Operations Roofline (Half Precision)
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/91de6d8a-3cc5-4255-bdd4-3b8f8fd16969" />
Memory: 88.20% / Compute: 36.68%, Kernel is heavily bounded by the memory subsystem, specifically 
L1/Shared Memory.

_____________________________________________________________________________________________________
Compute Workload Analysis
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/c9d49c18-d369-4ae2-98da-40149ec26191" />


_____________________________________________________________________________________________________
Memory Workload Analysis
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/1227b465-30df-4a8a-9727-2ed36d5da0c6" />


_____________________________________________________________________________________________________
Memory Chart
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/be49e134-2d60-480b-9a54-b05ad9461d1c" />
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/f74ff8a0-73bf-4249-96a1-fc944c3eba5b" />


_____________________________________________________________________________________________________
Warp State Statistics
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/72a77cc4-2581-46f8-b035-2f87f811ab87" />


_____________________________________________________________________________________________________
Warp State (All Cycles)
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/5a8101ce-f5dc-4e53-98d8-220609f21d2e" />


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


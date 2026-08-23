The kernel is currently heavily latency-bound and suffering from severe memory access inefficiencies.
GPU Speed Of Light Throughput
<img width="1024" height="590" alt="image" src="https://github.com/user-attachments/assets/9eb273ad-a46b-4188-9f82-bcb94dd8717e" />
The kernel is significantly underperforming its theoretical limits. Both Compute and Memory Speed Of Light (SOL) throughputs are stalled at approximately 45.48%. When compute and memory are both this low, it typically indicates that the GPU is struggling to hide latency and is spending too much time waiting for data rather than executing instructions.
________________________________________________________________________________________________
Floating Point Operations Roofline (Half Precision)
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/bcc31618-f664-4a5f-8c26-adcd32e392e1" />

________________________________________________________________________________________________
Compute Workload Analysis
<img width="1024" height="590" alt="image" src="https://github.com/user-attachments/assets/cc58b72e-e66c-4702-b245-ea699ec2b896" />


________________________________________________________________________________________________
Memory Workload Analysis
<img width="1024" height="590" alt="image" src="https://github.com/user-attachments/assets/155eaa3e-1dc2-49e4-8366-8c52b0143cbd" />
<img width="1024" height="590" alt="image" src="https://github.com/user-attachments/assets/46990ac3-7bf5-4fdd-8e01-83b10a2b341d" />
<img width="1024" height="590" alt="image" src="https://github.com/user-attachments/assets/7d650ebc-0f32-4742-87d8-14a1ef464d84" />




________________________________________________________________________________________________
Scheduler Statistics
<img width="1024" height="590" alt="image" src="https://github.com/user-attachments/assets/120bac71-ddbe-4e6f-9a7f-1a80dd837eb0" />


________________________________________________________________________________________________
Warps Per Scheduler
<img width="1024" height="590" alt="image" src="https://github.com/user-attachments/assets/54ad6f67-ddcd-4de5-b6a2-b3d76d0efc71" />


________________________________________________________________________________________________
Warp State Statistics
<img width="1024" height="590" alt="image" src="https://github.com/user-attachments/assets/703e931f-3a97-42c2-8efd-121def05958b" />


________________________________________________________________________________________________
Warp State (All Cycles)
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/98387f20-4426-433c-8e24-be3da969e894" />


________________________________________________________________________________________________
Occupancy
<img width="1024" height="590" alt="image" src="https://github.com/user-attachments/assets/5c994ab1-7807-432a-bfc5-284ac6273914" />

_________________________________________________________________________________________________
Impact of Varying Block Size
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/646b46ac-a8f6-4341-a1d7-4c29f54c372d" />




________________________________________________________________________________________________
Impact of Varying Shared Memory Usage Per Block
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/f8c12921-4f9e-473f-bb59-b91a2e1582a5" />

________________________________________________________________________________________________
Impact of Varying Block Barriers & Impact of Varying Cluster Size
<img width="1924" height="1109" alt="image" src="https://github.com/user-attachments/assets/17eeeb47-4a1b-4dfb-baa0-02ba2e45f496" />



________________________________________________________________________________________________
GPU and Memory Workload Distribution 
<img width="1024" height="590" alt="image" src="https://github.com/user-attachments/assets/ca36c27c-0226-4b0b-86aa-4304483e555e" />


________________________________________________________________________________________________
Source Counters
<img width="1024" height="590" alt="image" src="https://github.com/user-attachments/assets/b9fe5fba-67da-4950-8a6d-e42f1d4eb0c1" />


________________________________________________________________________________________________


________________________________________________________________________________________________


________________________________________________________________________________________________


________________________________________________________________________________________________

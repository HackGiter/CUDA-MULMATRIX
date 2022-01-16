### Single-precision Generic Matrix-matrix Multiplication

I try to optimize my own SGEMM to surpass the CUBLASS application. According to the results of mulMatrixKernelXI which is the best version compared with CUBLASS, I did make it. But I do my endeavor to reduce the usage of registers per threads, it comes out that the code is nasty and in a mess which trades with the best result. If you want another good look code, actually it just need to replace the calculation and some repeated items with macro or just add some constant variables(which will decrease the performance, but if you constrain the registers per thread you add ,you will still get the function surpass the CUBLASS).

During the experiments, I find some tips about the CUDA:

1. Shared Memory：If you need to reuse the datas frequently in each thread, please just store the data in shared memory which will be access by the threads in the same block.
2. Registers: If you want the GPU in busy, you need to let the registers per threads as less as you can, although there will be some thresholds where you decrease the registers but the performance just stay static. But in most situations, less is good.
3. Calculation: Maybe there are a lot of place you need to calculate the same values over and over again, but additions/subtraction/multiplication(Single precision) instruction actually take few clocks to finish, especially  additions and subtraction instruction, but if you just add a register store the values(repeated over and over again), the more threads per block you have, the more cost you have to pay in most case.
4. Double-precision multiplication: Please, Please, if you don't need to , try not to use. Every SM have far much less DP units compared to FP32 unit and INT, and FP64 may take more clocks than other instructions.
5. Bank Conflict: Shared memory is split into 32 banks to access in GPU.  If every thread in a warp(32 threads per warp) access the same bank but different words,  the access has to be serialized causing delays. But if access the same word in the same bank, the data will broadcast which will did benefit every thread. So try to let the thread in the block access different banks in every instruct. Another situation is that if you use float2/float4 data access, it seems that you just need to avoid half-warp/quarter-warp access the same bank different word.
6. Texture Memory: I didn't finish experiment that the data access by texture memory instead of global memory, which I will accomplish in the future.
7. Occupancy: The high occupancy will help your  warps to hide the latency caused by some stall(like accessing the data in global memory) by warp-schedule(shift the warp in active make full use of the arithmetic units) . And the occupancy relate to the registers per thread.





 
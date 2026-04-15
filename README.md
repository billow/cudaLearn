# cudaLearn
My own learning of cuda

 ./imflipG test.bmp testOut.bmp T
Number of GPU devices: 1
GPU support MaxThrPerBlk: 1024.

 Input File Name:          test.bmp (1920 x 1080) File Size: 6220800 bytes
 
 Output File Name:       testOut.bmp (1920 x 1080) File Size: 6220800 bytes
 
--...--

NVIDIA GeForce RTX 5060 ComputeCapab=12.0 [supports max 2048 M blocks]

--...--

imflipG test.bmp testOut.bmp T 256 [8640 BLOCKS, 8 BLOCKS/ROW]

--------------------...--------------------
CPU->GPU Transfer =  2.05 ms ...   5 MB ...  2.90 GB/s

Kernel Execution =  0.24 ms ...  23 MB ...100.43 GB/s

GPU->CPU Transfer =  1.90 ms ...   5 MB ...  3.13 GB/s

Total time elapsed =  2.05 ms
--------------------...--------------------

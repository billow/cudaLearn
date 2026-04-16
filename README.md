# cudaLearn
My own learning of cuda

 ./imflipG test.bmp testOut.bmp T
Number of GPU devices: 1
GPU support MaxThrPerBlk: 1024.

 Input File Name:          test.bmp (5228 x 3200) File Size: 50188800 bytes
 Output File Name:       testOut.bmp (5228 x 3200) File Size: 50188800 bytes
--...--
NVIDIA GeForce RTX 5060 ComputeCapab=12.0 [supports max 2048 M blocks]
--...--
imflipG test.bmp testOut.bmp T 256 [67200 BLOCKS, 21 BLOCKS/ROW]
--------------------...--------------------
CPU->GPU Transfer = 20.13 ms ...  47 MB ...  2.38 GB/s
Kernel Execution =  1.01 ms ... 191 MB ...190.24 GB/s
GPU->CPU Transfer = 13.61 ms ...  47 MB ...  3.52 GB/s
Total time elapsed = 34.75 ms
--------------------...--------------------

 ./imflipG test.bmp testOut.bmp M
Number of GPU devices: 1
GPU support MaxThrPerBlk: 1024.

 Input File Name:          test.bmp (5228 x 3200) File Size: 50188800 bytes
 Output File Name:       testOut.bmp (5228 x 3200) File Size: 50188800 bytes
--...--
NVIDIA GeForce RTX 5060 ComputeCapab=12.0 [supports max 2048 M blocks]
--...--
imflipG test.bmp testOut.bmp M 256 [49013 BLOCKS, 21 BLOCKS/ROW]
--------------------...--------------------
CPU->GPU Transfer = 20.04 ms ...  47 MB ...  2.39 GB/s
Kernel Execution =  0.71 ms ... 191 MB ...268.85 GB/s
GPU->CPU Transfer = 13.68 ms ...  47 MB ...  3.50 GB/s
Total time elapsed = 34.43 ms
--------------------...--------------------

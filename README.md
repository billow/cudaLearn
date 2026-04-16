# cudaLearn
My own learning of cuda

./imflipG.exe test.bmp testOut.bmp V 512
Number of GPU devices: 1
GPU support MaxThrPerBlk: 1024.

 Input File Name:          test.bmp (5228 x 3200) File Size: 50188800 bytes
 Output File Name:       testOut.bmp (5228 x 3200) File Size: 50188800 bytes
--...--
NVIDIA GeForce RTX 5060 ComputeCapab=12.0 [supports max 2048 M blocks]
--...--
imflipG test.bmp testOut.bmp V 512 [35200 BLOCKS, 11 BLOCKS/ROW]
--------------------...--------------------
CPU->GPU Transfer = 21.32 ms ...  47 MB ...  2.25 GB/s
Kernel Execution =  0.63 ms ...  95 MB ...152.70 GB/s
GPU->CPU Transfer = 13.71 ms ...  47 MB ...  3.49 GB/s
Total time elapsed = 35.66 ms
--------------------...--------------------

./imflipG.exe test.bmp testOut.bmp M 512
Number of GPU devices: 1
GPU support MaxThrPerBlk: 1024.

 Input File Name:          test.bmp (5228 x 3200) File Size: 50188800 bytes
 Output File Name:       testOut.bmp (5228 x 3200) File Size: 50188800 bytes
--...--
NVIDIA GeForce RTX 5060 ComputeCapab=12.0 [supports max 2048 M blocks]
--...--
imflipG test.bmp testOut.bmp M 512 [24507 BLOCKS, 11 BLOCKS/ROW]
--------------------...--------------------
CPU->GPU Transfer = 20.58 ms ...  47 MB ...  2.33 GB/s
Kernel Execution =  0.56 ms ...  95 MB ...170.31 GB/s
GPU->CPU Transfer = 13.67 ms ...  47 MB ...  3.50 GB/s
Total time elapsed = 34.81 ms
--------------------...--------------------

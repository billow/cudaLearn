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

 ./imflipg test.bmp testOut.bmp S 256 8
Number of GPU devices: 1
GPU support MaxThrPerBlk: 1024.

 Input File Name:          test.bmp (5228 x 3200) File Size: 50188800 bytes
Stream 0: Processing rows 0 to 399 (total 400 rows, 1568400 ints)
Stream 1: Processing rows 400 to 799 (total 400 rows, 1568400 ints)
Stream 2: Processing rows 800 to 1199 (total 400 rows, 1568400 ints)
Stream 3: Processing rows 1200 to 1599 (total 400 rows, 1568400 ints)
Stream 4: Processing rows 1600 to 1999 (total 400 rows, 1568400 ints)
Stream 5: Processing rows 2000 to 2399 (total 400 rows, 1568400 ints)
Stream 6: Processing rows 2400 to 2799 (total 400 rows, 1568400 ints)
Stream 7: Processing rows 2800 to 3199 (total 400 rows, 1568400 ints)
 Output File Name:       testOut.bmp (5228 x 3200) File Size: 50188800 bytes
--...--
NVIDIA GeForce RTX 5060 ComputeCapab=12.0 [supports max 2048 M blocks], support
overlap: Yes
--...--
imflipG test.bmp testOut.bmp S 256 [2043 BLOCKS]
--------------------...--------------------
Total time elapsed = 15.37 ms
--------------------...--------------------

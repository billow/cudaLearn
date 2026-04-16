#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <ctype.h>
#include <cuda.h>

typedef unsigned char uch;
typedef unsigned int ui;
typedef unsigned long ul;

uch *TheImg, *CopyImg;
uch *GPUImg, *GPUCopyImg, *GPUResult;
#define IPHB ip.Hbytes
#define IPH  ip.Hpixels
#define IPV ip.Vpixels
#define IMAGESIZE (IPHB * IPV)
#define IMAGEPIXELS (IPH * IPV)

struct ImgProp{
    ui Hpixels;
    ui Vpixels;
    uch HeaderInfo[54];
    ul Hbytes;
};
struct Pixel {
    uch B;
    uch G;
    uch R;
};
ImgProp ip;
char Flip = 'V';

uch *ReadBMPlin(char *fn)
{
    static uch *Img;
    FILE *f = fopen(fn, "rb");
    if (!f) {
        fprintf(stderr, "Error opening file %s\n", fn);
        exit(EXIT_FAILURE);
    }
    uch HeaderInfo[54];
    if (fread(HeaderInfo, sizeof(uch), 54, f) != 54) {
        fprintf(stderr, "Error reading BMP header from file %s\n", fn);
        fclose(f);
        exit(EXIT_FAILURE);
    }
    int width = *(int*)&HeaderInfo[18];
    IPH = width;
    int height = *(int*)&HeaderInfo[22];
    IPV = height;
    int RowBytes = (width * 3 + 3) & (~3);
    IPHB = RowBytes;
    memcpy(ip.HeaderInfo, &HeaderInfo, 54);
    printf("\n Input File Name: %17s (%u x %u) File Size: %u bytes\n", fn, ip.Hpixels, ip.Vpixels, IMAGESIZE);

    Img = (uch *)malloc(IMAGESIZE);
    if (!Img) {
        fprintf(stderr, "Memory allocation failed for image data.\n");
        fclose(f);
        return Img;
    }

    // Read the image data into Img
    fread(Img, sizeof(uch), IMAGESIZE, f);
    fclose(f);
    return Img;
}

void WriteBMPlin(uch *Img, char *fn)
{
    FILE *f = fopen(fn, "wb");
    if (!f) {
        fprintf(stderr, "Error opening file %s for writing\n", fn);
        exit(EXIT_FAILURE);
    }
    fwrite(ip.HeaderInfo, sizeof(uch), 54, f);
    fwrite(Img, sizeof(uch), IMAGESIZE, f);
    printf(" Output File Name: %17s (%u x %u) File Size: %u bytes\n", fn, ip.Hpixels, ip.Vpixels, IMAGESIZE);
    fclose(f);
}

__global__ void Vflip(uch *ImgDst, uch *ImgSrc, ui Hpixels, ui Vpixels, ui totalPixels)
{
    // row = idx / Hpixels
    // col = idx % Hpixels
    // dstRow = Vpixels - 1 - row
    // dstCol = col
    ui idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalPixels) return; // Out of bounds check
    ui srcIdx = idx * 3;
    ui dstIdx = ((Vpixels - 1 - (idx / Hpixels)) * Hpixels + (idx % Hpixels)) * 3;
    // Each pixel has 3 bytes (BGR)
    ImgDst[dstIdx] = ImgSrc[srcIdx];
    ImgDst[dstIdx + 1] = ImgSrc[srcIdx + 1];
    ImgDst[dstIdx + 2] = ImgSrc[srcIdx + 2];
}

__global__ void VflipM(ui *ImgDst, ui *ImgSrc, ui Vpixels, ui rowInts, ui totalInts)
{
    // 4 byts per thread
    // row = idx / rowInts
    // col = idx % rowInts
    // dstRow = Vpixels - 1 - row
    // dstCol = col
    ui idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalInts) return; // Out of bounds check
    ui dstIdx = (Vpixels - 1 - (idx / rowInts)) * rowInts + (idx % rowInts);
    ImgDst[dstIdx] = ImgSrc[idx];
}

__global__ void Hflip(uch *ImgDst, uch *ImgSrc, ui Hpixels, ui Vpixels, ui totalPixels)
{
    // row = idx / Hpixels
    // col = idx % Hpixels
    // dstRow = row
    // dstCol = Hpixels - 1 - col
    ui idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalPixels) return; // Out of bounds check
    ui srcIdx = idx * 3;
    ui dstIdx = ((idx / Hpixels) * Hpixels + (Hpixels - 1 - (idx % Hpixels))) * 3;
    // Each pixel has 3 bytes (BGR)
    ImgDst[dstIdx] = ImgSrc[srcIdx];
    ImgDst[dstIdx + 1] = ImgSrc[srcIdx + 1];
    ImgDst[dstIdx + 2] = ImgSrc[srcIdx + 2];
}

__global__ void HflipM(ui *ImgDst, ui *ImgSrc, ui Hpixels, ui rowInts, ui totalInts)
{
    // 4 byts per thread
    // row = idx / rowInts
    // col = idx % rowInts
    // dstRow = row
    // dstCol = Hpixels - 1 - col
    ui idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalInts) return; // Out of bounds check
    ui dstIdx = (idx / rowInts) * rowInts + (rowInts - 1 - (idx % rowInts));
    ImgDst[dstIdx] = ImgSrc[idx];
}

__global__ void PixCopy(uch *ImgDst, uch *ImgSrc, ui TotalPixels)
{
    ui idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TotalPixels) return; // Out of bounds check
    ImgDst[idx] = ImgSrc[idx];
}

int main(int argc, char *argv[]) {
    char InputFileName[255], OutputFileName[255], ProgName[255];
    cudaError_t cudaStatus, cudaStatus2;
    cudaEvent_t time1, time2, time3, time4;
    float totalTime, tfrCPUtoGPU, kernelExecutionTime, tfrGPUtoCPU;
    ui BlkPerRow, ThrPerBlk=256, NumBlocks, GPUDataTransfer;
    cudaDeviceProp GPUprop;
    ul SupportedKBlocks, SupportedMBlocks, MaxThrPerBlk;
    char SupportedBlocks[100];

    int NumGPUs = 0;
    cudaGetDeviceCount(&NumGPUs);
    if (NumGPUs == 0) {
        printf("No GPU device found.\n");
        return EXIT_FAILURE;
    }
    else
    {
        printf("Number of GPU devices: %d\n", NumGPUs);
    }
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
        return EXIT_FAILURE;
    }
    cudaStatus2 = cudaGetDeviceProperties(&GPUprop, 0);
    if (cudaStatus2 != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed!\n");
        return EXIT_FAILURE;
    }
    SupportedKBlocks = (ui)GPUprop.maxGridSize[0] * (ui)GPUprop.maxGridSize[1] * (ui)GPUprop.maxGridSize[2] / 1024;
    SupportedMBlocks = SupportedKBlocks / 1024;
    sprintf(SupportedBlocks, "%lu %c", (SupportedKBlocks >= 5) ? SupportedMBlocks : SupportedKBlocks, 
        (SupportedKBlocks >= 5) ? 'M' : 'K');
    MaxThrPerBlk = (ui)GPUprop.maxThreadsPerBlock;
    printf("GPU support MaxThrPerBlk: %lu.\n", MaxThrPerBlk);

    strcpy(ProgName, "imflipG");
    switch (argc) {
        case 5: ThrPerBlk = atoi(argv[4]);
        case 4: Flip = toupper(argv[3][0]);
        case 3:
            strcpy(InputFileName, argv[1]);
            strcpy(OutputFileName, argv[2]);
            break;
        default:
            printf("\n\nUsage: %s <inputFileName> <outputFileName>\n", ProgName);
            return EXIT_FAILURE;
    }

    TheImg = ReadBMPlin(InputFileName);
    CopyImg = (uch *)malloc(IMAGESIZE);

    cudaEventCreate(&time1);
    cudaEventCreate(&time2);
    cudaEventCreate(&time3);
    cudaEventCreate(&time4);  

    // Allocate memory on the GPU
    cudaEventRecord(time1, 0);
    cudaStatus = cudaMalloc((void**)&GPUImg, IMAGESIZE);
    cudaStatus2 = cudaMalloc((void**)&GPUCopyImg, IMAGESIZE);
    if (cudaStatus != cudaSuccess || cudaStatus2 != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed! Can't allocate GPU memory.\n");
        return EXIT_FAILURE;
    }

    // Copy the image data from host to device
    cudaStatus = cudaMemcpy(GPUImg, TheImg, IMAGESIZE, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! Can't copy data to GPU.\n");
        return EXIT_FAILURE;
    }
    cudaEventRecord(time2, 0);
    BlkPerRow = (IPH + ThrPerBlk - 1) / ThrPerBlk;
    NumBlocks = BlkPerRow * IPV;

    switch (Flip) {
        case 'H': Hflip <<<NumBlocks, ThrPerBlk>>>(GPUCopyImg, GPUImg, IPH, IPV, IMAGEPIXELS);
            GPUResult = GPUCopyImg;
            GPUDataTransfer = 2 * IMAGESIZE;
            break;
        case 'V': Vflip <<<NumBlocks, ThrPerBlk>>>(GPUCopyImg, GPUImg, IPH, IPV, IMAGEPIXELS);
            GPUResult = GPUCopyImg;
            GPUDataTransfer = 2 * IMAGESIZE;
            break;
        case 'T': Hflip <<<NumBlocks, ThrPerBlk>>>(GPUCopyImg, GPUImg, IPH, IPV, IMAGEPIXELS);
            Vflip <<<NumBlocks, ThrPerBlk>>>(GPUImg, GPUCopyImg, IPH, IPV, IMAGEPIXELS);
            GPUResult = GPUImg;
            GPUDataTransfer = 4 * IMAGESIZE;
            break;
        case 'C': NumBlocks = (IMAGESIZE + ThrPerBlk - 1) / ThrPerBlk;
            PixCopy <<<NumBlocks, ThrPerBlk>>>(GPUCopyImg, GPUImg, IMAGESIZE);
            GPUResult = GPUCopyImg;
            GPUDataTransfer = 2 * IMAGESIZE;
            break;
        case 'M': {
            ui rowInts = (IPH * 3) / 4; // Number of 4-byte integers per row
            ui totalInts = rowInts * IPV; // Total number of 4-byte integers in the image
            NumBlocks = (rowInts * IPV + ThrPerBlk - 1) / ThrPerBlk;
            VflipM <<<NumBlocks, ThrPerBlk>>>((ui*)GPUCopyImg, (ui*)GPUImg, IPV, rowInts, totalInts);
            GPUResult = GPUCopyImg;
            GPUDataTransfer = 2 * IMAGESIZE;
            break;
        }
        default:
            fprintf(stderr, "Invalid flip type! Use 'H' for horizontal or 'V' for vertical.\n");
            return EXIT_FAILURE;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize errorCode: %d.\n", cudaStatus);
        return EXIT_FAILURE;
    }
    cudaEventRecord(time3, 0);

    // // Copy the result back to host
    cudaStatus = cudaMemcpy(CopyImg, GPUResult, IMAGESIZE, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! Can't copy data from GPU.\n");
        return EXIT_FAILURE;
    }
    cudaEventRecord(time4, 0);

    cudaEventSynchronize(time1);
    cudaEventSynchronize(time2);
    cudaEventSynchronize(time3);
    cudaEventSynchronize(time4);
    cudaEventElapsedTime(&totalTime, time1, time4);
    cudaEventElapsedTime(&tfrCPUtoGPU, time1, time2);
    cudaEventElapsedTime(&kernelExecutionTime, time2, time3);
    cudaEventElapsedTime(&tfrGPUtoCPU, time3, time4);

    
    // // Free GPU memory
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize errorCode: %d.\n", cudaStatus);
        free(TheImg);
        free(CopyImg);
        return EXIT_FAILURE;
    }
    WriteBMPlin(CopyImg, OutputFileName);
    
    printf("--...--\n");
    printf("%s ComputeCapab=%d.%d [supports max %s blocks]\n", GPUprop.name, GPUprop.major, GPUprop.minor, SupportedBlocks);
    printf("--...--\n");
    printf("%s %s %s %c %u [%u BLOCKS, %u BLOCKS/ROW]\n", ProgName, InputFileName, OutputFileName, Flip, ThrPerBlk, NumBlocks, BlkPerRow);
    printf("--------------------...--------------------\n");
    printf("CPU->GPU Transfer = %5.2f ms ...%4d MB ...%6.2f GB/s\n", tfrCPUtoGPU, IMAGESIZE / (1024 * 1024), (float)IMAGESIZE / (tfrCPUtoGPU * 1024.0 * 1024.0));
    printf("Kernel Execution = %5.2f ms ...%4d MB ...%6.2f GB/s\n", kernelExecutionTime, GPUDataTransfer / (1024 * 1024), (float)GPUDataTransfer / (kernelExecutionTime * 1024.0 * 1024.0));
    printf("GPU->CPU Transfer = %5.2f ms ...%4d MB ...%6.2f GB/s\n", tfrGPUtoCPU, IMAGESIZE / (1024 * 1024), (float)IMAGESIZE / (tfrGPUtoCPU * 1024.0 * 1024.0));
    printf("Total time elapsed = %5.2f ms\n", totalTime);
    printf("--------------------...--------------------\n");

    cudaFree(GPUImg);
    cudaFree(GPUCopyImg);
    cudaEventDestroy(time1);
    cudaEventDestroy(time2);
    cudaEventDestroy(time3);
    cudaEventDestroy(time4);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!\n");
        free(TheImg);
        free(CopyImg);
        return EXIT_FAILURE;
    }
    free(TheImg);
    free(CopyImg);

    return EXIT_SUCCESS;
}
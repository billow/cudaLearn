#include <cuda_runtime.h>
#include <iostream>

const int MATRIX_SIZE = 256;

__global__ void matrixMultiplyShared(float* A, float* B, float* C, int N) {
    __shared__ float tileA[16][16];
    __shared__ float tileB[16][16];

    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;

    float value = 0.0f;

    for (int tile = 0; tile < (N + 15) / 16; ++tile) {
        if (row < N && tile * 16 + threadIdx.x < N) {
            tileA[threadIdx.y][threadIdx.x] = A[row * N + (tile * 16 + threadIdx.x)];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (tile * 16 + threadIdx.y < N && col < N) {
            tileB[threadIdx.y][threadIdx.x] = B[(tile * 16 + threadIdx.y) * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads(); // tileA[0, 0~15] = A[0, 0~15] and tileB[0~15, 0] = B[0~15, 0]

        for (int k = 0; k < 16; ++k) {
            value += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads(); // C[0, 0] = A[0, 0~15] * B[0~15, 0] + A[0, 16~31] * B[16~31, 0] + ... + A[0, 240~255] * B[240~255, 0]
    }

    C[row * N + col] = value;
}

void initializeMatrix(float* matrix, int N) {
    for (int i = 0; i < N * N; ++i) {
        matrix[i] = static_cast<float>(rand() % 10);
    }
}

void printMatrix(const float* matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cout << matrix[i * size + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    int N = MATRIX_SIZE;
    size_t bytes = N * N * sizeof(float);
    float* h_A = new float[N * N];
    float* h_B = new float[N * N];
    float* h_C = new float[N * N];

    initializeMatrix(h_A, N);
    initializeMatrix(h_B, N);

    float* d_A;
    float* d_B;
    float* d_C;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    float costTime;
    cudaEvent_t time1, time2;
    cudaEventCreate(&time1);
    cudaEventCreate(&time2);
    cudaEventRecord(time1);
    matrixMultiplyShared<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaEventRecord(time2);
    cudaEventSynchronize(time1);
    cudaEventSynchronize(time2);
    cudaEventElapsedTime(&costTime, time1, time2);


    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    std::cout << "Result of Matrix Multiplication:" << std::endl;
    printMatrix(h_C, 20);

    std::cout << "Time taken: " << costTime << " ms" << std::endl;

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
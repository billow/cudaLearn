#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

#define CHECK_CUBLAS(call) \
    { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << status << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

void initializeMatrix(float* matrix, int rows, int cols, float value) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = value;
    }
}

void printMatrix(const float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    const int M=4, N=3, K=5;
    const float alpha = 2.0f, beta = 0.5f;
    // Allocate host memory
    float h_A[M*K];
    float h_B[N*K];
    float h_C[M*N];

    // Initialize host matrices
    initializeMatrix(h_A, M, K, 1.0f); // Fill A with 1.0
    initializeMatrix(h_B, K, N, 2.0f); // Fill B with 2.0
    initializeMatrix(h_C, M, N, 1.0f); // Fill C with 0.0

    // Allocate device memory
    float* d_A; 
    float* d_B;
    float* d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Perform matrix multiplication: C = alpha * A * B + beta * C
    CHECK_CUBLAS(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             M, N, K,
                             &alpha,
                             d_A, M,
                             d_B, K,
                             &beta,                            
                             d_C, M));

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Print the result
    std::cout << "Result of A + B:" << std::endl;
    printMatrix(h_C, M, N);

    // Clean up
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
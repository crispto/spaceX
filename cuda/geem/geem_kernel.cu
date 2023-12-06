#include "geem.h"
#include "stdio.h"

__global__ void geem_float_kernel(float *A, float *B, float *C, int M, int N, int K)
{
    // int threadId = threadIdx.x + threadIdx.y * blockDim.x + (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x *
    // blockDim.y);
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 这里之所以是 blockDim  是因为我们要换算成线程的横纵坐标
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < K) {
        float total = 0;
        for (int i = 0; i < N; i++) {
            total += A[row * N + i] * B[i * K + col]; // 这里全都是访问的 global memory
        }
        C[row * K + col] = total;
    }
}

// 原始的 矩阵乘法，每一个 thread 负责结果中  [row, col] 这个结果的计算
void geem_float(float *A, float *B, float *C, int M, int N, int K)
{
    float *dev_A, *dev_B, *dev_C;
    cudaMalloc((void **)&dev_A, sizeof(float) * (M * N));
    cudaMalloc((void **)&dev_B, sizeof(float) * (N * K));
    cudaMalloc((void **)&dev_C, sizeof(float) * (M * K));

    cudaMemcpy(dev_A, A, sizeof(float) * (M * N), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, sizeof(float) * (N * K), cudaMemcpyHostToDevice);

    dim3 blockDim = { 16, 16 };
    dim3 gridDim = { (K + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y };

    geem_float_kernel<<<gridDim, blockDim>>>(dev_A, dev_B, dev_C, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(C, dev_C, sizeof(float) * (M * K), cudaMemcpyDeviceToHost);
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
}

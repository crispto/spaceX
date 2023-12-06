#include "geem.h"
#include "stdio.h"

// 每一个线程负责 C 中的一个元素
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

// 使用 shared_memory， 让一个 block 中的 thread 先把需要的数据都拷到 共享内存中
__global__ void geem_float_kernel2(float *A, float *B, float *C, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float array[];
    float *row_data = array;
    float *col_data = row_data + blockDim.y * N * sizeof(float);
    // 注意内存合并
    // 获取行
    for (int id = threadIdx.x; id < N; id += blockDim.x) {
        row_data[blockIdx.y * N + id] = A[row * N + id];
    }
    // 获取列
    for (int id = threadIdx.y; id < N; id += blockDim.y) {
        col_data[blockIdx.x * N + id] = B[id * K + col];
    }
    // 同步： 等 block 内的所有 thread 将 本 block 的数据从 global memory 中的数据都取回
    __syncthreads();
    float total = 0;
    for (int i = 0; i < N; i++) {
        total += row_data[blockIdx.y * N + i] * col_data[blockIdx.x * N + i];
    }
    C[row * K + col] = total;
}

void geem_float2(float *A, float *B, float *C, int M, int N, int K)
{
    float *dev_A, *dev_B, *dev_C;
    cudaMalloc((void **)&dev_A, sizeof(float) * (M * N));
    cudaMalloc((void **)&dev_B, sizeof(float) * (N * K));
    cudaMalloc((void **)&dev_C, sizeof(float) * (M * K));

    cudaMemcpy(dev_A, A, sizeof(float) * (M * N), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, sizeof(float) * (N * K), cudaMemcpyHostToDevice);

    dim3 blockDim = { 16, 16 };
    dim3 gridDim = { (K + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y };

    unsigned share_memory_size = blockDim.y * N * sizeof(float) + blockDim.x * N * sizeof(float);
    geem_float_kernel2<<<gridDim, blockDim, share_memory_size>>>(dev_A, dev_B, dev_C, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(C, dev_C, sizeof(float) * (M * K), cudaMemcpyDeviceToHost);
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
}

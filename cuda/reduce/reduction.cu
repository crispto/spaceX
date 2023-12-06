#include <cuda_runtime.h>
#include "a.hpp"
#include <cuda.h>

__global__ void reduce_kernel(float *p, ulong N, float *ret)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // 这里先使用 block 内的 shared memory 来汇总
    __shared__ float result[1024];
    result[threadIdx.x] = 0.0f;
    for (ulong i = tid; i < N; i += blockDim.x) {
      result[threadIdx.x] += p[i];
    }
    __syncthreads();
    if (threadIdx.x == 0){
      float tmp = 0;
      for (int i =0;i < 1024;i++){
          tmp += result[i];
      }
      *ret = tmp;
    }

    
}
// 使用 cuda 硬件加速
float reduce_cuda(float *p, ulong N)
{
    float *dev_p, *dev_ret;
    cudaMalloc((void **)&dev_p, N * sizeof(float));
    cudaMalloc((void **)&dev_ret, 1 * sizeof(float));
    cudaMemcpy(dev_p, p, N, cudaMemcpyHostToDevice);

    dim3 blockDim= 1024;
    dim3 gridDim = (N + blockDim.x - 1) / blockDim.x;

    reduce_kernel<<<1, blockDim>>>(dev_p, N, dev_ret);

    cudaDeviceSynchronize();
    float host_ret;

    cudaMemcpy(&host_ret, dev_ret, 1 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev_p);
    cudaFree(dev_ret);

    return host_ret/N;

}

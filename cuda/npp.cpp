#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;
// M * N  ** N*K
template<typename T>
void gemm_cpu(T *A, T* B, T* C, int M, int N, int K){
    for(int i = 0;i < M;i++){
        for (int j = 0;j < K;j++){
            T total = 0;
            for (int u = 0;u < N;u++){
                    total += A[i * N +u] * B[u * K + j];
            }
            C[i*K + j] = total;
        }
    }
}
void gemm_gpu_origin(float *A, float* B, float* C, int M, int N, int K){
    float *dev_A, *dev_B, *dev_C;
    cudaMalloc((void**)dev_A, sizeof(float) * (M*N));
    cudaMalloc((void**)dev_B, sizeof(float) * (N*K));
    cudaMalloc((void**)dev_C, sizeof(float) * (M*K));

    cudaMemcpy(dev_A, A, sizeof(float)*(M*N), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, sizeof(float)*(N*K), cudaMemcpyHostToDevice);

    dim3 blockDim = {32, 32};


}

template<typename T>
void display_matrix(T *A, int row, int col){
    for (int i = 0;i < row;i++){
        for (int j = 0;j < col;j++){
            cout << A[i*col + j] << "\t";
        }
        cout << endl;
    }
    cout << endl;

}


int main(){
    int M = 3;
    int N = 4;
    int K = 5;
    int *A = new int[M*N];
    int *B = new int[N*K];
    int *C = new int[M*K];
    for (int i = 0;i < M;i++){
        for (int j = 0;j < N;j++){
            A[i*N + j] = i*N + j;
        }
    }
    for (int i = 0;i < N;i++){
        for (int j = 0;j < K;j++){
            B[i*K + j] = i*K + j;
        }
    }
    gemm_cpu(A, B, C, M, N, K);
    display_matrix(A, M, N);
    display_matrix(B, N, K);
    display_matrix(C, M, K);
    return 0;

}

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "geem.h"
#include <cmath>
#include <random>
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
// mat check equal
template<typename T>
bool mat_check_equal(T *A, T *B, int row, int col){
    for (int i = 0;i < row;i++){
        for (int j = 0;j < col;j++){
            if (A[i*col + j] != B[i*col + j]){
                return false;
            }
        }
    }
    return true;
}

// cal rmsd of mat
template<typename T>
T mat_rmsd(T *A, T *B, int row, int col){
    T total = 0;
    for (int i = 0;i < row;i++){
        for (int j = 0;j < col;j++){
            total += powf(A[i*col + j] - B[i*col + j], 2.0);
        }
    }
    return total/(row*col);
}
// todo: 统计耗时
int main(){
    // float matrix 1000 * 1600;
    time_t t = time(nullptr);
    printf("time is %s\n", asctime(localtime(&t)));
    srand(t);

    int M = 100;
    int N = 120;
    int K = 140;
    float *A = new float[M * N];
    float *B = new float[N * K];
    float *C_cpu = new float[M * K];
    float *C_gpu = new float[M * K];
    for (int i = 0;i < M * N;i++){
        A[i] = rand() % 100;
    }
    for (int i = 0;i < N * K;i++){
        B[i] = rand() % 100;
    }

    gemm_cpu(A, B, C_cpu, M, N, K);
    geem_float(A, B, C_gpu, M, N, K);
    // display_matrix(C_cpu, M, K);
    // display_matrix(C_gpu, M, K);
    bool equal = mat_check_equal(C_cpu, C_gpu, M, K);
    std::cout << "equal: " << equal << std::endl;
    if (!equal){
        std::cout << "rmsd: " << mat_rmsd(C_cpu, C_gpu, M, K) << std::endl;
    }
    free(A);
    free(B);
    free(C_cpu);
    free(C_gpu);
    return 0;

}

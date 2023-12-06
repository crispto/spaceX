#include "geem.h"
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/trivial.hpp>

#include "profile.hpp"
#include <boost/format.hpp>
#include <cmath>
#include <co/cout.h>
#include <co/flag.h>
#include <co/log.h>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <functional>
#include <iostream>
#include <random>
#include <stdlib.h>
#include <string>
using boost::format;
namespace logging = boost::log;
using namespace std;

// M * N  ** N*K
template <typename T> void gemm_cpu(T *A, T *B, T *C, int M, int N, int K)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            T total = 0;
            for (int u = 0; u < N; u++) {
                total += A[i * N + u] * B[u * K + j];
            }
            C[i * K + j] = total;
        }
    }
}

template <typename T> void display_matrix(T *A, int row, int col)
{
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            cout << A[i * col + j] << "\t";
        }
        cout << endl;
    }
    cout << endl;
}
// mat check equal
template <typename T> bool mat_check_equal(T *A, T *B, int row, int col)
{
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            if (A[i * col + j] != B[i * col + j]) {
                return false;
            }
        }
    }
    return true;
}

// cal rmsd of mat
template <typename T> T mat_rmsd(T *A, T *B, int row, int col)
{
    T total = 0;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            total += powf(A[i * col + j] - B[i * col + j], 2.0);
        }
    }
    return total / (row * col);
}

// todo: 统计耗时

void co_write_callback(const void *buf, size_t buf_size)
{
    std::cout << "call back get log " << std::endl;
    std::cout << reinterpret_cast<const char *>(buf) << std::endl;
}
void init()
{
    logging::core::get()->set_filter(logging::trivial::severity >= logging::trivial::info);

    // coost init
    // auto cb = std::bind(co_write_callback, std::placeholders::_1, std::placeholders::_2);
    // log::set_write_cb(cb, 0);
    flag::set_value("cout", "true");
}
int main(int argc, char **argv)
{
    init();
    if (argc <= 1) {
        LOG_ERR((format("Usage: %s 10") % argv[0]).str());
        exit(0);
    } else {
        LOG_INFO((format("get args %s, %s") % argv[0] % argv[1]).str());
    }
    int scale = atoi(argv[1]);
    BOOST_LOG_TRIVIAL(trace) << "scale is " << scale << std::endl;

    // float matrix 1000 * 1600;
    int M = 100 * scale;
    int N = 120 * scale;
    int K = 140 * scale;
    float *A = new float[M * N];
    float *B = new float[N * K];
    float *C_cpu = new float[M * K];
    float *C_gpu = new float[M * K];
    for (int i = 0; i < M * N; i++) {
        A[i] = rand() % 100;
    }
    for (int i = 0; i < N * K; i++) {
        B[i] = rand() % 100;
    }

    {
        // PROF("cpu");
        PROF("cpu");
        gemm_cpu(A, B, C_cpu, M, N, K);
    }

    {
        PROF("gpu-origin");
        geem_float(A, B, C_gpu, M, N, K);
    }
    // display_matrix(C_cpu, M, K);
    // display_matrix(C_gpu, M, K);
    bool equal = mat_check_equal(C_cpu, C_gpu, M, K);
    std::cout << "equal: " << equal << std::endl;
    if (!equal) {
        std::cout << "rmsd: " << mat_rmsd(C_cpu, C_gpu, M, K) << std::endl;
    }
    free(A);
    free(B);
    free(C_cpu);
    free(C_gpu);
    return 0;
}

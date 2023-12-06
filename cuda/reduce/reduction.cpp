#include "a.hpp"
#include <assert.h>
#include <boost/format.hpp>
#include <co/cout.h>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>
using boost::format;
using namespace std;
const unsigned long N = 1 << 20;
using ulong = unsigned long;
float reduce_cpu(float *p, const unsigned long N)
{
    float total{ 0 };
    for (ulong i = 0; i < N; i++) {
        total += p[i];
    }
    return total;
}

// 分成四个线程去 reduce 并且将 future 加到一起

float param_reduce_cpu(float *p, ulong N, ulong block_size)
{
    vector<future<float>> futures;
    vector<packaged_task<float()>> tasks;
    vector<thread> threads;
    for (ulong i = 0; i < N; i += block_size) {
        ulong inner_block_size = std::min(block_size, N - i);
        auto task =
            make_shared<packaged_task<float()>>(std::packaged_task<float()>(std::bind(reduce_cpu, p, inner_block_size)));
        p += inner_block_size;

        threads.emplace_back(thread([task] { (*task)(); }));
        futures.emplace_back(task->get_future());
    }

    for (int i = 0; i < threads.size(); i++) {
        threads[i].join();
    }
    float total = 0.0f;
    for (auto &&fut : futures) {
        total += fut.get();
    }
    return total;
}

int64_t get_now_milli()
{
    return chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();
}

int main()
{
    std::srand(time(nullptr));

    auto data = shared_ptr<float>(new float[N], std::default_delete<float[]>{});
    for (ulong i = 0; i < N; i++) {
        data.get()[i] = float(std::rand() % 10000) / 100.f;
    }

    auto start = get_now_milli();

    float ret1 = reduce_cpu(data.get(), N);
    auto t1 = get_now_milli();
    cout << boost::format("reduce_cpu sum is %.4f, time cost is %.4f ms\n") % ret1 % (t1 - start);

    ulong block_size = N / 8;
    ret1 = param_reduce_cpu(data.get(), N, block_size);
    auto t2 = get_now_milli();
    cout << boost::format("param_reduce_cpu sum is %.4f, time cost is %.4f ms\n") % ret1 % (t2 - t1);

    ret1 = reduce_cuda(data.get(), N);
    auto t3 = get_now_milli();
    cout << boost::format("reduce_cuda sum is %.4f, time cost is %.4f ms\n") % ret1 % (t3 - t2);
}

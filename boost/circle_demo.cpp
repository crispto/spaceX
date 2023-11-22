// Create a circular buffer with a capacity for 3 integers.
#include <boost/circular_buffer.hpp>

#include <boost/log/trivial.hpp>
#include <chrono>
#include <condition_variable>
#include <ctime>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

/**
 * @brief 注意的点包括， 1. 唤醒的条件不止有 空->非空或满 ->不满；不要忘记还有 channel close 这个条件，
 * TODO:  可以看出虽然用了多线程，但是因为读写使用一把锁，完全没有利用到 ring buffer 的性质，所以出来的数据还是有序的
 * TODO: benchmark
 *
 */
using namespace std;
using namespace boost::log::trivial;
#define show_log(level, format, ...)                                                                                           \
    char buf[1024];                                                                                                            \
    sprintf(buf, format, __VA_ARGS__);                                                                                         \
    BOOST_LOG_TRIVIAL(level) << __FILE__ << ":" << __LINE__ << ": " << buf;

template <typename T> class Channel
{
public:
    Channel(int size) : cb_(size), close_(false)
    {
        show_log(warning, "create channel, capicity %ld, size: %ld\n", cb_.capacity(), cb_.size());
    };
    Channel(const Channel &) = delete;
    Channel &operator<<(T i);
    bool operator>>(T &i);
    inline void close();
    inline bool empty() const
    {
        return cb_.empty();
    };
    inline bool full() const
    {
        return cb_.full();
    };

private:
    boost::circular_buffer<T> cb_;
    std::mutex mtx_;
    std::condition_variable cd_read_;
    std::condition_variable cd_write_;
    bool close_{ false }; // channel is destroyed
};

/**
 * @brief 尝试写入数据，如果缓冲区满了， 则会陷入等待
 *
 * @tparam T
 * @param i
 * @return Channel<T>&
 */
template <typename T> Channel<T> &Channel<T>::operator<<(T i)
{
    std::unique_lock<std::mutex> lock(mtx_);
    if (close_) {
        throw std::runtime_error("write data to a destory channel");
    }
    if (cb_.full()) {
        cd_write_.wait(std::ref(lock), [this]() { return !cb_.full() || close_; });
    }
    if (close_) {
        return *this;
    }
    // cb_ is not full
    bool need_notify_read = cb_.empty();
    cb_.push_back(i);
    if (need_notify_read) {
        lock.unlock();
        cd_read_.notify_one();
    }
    return *this;
}

/**
 * @brief 尝试获取数据，如果缓冲区为空，会陷入等待
 *
 * @tparam T
 * @param out
 * @return bool channel 关闭且无数据，则返回 false
 */
template <typename T> bool Channel<T>::operator>>(T &out)
{
    std::unique_lock<std::mutex> lock(mtx_);
    // 只有当 channel 中无数据且关闭时 ok = false
    if (cb_.size() == 0 && close_) {
        return false;
    }
    if (cb_.empty()) {
        cd_read_.wait(std::ref(lock), [this] { return !cb_.empty() || close_; });
    }
    if (!cb_.empty()) {
        bool need_notify_write = cb_.full();

        out = cb_.front();
        cb_.pop_front();
        if (need_notify_write) {
            lock.unlock();
            cd_write_.notify_one();
        }
        return true;
    }
    // close_
    return false;
}

template <typename T> void Channel<T>::close()
{
    std::lock_guard<std::mutex> lock(mtx_);
    close_ = true;
}

int main()
{
    Channel<int> ch{ 10 };
    std::vector<thread> consumers;
    thread produce([&ch] {
        for (int i = 0; i < 1000; i++) {
            ch << i;
        }
        ch.close();
    });
    for (int i = 0; i < 10; i++) {
        consumers.emplace_back(std::thread([&ch, i] {
            bool ok;
            int var;
            while (1) {
                ok = ch >> var;
                if (!ok) {
                    show_log(info, "thread[%d] channel is closed", i);
                    return;
                }
                printf("thread[%d] process %d\n", i, var);
                this_thread::sleep_for(chrono::milliseconds(50));
            }
        }));
    }
    produce.join();
    for (auto &td : consumers) {
        td.join();
    }
    return 0;
}

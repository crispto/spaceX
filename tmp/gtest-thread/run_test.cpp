#include "gtest/gtest.h"
#include <condition_variable>
#include <functional>
#include <gtest/gtest.h>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

using namespace std;
void printFirst()
{
    cout << "first" << endl;
}
void printSecond()
{
    cout << "second" << endl;
}
void printThird()
{
    cout << "third" << endl;
}
class Foo
{
public:
    Foo()
    {
    }

    void first(function<void()> printFirst)
    {

        // printFirst() outputs "first". Do not change or remove this line.
        printFirst();
        order_ = 1;
        event_.notify_all();
    }

    void second(function<void()> printSecond)
    {
        std::unique_lock<std::mutex> lk(mtx_);
        event_.wait(lk, [this] { return this->order_ == 1; });
        // printSecond() outputs "second". Do not change or remove this line.
        printSecond();
        order_ = 2;
        lk.unlock();
        event_.notify_one();
    }

    void third(function<void()> printThird)
    {
        std::unique_lock<std::mutex> lk(mtx_);
        event_.wait(lk, [this] { return this->order_ == 2; });
        // printThird() outputs "third". Do not change or remove this line.
        printThird();
    }

private:
    std::mutex mtx_;
    std::condition_variable event_;
    int order_ = 0;
};

typedef std::vector<int> test_type_;
class SimpleTest : public testing::TestWithParam<test_type_>
{
};

static vector<test_type_> build_random_order()
{
    vector<test_type_> ret;
    ret.push_back({ 1, 2, 3 });
    ret.push_back({ 1, 3, 2 });
    ret.push_back({ 2, 1, 3 });
    ret.push_back({ 2, 3, 1 });
    ret.push_back({ 3, 1, 2 });
    ret.push_back({ 3, 2, 1 });
    return ret;
}

INSTANTIATE_TEST_SUITE_P(banana, SimpleTest, testing::ValuesIn(build_random_order()));

TEST_P(SimpleTest, test)
{
    Foo foo;
    vector<thread> threads;
    std::map<int, std::function<void()>> m;
    auto v = GetParam();
    for (auto i : v) {
        if (i == 1) {
            threads.emplace_back(&Foo::first, &foo, printFirst);
        } else if (i == 2) {
            threads.emplace_back(&Foo::second, &foo, printSecond);
        } else if (i == 3) {
            threads.emplace_back(&Foo::third, &foo, printThird);
        }
    }
    for (auto &t : threads) {
        t.join();
    }
    std::cout << endl;
}

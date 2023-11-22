#include "ringbuf.hpp"
#include <iostream>
#include <string>
#include <thread>
using namespace std;
const int N = 1000;
int main()
{
    ringbuffer<int, 32> r;
    std::thread t1([&r] {
        for (int i = 0; i < N; i++) {
            r.push(i);
            this_thread::sleep_for(chrono::milliseconds(10));
        }
    });
    std::thread t2([&r] {
        for (int i = 0; i < N; i++) {
            int value;
            r.pop(value);
            cout << value << endl;
            this_thread::sleep_for(chrono::milliseconds(10));
        }
    });
    t1.join();
    t2.join();
}

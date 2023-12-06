#ifndef __SPACE_X_PROFILE__
#define __SPACE_X_PROFILE__

#include "log.hpp"
#include <boost/format.hpp>
#include <chrono>

int64_t get_now_milli()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

class Profiler
{
public:
    Profiler(const char *name) : name_(name)
    {
        start_milli = get_now_milli();
    };
    ~Profiler()
    {
        int64_t cost = get_now_milli() - start_milli;
        LOG_INFO((boost::format("%s time: %dms") % name_ % cost).str());
    }

private:
    std::string name_;
    int64_t start_milli;
};

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
#define PROF(x) Profiler profiler_##__LINE__(__FILE__ ":" STR(__LINE__) "[" x "]")
#endif // __SPACE_X_PROFILE__

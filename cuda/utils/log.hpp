#ifndef __SPACE_X_LOG__
#define __SPACE_X_LOG__
#include <co/cout.h>
#include <co/log.h>
using namespace co::text;
#define LOG_ERR(X) ELOG << text::red(X)
#define LOG_INFO(X) LOG << text::blue(X)
#define LOG_DEBUG(X) DLOG << text::yello(X)
#endif

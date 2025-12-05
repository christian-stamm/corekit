#pragma once

#include "corekit/structs/mutex.hpp"

#include <iostream>
#include <mutex>
#include <ostream>
#include <string>

namespace corekit {
namespace logging {

    using namespace corekit::structs;

    class LogBuffer : public std::streambuf {
      public:
        virtual std::streambuf::int_type overflow(std::streambuf::int_type c) override;
        virtual std::streamsize          xsputn(const char* s, std::streamsize count) override;
    };

    class Logstream : public std::ostream {

      public:
        Logstream(const std::string& prefix);
        ~Logstream();

      protected:
        static IMutex    mutex;
        static LogBuffer buffer;

        std::scoped_lock<IMutex> lock;
    };

    inline IMutex    Logstream::mutex;
    inline LogBuffer Logstream::buffer;

}; // namespace logging
}; // namespace corekit
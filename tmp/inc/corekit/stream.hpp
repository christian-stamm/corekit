#pragma once

#include <iostream>
#include <mutex>
#include <ostream>
#include <string>

#include "corekit/system/mutex.hpp"

namespace corekit {
    namespace utils {

        using namespace corekit::system;

        class LogBuffer : public std::streambuf {
           public:
            virtual std::streamsize          xsputn(const char*     s,
                                                    std::streamsize count) override;
            virtual std::streambuf::int_type overflow(
                std::streambuf::int_type c) override;
        };

        class Logstream : public std::ostream {
           public:
            Logstream(const std::string& prefix);
            ~Logstream();

           protected:
            static Mutex     mutex;
            static LogBuffer buffer;

            std::scoped_lock<Mutex> lock;
        };

        inline Mutex     Logstream::mutex;
        inline LogBuffer Logstream::buffer;

    };  // namespace utils
};      // namespace corekit

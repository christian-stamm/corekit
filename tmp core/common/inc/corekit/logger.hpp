#pragma once

#include <iostream>
#include <memory>
#include <string>

#include "corekit/platform.hpp"
#include "corekit/serialdevice.hpp"
#include "corekit/types.hpp"

namespace corekit {

    using LogDevice = SerialDevice<uint8_t>;

    enum class LogLevel {
        FATAL = (1 << 0),
        ERROR = (1 << 1),
        WARN  = (1 << 2),
        INFO  = (1 << 3),
        DEBUG = (1 << 4),
    };

    class StreamBuf : public std::streambuf {
        friend class Logging;

       public:
        StreamBuf(const LogDevice::Ptr& device);

        virtual std::streambuf::int_type overflow(
            std::streambuf::int_type c) override;
        virtual std::streamsize xsputn(const char*     s,
                                       std::streamsize count) override;

        virtual std::streambuf::int_type underflow() override;
        virtual std::streamsize xsgetn(char* s, std::streamsize count) override;

       private:
        char           buffer[1]{0};
        LogDevice::Ptr device;
    };

    class LogStream : public std::ostream {
       public:
        LogStream(const std::string& prefix = "");
        ~LogStream();

       private:
        std::scoped_lock<Mutex> lock;
        static Mutex            mutex;
    };

    class Logger {
       public:
        using Ptr = std::shared_ptr<Logger>;
        Logger(const Name& name);

        LogStream operator()(const LogLevel& level = LogLevel::DEBUG) const;

        LogStream info() const;
        LogStream debug() const;
        LogStream warn() const;
        LogStream error() const;
        LogStream fatal() const;

        const Name& getName() const;

        static void clear();

       private:
        std::string format(const LogLevel& level) const;

        static std::string name2string(const Name& name);
        static std::string stamp2string(bool precise = true);
        static std::string level2string(const LogLevel& level);

        Name name;
    };

    class Logging {
        friend LogStream;

       public:
        static void setStream(const LogDevice::Ptr& stream);

        static void     setLevel(const LogLevel& level);
        static LogLevel getLevel();

       private:
        static StreamBuf stream;
        static LogLevel  level;
    };

};  // namespace corekit

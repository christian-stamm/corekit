#pragma once

#include <memory>
#include <mutex>
#include <ostream>
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
       public:
        StreamBuf() = default;

        std::streambuf::int_type overflow(std::streambuf::int_type c) override;
        std::streamsize xsputn(const char* s, std::streamsize count) override;
    };

    class LogStream : public std::ostream {
       public:
        LogStream(const std::string& prefix = "");
        ~LogStream();

       private:
        std::scoped_lock<Mutex> lock;
        static Mutex            mutex;
        static StreamBuf        buffer;
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
       public:
        static void           setDevice(const LogDevice::Ptr& device);
        static LogDevice::Ptr getDevice();

        static void     setLevel(const LogLevel& level);
        static LogLevel getLevel();

        static Logger::Ptr get(const Name& name);

       private:
        static LogDevice::Ptr device;
        static LogLevel       level;
    };

};  // namespace corekit

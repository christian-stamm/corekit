#pragma once

#include <iostream>
#include <memory>
#include <mutex>
#include <string>

#include "corekit/mutex.hpp"
#include "corekit/streambuffer.hpp"

namespace corekit {

    constexpr std::size_t NAME_SIZE  = 12;
    constexpr std::string CLEAR_CMD  = "\033[2J";
    constexpr std::string RED_CMD    = "\033[0;31m";
    constexpr std::string GREEN_CMD  = "\033[0;32m";
    constexpr std::string YELLOW_CMD = "\033[0;33m";
    constexpr std::string CYAN_CMD   = "\033[0;36m";
    constexpr std::string WHITE_CMD  = "\033[0;37m";
    constexpr std::string RESET_CMD  = "\033[0;39m";

    enum class LogLevel {
        FATAL = (1 << 0),
        ERROR = (1 << 1),
        WARN  = (1 << 2),
        INFO  = (1 << 3),
        DEBUG = (1 << 4),
    };

    class LogStream
        : public std::iostream
        , private std::scoped_lock<Mutex> {
        friend class Logging;

       public:
        LogStream(const std::string& prefix = "");
        ~LogStream();

       private:
        static std::string stamp2string();

        static Mutex mutex;
    };

    class Logger {
       public:
        using Ptr = std::shared_ptr<Logger>;
        Logger(const std::string& name);

        LogStream operator()(const LogLevel& level = LogLevel::DEBUG) const;

        LogStream info() const;
        LogStream debug() const;
        LogStream warn() const;
        LogStream error() const;
        LogStream fatal() const;

        static void clear();

        const std::string name;

       private:
        std::string format(const LogLevel& level) const;

        static std::string name2string(const std::string& name);
        static const char* level2string(const LogLevel& level);
    };

    class Logging {
       public:
        static void reconfigure(const StreamDevice::Ptr& output);

        static void     setLevel(const LogLevel& level);
        static LogLevel getLevel();

       private:
        static StreamBuffer stream;
        static LogLevel     level;
    };

};  // namespace corekit

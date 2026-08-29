#include "corekit/logger.hpp"

#include <chrono>
#include <format>
#include <iostream>
#include <mutex>
#include <string>

namespace corekit {

    // -------------------------------------------------------------------------
    // Static Members
    // -------------------------------------------------------------------------

    StreamBuffer Logging::stream(nullptr);
    LogLevel     Logging::level = LogLevel::DEBUG;
    Mutex        LogStream::mutex;

    // -------------------------------------------------------------------------
    // LogStream
    // -------------------------------------------------------------------------

    LogStream::LogStream(const std::string& prefix)
        : std::iostream(std::cout.rdbuf())
        , std::scoped_lock<Mutex>(mutex) {
        *this << stamp2string() << prefix;
    }

    LogStream::~LogStream() {
        *this << std::endl;
    }

    std::string LogStream::stamp2string() {
        static const auto ref = std::chrono::steady_clock::now();
        auto              dt  = std::chrono::steady_clock::now() - ref;

        const auto hrs = std::chrono::duration_cast<std::chrono::hours>(dt);
        dt -= hrs;

        const auto mns = std::chrono::duration_cast<std::chrono::minutes>(dt);
        dt -= mns;

        const auto secs = std::chrono::duration_cast<std::chrono::seconds>(dt);
        dt -= secs;

        const auto ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(dt);

        return std::format("[{:02}:{:02}:{:02}:{:02}.{:09}]",
                           hrs.count() / 24,
                           hrs.count() % 24,
                           mns.count(),
                           secs.count(),
                           ns.count());
    }

    // -------------------------------------------------------------------------
    // Logger
    // -------------------------------------------------------------------------

    Logger::Logger(const std::string& name) : name(name2string(name)) {}

    LogStream Logger::operator()(const LogLevel& level) const {
        return LogStream(format(level));
    }

    LogStream Logger::debug() const {
        return (*this)(LogLevel::DEBUG);
    }

    LogStream Logger::info() const {
        return (*this)(LogLevel::INFO);
    }

    LogStream Logger::warn() const {
        return (*this)(LogLevel::WARN);
    }

    LogStream Logger::error() const {
        return (*this)(LogLevel::ERROR);
    }

    LogStream Logger::fatal() const {
        return (*this)(LogLevel::FATAL);
    }

    std::string Logger::format(const LogLevel& level) const {
        return std::move(level2string(level) + name);
    }

    std::string Logger::name2string(const std::string& name) {
        std::string tf = name.substr(0, NAME_SIZE);
        std::transform(tf.begin(), tf.end(), tf.begin(), ::toupper);
        return std::format("[{:<{}}] ", tf, NAME_SIZE);
    }

    const char* Logger::level2string(const LogLevel& level) {
        static const std::string DEBUG_CODE =
            "[" + CYAN_CMD + "DEBUG" + RESET_CMD + "]";
        static const std::string INFO_CODE =
            "[" + GREEN_CMD + "INFO " + RESET_CMD + "]";
        static const std::string WARN_CODE =
            "[" + YELLOW_CMD + "WARN " + RESET_CMD + "]";
        static const std::string ERROR_CODE =
            "[" + RED_CMD + "ERROR" + RESET_CMD + "]";
        static const std::string FATAL_CODE =
            "[" + RED_CMD + "FATAL" + RESET_CMD + "]";
        static const std::string RESET_CODE =
            "[" + RESET_CMD + "UNKWN" + RESET_CMD + "]";

        switch (level) {
            case LogLevel::DEBUG: return DEBUG_CODE.c_str();
            case LogLevel::INFO: return INFO_CODE.c_str();
            case LogLevel::WARN: return WARN_CODE.c_str();
            case LogLevel::ERROR: return ERROR_CODE.c_str();
            case LogLevel::FATAL: return FATAL_CODE.c_str();
            default: return RESET_CODE.c_str();
        }
    }

    void Logger::clear() {
        std::system("clear");
    }

    // -------------------------------------------------------------------------
    // Logging
    // -------------------------------------------------------------------------

    void Logging::reconfigure(const StreamDevice::Ptr& output) {
        if (!output) {
            return;
        }

        if (!output->isLoaded()) {
            output->load();
        }

        std::scoped_lock lock(LogStream::mutex);
        stream = std::move(StreamBuffer(output));
        std::cout.rdbuf(&stream);
    }

    LogLevel Logging::getLevel() {
        return level;
    }

    void Logging::setLevel(const LogLevel& level) {
        Logging::level = level;
    }

}  // namespace corekit
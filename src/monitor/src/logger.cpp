#include "corekit/logger.hpp"

#include <chrono>
#include <cstdint>
#include <format>
#include <string>

namespace corekit {

    // -------------------------------------------------------------------------
    // Constants
    // -------------------------------------------------------------------------

    constexpr std::size_t NAME_SIZE  = 24;
    constexpr std::string CLEAR_CMD  = "\033[2J";
    constexpr std::string RED_CMD    = "\033[0;31m";
    constexpr std::string GREEN_CMD  = "\033[0;32m";
    constexpr std::string YELLOW_CMD = "\033[0;33m";
    constexpr std::string CYAN_CMD   = "\033[0;36m";
    constexpr std::string WHITE_CMD  = "\033[0;37m";
    constexpr std::string RESET_CMD  = "\033[0;39m";

    // -------------------------------------------------------------------------
    // Static Members
    // -------------------------------------------------------------------------

    LogDevice::Ptr Logging::device = nullptr;
    LogLevel       Logging::level  = LogLevel::DEBUG;
    Mutex          LogStream::mutex;
    StreamBuf      LogStream::buffer;

    // -------------------------------------------------------------------------
    // StreamBuf
    // -------------------------------------------------------------------------

    std::streambuf::int_type StreamBuf::overflow(std::streambuf::int_type c) {
        LogDevice::Ptr dev = Logging::getDevice();

        if (c != EOF && dev != nullptr) {
            dev->write(static_cast<uint8_t>(c));
            return std::streambuf::traits_type::to_int_type(c);
        }

        return EOF;
    }

    std::streamsize StreamBuf::xsputn(const char* s, std::streamsize count) {
        LogDevice::Ptr dev = Logging::getDevice();

        if (dev != nullptr) {
            const auto msg = std::string(s, count);

            if (dev->writeBulk(std::span<uint8_t>(
                    reinterpret_cast<uint8_t*>(const_cast<char*>(msg.data())),
                    static_cast<std::size_t>(msg.size())))) {
                return count;
            }
        }

        return EOF;
    }

    // -------------------------------------------------------------------------
    // LogStream
    // -------------------------------------------------------------------------

    LogStream::LogStream(const std::string& prefix)
        : std::ostream(&buffer)
        , lock(mutex) {
        if (!prefix.empty()) {
            *this << prefix;
        }
    }

    LogStream::~LogStream() {
        *this << std::endl;
    }

    // -------------------------------------------------------------------------
    // Logger
    // -------------------------------------------------------------------------

    Logger::Logger(const Name& name) : name(name2string(name)) {}

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
        return std::format("[{}][{}][{:<{}}] ",
                           level2string(level),
                           stamp2string(),
                           name,
                           NAME_SIZE);
    }

    std::string Logger::name2string(const Name& name) {
        std::string tf = name.substr(0, NAME_SIZE);
        std::transform(tf.begin(), tf.end(), tf.begin(), [](unsigned char c) {
            return std::toupper(c);
        });
        return tf;
    }

    std::string Logger::stamp2string(bool precise) {
        auto     t = std::chrono::system_clock::now().time_since_epoch();
        uint32_t h =
            std::chrono::duration_cast<std::chrono::hours>(t).count() % 24;
        uint32_t m =
            std::chrono::duration_cast<std::chrono::minutes>(t).count() % 60;
        uint32_t s =
            std::chrono::duration_cast<std::chrono::seconds>(t).count() % 60;
        uint32_t us =
            std::chrono::duration_cast<std::chrono::microseconds>(t).count() %
            1000000;
        return std::format("{:02}:{:02}:{:02}", h, m, s) +
               (precise ? std::format(":{:06}", us) : "");
    }

    std::string Logger::level2string(const LogLevel& level) {
        switch (level) {
            case LogLevel::DEBUG: return CYAN_CMD + "DEBUG" + RESET_CMD;
            case LogLevel::INFO: return GREEN_CMD + "INFO " + RESET_CMD;
            case LogLevel::WARN: return YELLOW_CMD + "WARN " + RESET_CMD;
            case LogLevel::ERROR: return RED_CMD + "ERROR" + RESET_CMD;
            case LogLevel::FATAL: return RED_CMD + "FATAL" + RESET_CMD;
            default: return RESET_CMD + "UNKWN" + RESET_CMD;
        }
    }

    const Name& Logger::getName() const {
        return name;
    }

    void Logger::clear() {
        std::system("clear");
    }

    // -------------------------------------------------------------------------
    // Logging
    // -------------------------------------------------------------------------

    LogDevice::Ptr Logging::getDevice() {
        return device;
    }

    void Logging::setDevice(const LogDevice::Ptr& device) {
        Logging::device = device;
    }

    LogLevel Logging::getLevel() {
        return level;
    }

    void Logging::setLevel(const LogLevel& level) {
        Logging::level = level;
    }

}  // namespace corekit
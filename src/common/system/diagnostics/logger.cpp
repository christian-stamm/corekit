#include "corekit/system/diagnostics/logger.hpp"

#include <chrono>
#include <cstdint>
#include <format>
#include <string>

namespace corekit {
    namespace system {
        namespace diagnostics {

            constexpr std::size_t NAME_SIZE  = 8;
            constexpr std::string CLEAR_CMD  = "\033[2J";
            constexpr std::string RED_CMD    = "\033[0;31m";
            constexpr std::string GREEN_CMD  = "\033[0;32m";
            constexpr std::string YELLOW_CMD = "\033[0;33m";
            constexpr std::string CYAN_CMD   = "\033[0;36m";
            constexpr std::string WHITE_CMD  = "\033[0;37m";
            constexpr std::string RESET_CMD  = "\033[0;39m";

            Logger::Logger(const Name& name) : name(name2string(name)) {}

            Logstream Logger::operator()(const Level& level) const {
                return Logstream(format(level));
            }

            Logstream Logger::debug() const {
                return (*this)(Level::DEBUG);
            }

            Logstream Logger::info() const {
                return (*this)(Level::INFO);
            }

            Logstream Logger::warn() const {
                return (*this)(Level::WARN);
            }

            Logstream Logger::error() const {
                return (*this)(Level::ERROR);
            }

            Logstream Logger::fatal() const {
                return (*this)(Level::FATAL);
            }

            std::string Logger::format(const Level& level) const {
                return std::format("[{}][{}][{:<{}}] ",
                                   level2string(level),
                                   stamp2string(),
                                   name,
                                   NAME_SIZE);
            }

            std::string Logger::name2string(const Name& name) {
                std::string tf = name.substr(0, NAME_SIZE);
                std::transform(tf.begin(),
                               tf.end(),
                               tf.begin(),
                               [](unsigned char c) { return std::toupper(c); });
                return tf;
            }

            std::string Logger::stamp2string(bool precise) {
                auto t = std::chrono::system_clock::now().time_since_epoch();
                uint32_t h =
                    std::chrono::duration_cast<std::chrono::hours>(t).count() %
                    24;
                uint32_t m = std::chrono::duration_cast<std::chrono::minutes>(t)
                                 .count() %
                             60;
                uint32_t s = std::chrono::duration_cast<std::chrono::seconds>(t)
                                 .count() %
                             60;
                uint32_t us =
                    std::chrono::duration_cast<std::chrono::microseconds>(t)
                        .count() %
                    1000000;
                return std::format("{:02}:{:02}:{:02}", h, m, s) +
                       (precise ? std::format(":{:06}", us) : "");
            }

            std::string Logger::level2string(const Level& level) {
                switch (level) {
                    case Level::DEBUG: return CYAN_CMD + "DEBUG" + RESET_CMD;
                    case Level::INFO: return GREEN_CMD + "INFO " + RESET_CMD;
                    case Level::WARN: return YELLOW_CMD + "WARN " + RESET_CMD;
                    case Level::ERROR: return RED_CMD + "ERROR" + RESET_CMD;
                    case Level::FATAL: return RED_CMD + "FATAL" + RESET_CMD;
                    default: return RESET_CMD + "UNKWN" + RESET_CMD;
                }
            }

            const Name& Logger::getName() const {
                return name;
            }

            void Logger::clear() {
                std::system("clear");
            }

        };  // namespace diagnostics
    };      // namespace system
};          // namespace corekit
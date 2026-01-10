#pragma once

#include <memory>
#include <string>

#include "corekit/logging/stream.hpp"

namespace corekit {
    namespace logging {

        enum class Level {
            FATAL = (1 << 0),
            ERROR = (1 << 1),
            WARN  = (1 << 2),
            INFO  = (1 << 3),
            DEBUG = (1 << 4),
        };

        class Logger {
           public:
            using Ptr = std::shared_ptr<Logger>;
            Logger(const std::string& name);

            Logstream operator()(const Level& level = Level::DEBUG) const;

            Logstream info() const;
            Logstream debug() const;
            Logstream warn() const;
            Logstream error() const;
            Logstream fatal() const;

            std::string getName() const;

           private:
            std::string format(const Level& level) const;

            static std::string name2string(const std::string& name);
            static std::string stamp2string(bool precise = true);
            static std::string level2string(const Level& level);

            std::string name;
        };

    };  // namespace logging
};  // namespace corekit
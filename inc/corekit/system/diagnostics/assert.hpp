#pragma once

#include <source_location>
#include <string>

namespace corekit {
    namespace system {
        namespace diagnostics {

            using Location = std::source_location;

            void corecheck(bool               condition,
                           const std::string& message  = "<NO DESCRIPTION>",
                           const Location&    location = Location::current());

            void glCheckError(std::string     message  = "<NO DESCRIPTION>",
                              const Location& location = Location::current());

        };  // namespace diagnostics
    };  // namespace system
};  // namespace corekit

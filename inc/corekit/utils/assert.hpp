#pragma once

#include <source_location>
#include <string>

namespace corekit {
    namespace utils {

        using Location = std::source_location;

        void corecheck(bool               condition,
                       const std::string& message  = "<NO DESCRIPTION>",
                       const Location&    location = Location::current());

        void glCheckError(std::string     message  = "<NO DESCRIPTION>",
                          const Location& location = Location::current());

    };  // namespace utils
};      // namespace corekit

#pragma once

#include <source_location>
#include <string>

#include "corekit/types.hpp"

namespace corekit {
    namespace utils {

        using namespace corekit::types;

        using Location = std::source_location;

        void corecheck(bool            condition,
                       const Status&   message  = "<NO DESCRIPTION>",
                       const Location& location = Location::current());

    };  // namespace utils
};      // namespace corekit

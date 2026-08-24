#pragma once

#include <source_location>
#include <string>

#include "corekit/types.hpp"

namespace corekit {

    void corecheck(bool            condition,
                   const Status&   message  = "<NO DESCRIPTION>",
                   const Location& location = std::source_location::current());

};  // namespace corekit

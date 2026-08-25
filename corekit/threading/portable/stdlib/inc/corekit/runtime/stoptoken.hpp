#pragma once
#include <stop_token>

namespace corekit::runtime {

    using StopToken  = std::stop_token;
    using StopSource = std::stop_source;

}  // namespace corekit::runtime
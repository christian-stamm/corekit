#pragma once
#include "corekit/error.hpp"

namespace corekit {

    extern bool corecheck();
    extern bool corecheck(bool condition, const Error& error);

}  // namespace corekit
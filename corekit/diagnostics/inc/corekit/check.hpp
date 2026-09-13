#pragma once
#include "corekit/error.hpp"

namespace corekit::core {

    extern void verify();
    extern bool ok();
    extern bool check(bool condition, const Error& error = RuntimeError());

} // namespace corekit::core
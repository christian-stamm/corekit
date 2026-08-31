#pragma once
#include "corekit/error.hpp"

namespace corekit {

    inline bool corecheck(bool condition, const Error& error = RuntimeError()) {
        // Perform a core check to ensure that the system is in a valid state.

        // #ifndef NDEBUG

        if (!condition) {
            std::cout << std::format("Core check failed: {}", error.what()) << std::endl;

            std::terminate();
        }

        // #endif
        return true;
    }

} // namespace corekit
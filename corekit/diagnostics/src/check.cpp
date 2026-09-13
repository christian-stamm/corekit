#include "corekit/check.hpp"

#include "corekit/atomic.hpp"

namespace corekit::core {

    Atomic<bool> ok_{true};

    bool ok() { return ok_.load(); }

    bool check(bool condition, const Error& error) {
        // #ifndef NDEBUG

        if (!condition) {
            std::cout << std::format("Core check failed: {}", error.what()) << std::endl;
            ok_.store(false);
        }

        // #endif
        return ok();
    }

    void verify() { check(ok(), RuntimeError("Verification Failed")); }

} // namespace corekit::core
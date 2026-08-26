#include "corekit/assert.hpp"

namespace corekit {

    VoidResult corecheck(bool condition, const Error& error) {
#ifndef NDEBUG
        if (!condition) {
            return VoidResult(error);
        }
#endif
        return VoidResult();
    }

}  // namespace corekit
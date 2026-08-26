#include "corekit/assert.hpp"

namespace corekit {

    Result<void> corecheck(bool condition, const Error& error) {
#ifndef NDEBUG
        if (!condition) {
            return std::unexpected(error);
        }
#endif
        return Result<void>(std::in_place);
    }

}  // namespace corekit
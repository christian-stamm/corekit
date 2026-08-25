#include "corekit/atomic.hpp"

namespace corekit {

    static_assert(AtomicLike<Atomic<bool>, bool>);

};  // namespace corekit
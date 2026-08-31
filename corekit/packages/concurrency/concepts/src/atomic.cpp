#include "corekit/atomic.hpp"

namespace corekit {

    static_assert(AtomicLike<Atomic<bool>, bool>);
    static_assert(AtomicIntLike<Atomic<int>, int>);

}; // namespace corekit
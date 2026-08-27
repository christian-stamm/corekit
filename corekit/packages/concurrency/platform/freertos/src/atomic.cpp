#include "corekit/platform/atomic.hpp"

namespace corekit::platform {

    template class Atomic<bool>;
    template class Atomic<uint>;
    template class Atomic<int>;

}  // namespace corekit::platform
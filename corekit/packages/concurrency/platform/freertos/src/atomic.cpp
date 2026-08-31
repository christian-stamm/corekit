#include "corekit/platform/atomic.hpp"

#include <cstdint>

namespace corekit::platform {

    template class Atomic<bool>;
    template class Atomic<unsigned int>;
    template class Atomic<int>;

} // namespace corekit::platform

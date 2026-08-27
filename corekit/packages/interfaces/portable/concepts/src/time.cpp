#include "corekit/time.hpp"

namespace corekit {

    static_assert(Timeable<Time>,
                  "Time does not provide the required interface");

}  // namespace corekit
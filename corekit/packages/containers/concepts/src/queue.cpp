#include "corekit/queue.hpp"

namespace corekit {

    static_assert(QueueLike<Queue<bool>, bool>);

};  // namespace corekit
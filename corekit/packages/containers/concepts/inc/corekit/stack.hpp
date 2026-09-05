#pragma once

#include "corekit/platform/stack.hpp"
#include "corekit/queue.hpp"

namespace corekit {

    template <typename T>
    using Stack = platform::Stack<T>;

    template <typename Stack, typename Item>
    concept StackLike = QueueLike<Stack, Item>;

}  // namespace corekit
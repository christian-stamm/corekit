#pragma once

#include <concepts>

#include "corekit/platform/queue.hpp"

namespace corekit {

    template <typename T>
    using Queue = platform::Queue<T>;

    template <typename Queue, typename Item>
    concept QueueLike = requires(Queue q, Item i) {
        { q.push(i, false) } -> std::convertible_to<bool>;
        { q.pop(i, false) } -> std::convertible_to<bool>;
    };

}  // namespace corekit
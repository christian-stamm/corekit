#pragma once

#include <concepts>

#include "corekit/platform/queue.hpp"
#include "corekit/result.hpp"

namespace corekit {

    template <typename T>
    using Queue = platform::Queue<T>;

    template <typename Queue, typename Item>
    concept QueueLike = requires(Queue q, Item i) {
        { q.push(i, false) } -> std::convertible_to<VoidResult>;
        { q.pop(i, false) } -> std::convertible_to<Result<Item>>;
    };

}  // namespace corekit
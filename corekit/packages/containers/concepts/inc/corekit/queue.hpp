#pragma once

#include <concepts>

#include "corekit/platform/queue.hpp"
#include "corekit/result.hpp"

namespace corekit {

    template <typename T>
    using Queue = platform::Queue<T>;

    template <typename Queue, typename Item>
    concept QueueLike = requires(Queue q, Item i) {
        { q.push(i) } -> std::convertible_to<bool>;
        { q.pop(i) } -> std::convertible_to<bool>;
        { q.try_push(i) } -> std::convertible_to<bool>;
        { q.try_pop(i) } -> std::convertible_to<bool>;
    };

} // namespace corekit
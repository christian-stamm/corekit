#pragma once

#include "corekit/platform/atomic.hpp"

namespace corekit {

    template <typename T>
    using Atomic = platform::Atomic<T>;

    template <typename T, typename V>
    concept AtomicLike = requires(T a, V value) {
        { a.load() } -> std::convertible_to<V>;
        { a.store(value) } -> std::convertible_to<void>;
        { a.compare_exchange(value, value) } -> std::convertible_to<bool>;
    };

    static_assert(AtomicLike<Atomic<bool>, bool>);

};  // namespace corekit
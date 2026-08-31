#pragma once

#include <concepts>

#include "corekit/platform/atomic.hpp"

namespace corekit {

    template <typename T>
    using Atomic = platform::Atomic<T>;

    template <typename Atomic, typename T>
    concept AtomicLike = requires(const Atomic& catomic, Atomic& atomic, T desired, T expected) {
        { catomic.load() } -> std::convertible_to<T>;
        { atomic.store(desired) } -> std::same_as<void>;
        { atomic.exchange(desired) } -> std::convertible_to<T>;
        { atomic.compare_exchange(expected, desired) } -> std::convertible_to<bool>;
    };

    template <typename Atomic, typename T>
    concept AtomicIntLike = AtomicLike<Atomic, T> && requires(Atomic& atomic) {
        { atomic.fetch_add(1) } -> std::convertible_to<T>;
        { atomic.fetch_sub(1) } -> std::convertible_to<T>;
        { atomic.fetch_or(1) } -> std::convertible_to<T>;
        { atomic.fetch_and(1) } -> std::convertible_to<T>;
        { atomic.fetch_xor(1) } -> std::convertible_to<T>;
    };

} // namespace corekit

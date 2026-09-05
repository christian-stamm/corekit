#pragma once

#include "corekit/platform/atomic.hpp"

namespace corekit {

    template <typename T>
    using Atomic = platform::Atomic<T>;

    template <typename T>
    concept AtomicLike = requires(  //
        const Atomic<T> &catomic,
        Atomic<T>        atomic,
        T                desired,
        T                expected  //
    ) {
        { catomic.load() } -> std::convertible_to<T>;
        { atomic.store(desired) } -> std::convertible_to<void>;
        {
            atomic.compare_exchange(expected, desired)
        } -> std::convertible_to<bool>;
    };

};  // namespace corekit
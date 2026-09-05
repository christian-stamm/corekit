#pragma once

#include "corekit/platform/atomic.hpp"  // IWYU pragma: keep

namespace corekit {

    template <typename T>
    using Atomic = platform::Atomic<T>;

    template <typename AtomicT, typename V>
    concept AtomicLike = requires(  //
        const AtomicT &catomic,
        AtomicT        atomic,
        V              desired,
        V              expected  //
    ) {
        { catomic.load() } -> std::convertible_to<V>;
        { atomic.store(desired) } -> std::convertible_to<void>;
        {
            atomic.compare_exchange(expected, desired)
        } -> std::convertible_to<bool>;
    };

};  // namespace corekit
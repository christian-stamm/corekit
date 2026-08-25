#pragma once

#include "corekit/runtime/semaphore.hpp"

namespace corekit {

    using Semaphore = runtime::Semaphore;

    template <typename T>
    concept SemaphoreLike = requires(T s) {
        { s.acquire() } -> std::convertible_to<void>;
        { s.release() } -> std::convertible_to<void>;
        { s.try_acquire() } -> std::convertible_to<bool>;
    };

    static_assert(SemaphoreLike<Semaphore>);

};  // namespace corekit
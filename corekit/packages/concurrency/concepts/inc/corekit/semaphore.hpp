#pragma once

#include "corekit/platform/semaphore.hpp"  // IWYU pragma: keep

namespace corekit {

    using Semaphore = platform::Semaphore;

    template <typename T>
    concept SemaphoreLike = requires(T s) {
        { s.acquire() } -> std::convertible_to<void>;
        { s.release() } -> std::convertible_to<void>;
        { s.try_acquire() } -> std::convertible_to<bool>;
    };

};  // namespace corekit
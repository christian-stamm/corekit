#pragma once

#include "corekit/platform/semaphore.hpp"

namespace corekit {

    using Semaphore = platform::Semaphore;

    template <typename Sem = Semaphore>
    concept SemaphoreLike = requires(Sem s) {
        { s.acquire() } -> std::convertible_to<void>;
        { s.release() } -> std::convertible_to<void>;
        { s.try_acquire() } -> std::convertible_to<bool>;
    };

};  // namespace corekit
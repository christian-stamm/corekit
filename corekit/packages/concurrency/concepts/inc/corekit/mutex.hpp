#pragma once

#include "corekit/platform/mutex.hpp"  // IWYU pragma: keep

namespace corekit {

    using Mutex = platform::Mutex;

    template <typename T>
    concept MutexLike = requires(T m) {
        { m.lock() } -> std::convertible_to<void>;
        { m.unlock() } -> std::convertible_to<void>;
        { m.try_lock() } -> std::convertible_to<bool>;
    };

};  // namespace corekit
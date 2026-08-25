#pragma once

#include "corekit/concepts/mutex.hpp"
#include "corekit/platform/mutex.hpp"

namespace corekit {

    using Mutex = platform::Mutex;

    template <typename T>
    concept LockLike = requires(T m) {
        { m.lock() } -> std::convertible_to<void>;
        { m.unlock() } -> std::convertible_to<void>;
    };

    template <typename T>
    concept MutexLike =  //
        LockLike<T> && requires(T m) {
            { m.try_lock() } -> std::convertible_to<bool>;
        };

    static_assert(MutexLike<Mutex>);

};  // namespace corekit
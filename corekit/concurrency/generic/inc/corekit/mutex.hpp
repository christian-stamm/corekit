#pragma once

#include "corekit/runtime/mutex.hpp"

namespace corekit {

    using Mutex = runtime::Mutex;

    template <typename T>
    concept MutexLike =  //
        requires(T m) {
            { m.lock() } -> std::convertible_to<void>;
            { m.unlock() } -> std::convertible_to<void>;
            { m.try_lock() } -> std::convertible_to<bool>;
        };

};  // namespace corekit
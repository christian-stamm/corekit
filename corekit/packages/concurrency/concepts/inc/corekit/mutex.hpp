#pragma once

#include "corekit/platform/mutex.hpp"

namespace corekit {

    using Mutex = platform::Mutex;

    template <typename Mtx = Mutex>
    concept MutexLike =  //
        requires(Mtx m) {
            { m.lock() } -> std::convertible_to<void>;
            { m.unlock() } -> std::convertible_to<void>;
            { m.try_lock() } -> std::convertible_to<bool>;
        };

};  // namespace corekit
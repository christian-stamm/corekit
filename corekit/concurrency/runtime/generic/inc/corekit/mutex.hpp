#pragma once

#include "corekit/mutex.hpp"

namespace corekit {

    template <typename T>
    concept MutexLike =  //
        requires(T m) {
            { m.lock() } -> std::convertible_to<void>;
            { m.unlock() } -> std::convertible_to<void>;
            { m.try_lock() } -> std::convertible_to<bool>;
        };

    static_assert(MutexLike<Mutex>);

};  // namespace corekit
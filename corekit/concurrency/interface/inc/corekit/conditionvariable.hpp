#pragma once

#include "corekit/mutex.hpp"
#include "corekit/platform/conditionvariable.hpp"

namespace corekit {

    using ConditionVariable = platform::ConditionVariable;

    template <typename T, typename Lock>
    concept ConditionVariableLike =  //
        LockLike<Lock> && requires(T cv, Lock& lock) {
            { cv.wait(lock) } -> std::convertible_to<void>;
            { cv.notify_one() } -> std::convertible_to<void>;
            { cv.notify_all() } -> std::convertible_to<void>;
        };

    static_assert(ConditionVariableLike<ConditionVariable, Mutex>);

};  // namespace corekit
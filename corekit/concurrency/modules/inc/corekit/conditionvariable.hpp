#pragma once

#include <mutex>

#include "corekit/mutex.hpp"
#include "corekit/runtime/conditionvariable.hpp"

namespace corekit {

    using ConditionVariable = runtime::ConditionVariable;

    template <typename T, typename Lock>
    concept ConditionVariableLike =  //
        requires(T cv, Lock& lock) {
            { cv.wait(lock) } -> std::convertible_to<void>;
            { cv.notify_one() } -> std::convertible_to<void>;
            { cv.notify_all() } -> std::convertible_to<void>;
        };

    static_assert(
        ConditionVariableLike<ConditionVariable, std::unique_lock<Mutex>>);

};  // namespace corekit
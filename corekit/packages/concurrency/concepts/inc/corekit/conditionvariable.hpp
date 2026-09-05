#pragma once

#include <mutex>

#include "corekit/mutex.hpp"
#include "corekit/platform/conditionvariable.hpp"

namespace corekit {

    using ConditionVariable = platform::ConditionVariable;

    template <typename Predicate>               //
    concept ConditionVariableLike =             //
        std::predicate<Predicate> &&            //
        requires(                               //
            ConditionVariable        cv,        //
            std::unique_lock<Mutex>& lock,      //
            Predicate                predicate  //
        ) {
            { cv.wait(lock) } -> std::convertible_to<void>;
            { cv.wait(lock, predicate) } -> std::convertible_to<void>;
            { cv.notify_one() } -> std::convertible_to<void>;
            { cv.notify_all() } -> std::convertible_to<void>;
        };

};  // namespace corekit
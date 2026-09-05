#pragma once

#include <mutex>

#include "corekit/platform/conditionvariable.hpp"  // IWYU pragma: keep

namespace corekit {

    using ConditionVariable = platform::ConditionVariable;

    template <typename T, typename Lock, typename Predicate>

    concept ConditionVariableLike =
        std::predicate<Predicate> &&
        requires(T cv, std::unique_lock<Lock> &lock, Predicate predicate) {
            { cv.wait(lock) } -> std::convertible_to<void>;
            { cv.wait(lock, predicate) } -> std::convertible_to<void>;
            { cv.notify_one() } -> std::convertible_to<void>;
            { cv.notify_all() } -> std::convertible_to<void>;
        };

};  // namespace corekit
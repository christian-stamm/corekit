#include "corekit/conditionvariable.hpp"

#include <gtest/gtest.h>

#include "corekit/mutex.hpp"

namespace corekit::test {

    /*
     * Match std::condition_variable semantics.
     *
     * Remove if your implementation intentionally
     * supports copying or moving.
     */
    static_assert(!std::is_copy_constructible_v<ConditionVariable>);
    static_assert(!std::is_copy_assignable_v<ConditionVariable>);

    static_assert(!std::is_move_constructible_v<ConditionVariable>);
    static_assert(!std::is_move_assignable_v<ConditionVariable>);

    TEST(ConditionVariableTest, DefaultConstruction) {
        ConditionVariable cv;

        SUCCEED();
    }

    TEST(ConditionVariableTest, NotifyOneCompiles) {
        ConditionVariable cv;

        cv.notify_one();

        SUCCEED();
    }

    TEST(ConditionVariableTest, NotifyAllCompiles) {
        ConditionVariable cv;

        cv.notify_all();

        SUCCEED();
    }

    TEST(ConditionVariableTest, NotifyOneCanBeCalledRepeatedly) {
        ConditionVariable cv;

        for (int i = 0; i < 1000; ++i) {
            cv.notify_one();
        }

        SUCCEED();
    }

    TEST(ConditionVariableTest, NotifyAllCanBeCalledRepeatedly) {
        ConditionVariable cv;

        for (int i = 0; i < 1000; ++i) {
            cv.notify_all();
        }

        SUCCEED();
    }

    TEST(ConditionVariableTest, WorksWithUniqueLockType) {
        Mutex mutex;

        std::unique_lock<Mutex> lock(mutex);

        ConditionVariable cv;

        (void)cv;
        (void)lock;

        SUCCEED();
    }

}  // namespace corekit::test
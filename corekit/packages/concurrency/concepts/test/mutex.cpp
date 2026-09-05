#include "corekit/mutex.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace corekit::test {

    /*
     * A normal mutex must not be copyable or movable.
     *
     * Moving a mutex while another thread may be using it would invalidate
     * synchronization state. These assertions match std::mutex semantics.
     */
    static_assert(!std::is_copy_constructible_v<Mutex>);
    static_assert(!std::is_copy_assignable_v<Mutex>);
    static_assert(!std::is_move_constructible_v<Mutex>);
    static_assert(!std::is_move_assignable_v<Mutex>);

    /*
     * Mutex must be usable with the standard C++ lock wrappers.
     */
    static_assert(std::is_constructible_v<std::lock_guard<Mutex>, Mutex &>);
    static_assert(std::is_constructible_v<std::unique_lock<Mutex>, Mutex &>);

    /*
     * Basic lock/unlock behavior.
     */

    TEST(Mutex, LockAndUnlockCompletes) {
        Mutex mutex;

        mutex.lock();
        mutex.unlock();

        SUCCEED();
    }

    TEST(Mutex, MutexCanBeLockedRepeatedlyAfterUnlocking) {
        Mutex mutex;

        for (int iteration = 0; iteration < 1'000; ++iteration) {
            mutex.lock();
            mutex.unlock();
        }

        SUCCEED();
    }

    TEST(Mutex, TryLockSucceedsWhenMutexIsUnlocked) {
        Mutex mutex;

        const bool acquired = mutex.try_lock();

        EXPECT_TRUE(acquired);

        if (acquired) {
            mutex.unlock();
        }
    }

    TEST(Mutex, TryLockSucceedsAgainAfterUnlock) {
        Mutex mutex;

        ASSERT_TRUE(mutex.try_lock());
        mutex.unlock();

        ASSERT_TRUE(mutex.try_lock());
        mutex.unlock();
    }

    TEST(Mutex, TryLockCanBeRepeatedAfterEachUnlock) {
        Mutex mutex;

        for (int iteration = 0; iteration < 1'000; ++iteration) {
            ASSERT_TRUE(mutex.try_lock())
                << "try_lock failed at iteration " << iteration;

            mutex.unlock();
        }
    }

    TEST(Mutex, LockWorksAfterSuccessfulTryLockAndUnlock) {
        Mutex mutex;

        ASSERT_TRUE(mutex.try_lock());
        mutex.unlock();

        mutex.lock();
        mutex.unlock();

        SUCCEED();
    }

    TEST(Mutex, TryLockWorksAfterLockAndUnlock) {
        Mutex mutex;

        mutex.lock();
        mutex.unlock();

        ASSERT_TRUE(mutex.try_lock());
        mutex.unlock();
    }

    TEST(Mutex, LockWorksAfterTryLockFails) {
        Mutex mutex;

        ASSERT_TRUE(mutex.try_lock());
        ASSERT_FALSE(mutex.try_lock());

        mutex.unlock();

        mutex.lock();
        mutex.unlock();

        SUCCEED();
    }

    TEST(Mutex, TryLockWorksAfterLockFails) {
        Mutex mutex;

        mutex.lock();

        ASSERT_FALSE(mutex.try_lock());
        mutex.unlock();

        ASSERT_TRUE(mutex.try_lock());
        mutex.unlock();
    }

}  // namespace corekit::test
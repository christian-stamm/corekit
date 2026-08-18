#include "corekit/semaphore.hpp"

#include <gtest/gtest.h>

namespace corekit::test {

    /*
     * Standard semaphore objects generally should not be copyable.
     *
     * Remove these assertions if your implementation intentionally
     * supports copy or move semantics.
     */
    static_assert(!std::is_copy_constructible_v<Semaphore>);
    static_assert(!std::is_copy_assignable_v<Semaphore>);
    static_assert(!std::is_move_constructible_v<Semaphore>);
    static_assert(!std::is_move_assignable_v<Semaphore>);

    TEST(Semaphore, ReleaseMakesTryAcquireSucceed) {
        Semaphore semaphore;

        semaphore.release();

        EXPECT_TRUE(semaphore.try_acquire());
    }

    TEST(Semaphore, SingleReleaseProvidesSinglePermit) {
        Semaphore semaphore;

        semaphore.release();

        EXPECT_TRUE(semaphore.try_acquire());
        EXPECT_FALSE(semaphore.try_acquire());
    }

    TEST(Semaphore, MultipleReleasesProvideMultiplePermits) {
        Semaphore semaphore;

        semaphore.release();
        semaphore.release();
        semaphore.release();

        EXPECT_TRUE(semaphore.try_acquire());
        EXPECT_TRUE(semaphore.try_acquire());
        EXPECT_TRUE(semaphore.try_acquire());

        EXPECT_FALSE(semaphore.try_acquire());
    }

    TEST(Semaphore, AcquireConsumesPermit) {
        Semaphore semaphore;

        semaphore.release();

        semaphore.acquire();

        EXPECT_FALSE(semaphore.try_acquire());
    }

    TEST(Semaphore, AcquireConsumesOnlyOnePermit) {
        Semaphore semaphore;

        semaphore.release();
        semaphore.release();

        semaphore.acquire();

        EXPECT_TRUE(semaphore.try_acquire());
        EXPECT_FALSE(semaphore.try_acquire());
    }

    TEST(Semaphore, PermitCanBeReusedAfterAcquire) {
        Semaphore semaphore;

        semaphore.release();

        semaphore.acquire();

        EXPECT_FALSE(semaphore.try_acquire());

        semaphore.release();

        EXPECT_TRUE(semaphore.try_acquire());
        EXPECT_FALSE(semaphore.try_acquire());
    }

    TEST(Semaphore, AlternatingReleaseAndAcquireWorks) {
        Semaphore semaphore;

        for (int i = 0; i < 1000; ++i) {
            semaphore.release();

            EXPECT_TRUE(semaphore.try_acquire());
            EXPECT_FALSE(semaphore.try_acquire());
        }
    }

    TEST(Semaphore, ConsecutiveReleasesAccumulatePermits) {
        Semaphore semaphore;

        constexpr int permit_count = 100;

        for (int i = 0; i < permit_count; ++i) {
            semaphore.release();
        }

        for (int i = 0; i < permit_count; ++i) {
            EXPECT_TRUE(semaphore.try_acquire());
        }

        EXPECT_FALSE(semaphore.try_acquire());
    }

    TEST(Semaphore, ReleasingAfterExhaustionCreatesNewPermit) {
        Semaphore semaphore;

        semaphore.release();

        EXPECT_TRUE(semaphore.try_acquire());
        EXPECT_FALSE(semaphore.try_acquire());

        semaphore.release();

        EXPECT_TRUE(semaphore.try_acquire());
        EXPECT_FALSE(semaphore.try_acquire());
    }

}  // namespace corekit::test
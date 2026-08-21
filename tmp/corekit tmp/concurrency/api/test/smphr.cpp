// #include <gtest/gtest.h>

// #include "corekit/semaphore.hpp"

// namespace corekit {

//     //
//     -------------------------------------------------------------------------
//     // Acquire and Release
//     //
//     -------------------------------------------------------------------------

//     TEST(Semaphore, CanAcquireAndRelease) {
//         Semaphore semaphore(1);

//         EXPECT_TRUE(semaphore.try_acquire());
//         EXPECT_FALSE(semaphore.try_acquire());
//         semaphore.release();
//         EXPECT_TRUE(semaphore.try_acquire());
//         semaphore.release();
//     }

//     //
//     -------------------------------------------------------------------------
//     // Try Acquire
//     //
//     -------------------------------------------------------------------------
//     TEST(Semaphore, TryAcquireReturnsFalseWhenCountIsZero) {
//         Semaphore semaphore(0);

//         EXPECT_FALSE(semaphore.try_acquire());
//     }

//     TEST(Semaphore, TryAcquireReturnsTrueWhenCountIsGreaterThanZero) {
//         Semaphore semaphore(1);

//         EXPECT_TRUE(semaphore.try_acquire());
//         EXPECT_FALSE(semaphore.try_acquire());
//     }

// }  // namespace corekit
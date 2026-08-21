// #include "corekit/mutex.hpp"

// #include <gtest/gtest.h>

// namespace corekit {

//     //
//     -------------------------------------------------------------------------
//     // Lock and Unlock
//     //
//     -------------------------------------------------------------------------

//     TEST(Mutex, CanLockAndUnlock) {
//         Mutex mutex;

//         mutex.lock();
//         EXPECT_FALSE(mutex.try_lock());
//         mutex.unlock();
//         EXPECT_TRUE(mutex.try_lock());
//         mutex.unlock();
//     }

//     //
//     -------------------------------------------------------------------------
//     // Try Lock
//     //
//     -------------------------------------------------------------------------
//     TEST(Mutex, TryLock) {
//         Mutex mutex;

//         EXPECT_TRUE(mutex.try_lock());
//         EXPECT_FALSE(mutex.try_lock());
//         mutex.unlock();
//         EXPECT_TRUE(mutex.try_lock());
//         mutex.unlock();
//     }

// }  // namespace corekit
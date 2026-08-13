#include <gtest/gtest.h>

#include "corekit/platform.hpp"

namespace corekit {

    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    TEST(Atomic, StartsAtZero) {
        Atomic<int> atomic(0);
        EXPECT_EQ(atomic.load(), 0);
    }

    // -------------------------------------------------------------------------
    // Load and Store
    // -------------------------------------------------------------------------

    TEST(Atomic, LoadAndStore) {
        Atomic<int> atomic(0);
        EXPECT_EQ(atomic.load(), 0);

        atomic.store(42);
        EXPECT_EQ(atomic.load(), 42);
    }

    // -------------------------------------------------------------------------
    // Compare and Exchange
    // -------------------------------------------------------------------------
    TEST(Atomic, CompareAndExchange) {
        Atomic<int> atomic(0);
        EXPECT_EQ(atomic.load(), 0);

        int expected = 0;
        int desired  = 42;

        EXPECT_TRUE(atomic.compare_exchange(expected, desired));
        EXPECT_EQ(atomic.load(), 42);

        expected = 0;
        desired  = 100;

        EXPECT_FALSE(atomic.compare_exchange(expected, desired));
        EXPECT_EQ(atomic.load(), 42);
    }

}  // namespace corekit
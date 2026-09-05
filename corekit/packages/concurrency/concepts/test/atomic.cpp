#include "corekit/atomic.hpp"

#include <gtest/gtest.h>

namespace corekit::test {

    /*
     * Standard atomic objects are not copyable or movable.
     *
     * These assertions are stricter than AtomicLike. Remove them if
     * corekit's wrapper deliberately defines different ownership semantics.
     */
    static_assert(!std::is_copy_constructible_v<Atomic<bool>>);
    static_assert(!std::is_copy_assignable_v<Atomic<bool>>);
    static_assert(!std::is_move_constructible_v<Atomic<bool>>);
    static_assert(!std::is_move_assignable_v<Atomic<bool>>);

    TEST(Atomic, StoreThenLoadReturnsStoredValue) {
        Atomic<int> atomic;
        atomic.store(42);

        EXPECT_EQ(atomic.load(), 42);
    }

    TEST(Atomic, LoadDoesNotModifyStoredValue) {
        Atomic<int>   atomic;
        constexpr int stored = 42;

        atomic.store(stored);

        const int first  = atomic.load();
        const int second = atomic.load();

        EXPECT_EQ(first, stored);
        EXPECT_EQ(second, stored);
    }

    TEST(Atomic, ConstLoadReturnsStoredValue) {
        Atomic<int>   atomic;
        constexpr int stored = 42;

        atomic.store(stored);

        const Atomic<int> &const_value = atomic;
        EXPECT_EQ(const_value.load(), stored);
    }

    TEST(Atomic, RepeatedStoresReplacePreviousValue) {
        Atomic<int> atomic;

        atomic.store(42);
        EXPECT_EQ(atomic.load(), 42);

        atomic.store(0);
        EXPECT_EQ(atomic.load(), 0);

        atomic.store(42);
        EXPECT_EQ(atomic.load(), 42);
    }

    TEST(Atomic, StoreCanWriteSameValueRepeatedly) {
        Atomic<int>   atomic;
        constexpr int stored = 42;

        atomic.store(stored);
        atomic.store(stored);
        atomic.store(stored);

        EXPECT_EQ(atomic.load(), stored);
    }

    TEST(Atomic, CompareExchangeSucceedsForMatchingValue) {
        Atomic<int>   atomic;
        constexpr int stored = 42;

        constexpr int initial = 42;
        constexpr int desired = 0;

        atomic.store(initial);

        int expected = initial;

        const bool exchanged = atomic.compare_exchange(expected, desired);

        EXPECT_TRUE(exchanged);
        EXPECT_EQ(atomic.load(), desired);
    }

    TEST(Atomic, CompareExchangeFailsForMismatchingValue) {
        Atomic<int>   atomic;
        constexpr int stored_value = 42;
        constexpr int wrong_value  = 0;
        constexpr int desired      = 0;

        atomic.store(stored_value);

        int expected = wrong_value;

        const bool exchanged = atomic.compare_exchange(expected, desired);

        EXPECT_FALSE(exchanged);
        EXPECT_EQ(atomic.load(), stored_value);
    }

    TEST(Atomic, FailedCompareExchangeDoesNotStoreDesiredValue) {
        Atomic<int>   atomic;
        constexpr int stored_value = 42;
        constexpr int wrong_value  = 0;
        constexpr int desired      = 0;

        atomic.store(stored_value);

        int expected = wrong_value;

        const bool exchanged = atomic.compare_exchange(expected, desired);

        EXPECT_FALSE(exchanged);
        EXPECT_EQ(atomic.load(), stored_value);

        expected = wrong_value;

        EXPECT_FALSE(atomic.compare_exchange(expected, desired));

        EXPECT_NE(atomic.load(), desired);
        EXPECT_EQ(atomic.load(), stored_value);
    }

    TEST(Atomic, CompareExchangeToSameValueSucceeds) {
        Atomic<int> atomic;

        constexpr int stored = 42;

        atomic.store(stored);

        int expected = stored;

        EXPECT_TRUE(atomic.compare_exchange(expected, stored));

        EXPECT_EQ(atomic.load(), stored);
    }

    TEST(Atomic, FailedExchangeLeavesValueAvailableForLaterSuccess) {
        Atomic<int> atomic;

        constexpr int stored        = 20;
        constexpr int wrong         = 10;
        constexpr int first_desired = 30;
        constexpr int final_desired = 40;

        atomic.store(stored);

        int wrong_expected = wrong;

        EXPECT_FALSE(atomic.compare_exchange(wrong_expected, first_desired));

        EXPECT_EQ(atomic.load(), stored);

        int correct_expected = stored;

        EXPECT_TRUE(atomic.compare_exchange(correct_expected, final_desired));

        EXPECT_EQ(atomic.load(), final_desired);
    }

    TEST(Atomic, OldExpectedValueFailsAfterSuccessfulExchange) {
        Atomic<int> atomic;

        constexpr int initial       = 10;
        constexpr int first_desired = 20;
        constexpr int next_desired  = 30;

        atomic.store(initial);

        int first_expected = initial;

        ASSERT_TRUE(atomic.compare_exchange(first_expected, first_desired));

        int stale_expected = initial;

        EXPECT_FALSE(atomic.compare_exchange(stale_expected, next_desired));

        EXPECT_EQ(atomic.load(), first_desired);
    }

    TEST(Atomic, NewExpectedValueSucceedsAfterSuccessfulExchange) {
        Atomic<int> atomic;

        constexpr int initial       = 10;
        constexpr int first_desired = 20;
        constexpr int next_desired  = 30;

        atomic.store(initial);

        int first_expected = initial;

        ASSERT_TRUE(atomic.compare_exchange(first_expected, first_desired));

        int next_expected = first_desired;

        EXPECT_TRUE(atomic.compare_exchange(next_expected, next_desired));

        EXPECT_EQ(atomic.load(), next_desired);
    }

    TEST(Atomic, SequentialCompareExchangeTransitionsWork) {
        Atomic<int> atomic;

        /*
         * Keep the range small enough for int8_t and uint8_t.
         */
        constexpr int transition_count = 100;

        atomic.store(0);

        for (int iteration = 0; iteration < transition_count; ++iteration) {
            int expected = iteration;
            int desired  = iteration + 1;

            ASSERT_TRUE(atomic.compare_exchange(expected, desired))
                << "Failed at transition " << iteration;

            EXPECT_EQ(atomic.load(), desired);
        }

        EXPECT_EQ(atomic.load(), transition_count);
    }

}  // namespace corekit::test
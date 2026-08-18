#include "corekit/assert.hpp"

#include <gtest/gtest.h>

namespace corekit {

    // -------------------------------------------------------------------------
    // Result<T> Success
    // -------------------------------------------------------------------------

    TEST(Result, DefaultConstructedContainsDefaultValue) {
        Result<int> result;

        ASSERT_TRUE(result);
        EXPECT_EQ(result.value(), 0);
    }

    TEST(Result, ConstructedWithValue) {
        Result<int> result(42);

        ASSERT_TRUE(result);
        EXPECT_EQ(result.value(), 42);
    }

    TEST(Result, StoresComplexValue) {
        Result<std::string> result("hello");

        ASSERT_TRUE(result);
        EXPECT_EQ(result.value(), "hello");
    }

    // -------------------------------------------------------------------------
    // Copy Semantics
    // -------------------------------------------------------------------------

    TEST(Result, CanBeCopied) {
        Result<int> result(42);

        Result<int> copy(result);

        ASSERT_TRUE(copy);
        EXPECT_EQ(copy.value(), 42);
    }

    TEST(Result, ErrorCanBeCopied) {
        Result<int> result(RuntimeError("failure"));

        Result<int> copy(result);

        ASSERT_FALSE(copy);
        EXPECT_EQ(copy.error().what(), "failure");
    }

    // -------------------------------------------------------------------------
    // Move Semantics
    // -------------------------------------------------------------------------

    TEST(Result, CanBeMoved) {
        Result<int> result(42);

        Result<int> moved(std::move(result));

        ASSERT_TRUE(moved);
        EXPECT_EQ(moved.value(), 42);
    }

    TEST(Result, ErrorCanBeMoved) {
        Result<int> result(RuntimeError("failure"));

        Result<int> moved(std::move(result));

        ASSERT_FALSE(moved);
        EXPECT_EQ(moved.error().what(), "failure");
    }

    // -------------------------------------------------------------------------
    // Bool Conversion
    // -------------------------------------------------------------------------

    TEST(Result, SuccessEvaluatesToTrue) {
        Result<int> result(42);

        EXPECT_TRUE(result);
    }

    TEST(Result, ErrorEvaluatesToFalse) {
        Result<int> result(RuntimeError("failure"));

        EXPECT_FALSE(result);
    }

    // -------------------------------------------------------------------------
    // Result
    // -------------------------------------------------------------------------

    TEST(Result, TrueValue) {
        Result result(true);

        ASSERT_TRUE(result);
        EXPECT_TRUE(result.value());
    }

    TEST(Result, FalseValue) {
        BoolResult result(false);

        ASSERT_TRUE(result);
        EXPECT_FALSE(result.value());
    }

    TEST(Result, Error) {
        BoolResult result(RuntimeError("failure"));

        ASSERT_FALSE(result);
        EXPECT_EQ(result.error().what(), "failure");
    }

    TEST(Result, BooleanValueDoesNotBecomeSuccessFlag) {
        Result result(false);

        ASSERT_TRUE(result);
        EXPECT_FALSE(result.value());
    }

    TEST(Result, DefaultConstructedIsSuccess) {
        VoidResult result;

        EXPECT_TRUE(result);
    }

    TEST(Result, ConstructedWithError) {
        VoidResult result(RuntimeError("failure"));

        EXPECT_FALSE(result);
    }

    TEST(Result, ErrorCanBeAccessed) {
        VoidResult result(RuntimeError("failure"));

        ASSERT_FALSE(result);
        EXPECT_EQ(result.error().what(), "failure");
    }

    // -------------------------------------------------------------------------
    // corecheck
    // -------------------------------------------------------------------------

    TEST(corecheck, CorecheckReturnsSuccessWhenConditionIsTrue) {
        auto result = corecheck(true);

        EXPECT_TRUE(result);
    }

    TEST(corecheck, CorecheckReturnsErrorWhenConditionIsFalse) {
        auto result = corecheck(false, RuntimeError("failed"));

        EXPECT_FALSE(result);
        EXPECT_EQ(result.error().what(), "failed");
    }

}  // namespace corekit
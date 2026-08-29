#include "corekit/assert.hpp"

#include <gtest/gtest.h>

namespace corekit {

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
#include "corekit/stoptoken.hpp"

#include <gtest/gtest.h>

namespace corekit::test {

    /*
     * A stop token should be lightweight and copyable.
     *
     * Remove these assertions if your implementation deliberately differs.
     */
    static_assert(std::is_copy_constructible_v<StopToken>);
    static_assert(std::is_copy_assignable_v<StopToken>);

    /*
     * Basic source construction.
     */

    TEST(StopSource, NewlyConstructedSourceHasNoStopRequested) {
        StopSource source;

        EXPECT_FALSE(source.stop_requested());
    }

    TEST(StopSource, NewlyConstructedSourceSupportsStopping) {
        StopSource source;

        EXPECT_TRUE(source.stop_possible());
    }

    TEST(StopSource, SourceCanCreateToken) {
        StopSource source;

        [[maybe_unused]] StopToken token = source.get_token();

        SUCCEED();
    }

    TEST(StopSource, TokenInitiallyReportsNoStopRequested) {
        StopSource source;

        StopToken token = source.get_token();

        EXPECT_FALSE(token.stop_requested());
    }

    TEST(StopSource, TokenInitiallyReportsStoppingPossible) {
        StopSource source;

        StopToken token = source.get_token();

        EXPECT_TRUE(token.stop_possible());
    }

    /*
     * request_stop().
     */

    TEST(StopSource, RequestStopReturnsTrueFirstTime) {
        StopSource source;

        EXPECT_TRUE(source.request_stop());
    }

    TEST(StopSource, StopRequestedIsTrueAfterRequestStop) {
        StopSource source;

        source.request_stop();

        EXPECT_TRUE(source.stop_requested());
    }

    TEST(StopSource, TokenReportsStopRequestedAfterSourceRequestStop) {
        StopSource source;

        StopToken token = source.get_token();

        source.request_stop();

        EXPECT_TRUE(token.stop_requested());
    }

    TEST(StopSource, SourceAndTokenRemainConsistentAfterRequestStop) {
        StopSource source;

        StopToken token = source.get_token();

        source.request_stop();

        EXPECT_TRUE(source.stop_requested());
        EXPECT_TRUE(token.stop_requested());
    }

    TEST(StopSource, RequestStopCanBeCalledRepeatedly) {
        StopSource source;

        source.request_stop();
        source.request_stop();
        source.request_stop();

        EXPECT_TRUE(source.stop_requested());
    }

    TEST(StopSource, SecondRequestStopReturnsFalse) {
        StopSource source;

        EXPECT_TRUE(source.request_stop());
        EXPECT_FALSE(source.request_stop());
    }

    /*
     * Multiple tokens observing same state.
     */

    TEST(StopToken, MultipleTokensObserveSameSourceState) {
        StopSource source;

        StopToken token1 = source.get_token();
        StopToken token2 = source.get_token();
        StopToken token3 = source.get_token();

        EXPECT_FALSE(token1.stop_requested());
        EXPECT_FALSE(token2.stop_requested());
        EXPECT_FALSE(token3.stop_requested());

        source.request_stop();

        EXPECT_TRUE(token1.stop_requested());
        EXPECT_TRUE(token2.stop_requested());
        EXPECT_TRUE(token3.stop_requested());
    }

    TEST(StopToken, TokenCopiesObserveSameState) {
        StopSource source;

        StopToken original = source.get_token();
        StopToken copy     = original;

        EXPECT_FALSE(original.stop_requested());
        EXPECT_FALSE(copy.stop_requested());

        source.request_stop();

        EXPECT_TRUE(original.stop_requested());
        EXPECT_TRUE(copy.stop_requested());
    }

    TEST(StopToken, TokenAssignmentObservesSameState) {
        StopSource source;

        StopToken token1 = source.get_token();
        StopToken token2;

        token2 = token1;

        source.request_stop();

        EXPECT_TRUE(token1.stop_requested());
        EXPECT_TRUE(token2.stop_requested());
    }

    /*
     * State remains stable.
     */

    TEST(StopToken, StopRequestedRemainsTrueOnceSet) {
        StopSource source;

        StopToken token = source.get_token();

        source.request_stop();

        EXPECT_TRUE(token.stop_requested());
        EXPECT_TRUE(token.stop_requested());
        EXPECT_TRUE(token.stop_requested());
    }

    TEST(StopSource, StopRequestedRemainsTrueOnceSet) {
        StopSource source;

        source.request_stop();

        EXPECT_TRUE(source.stop_requested());
        EXPECT_TRUE(source.stop_requested());
        EXPECT_TRUE(source.stop_requested());
    }

    TEST(StopToken, StopPossibleCanBeQueriedRepeatedly) {
        StopSource source;

        StopToken token = source.get_token();

        EXPECT_TRUE(token.stop_possible());
        EXPECT_TRUE(token.stop_possible());
        EXPECT_TRUE(token.stop_possible());
    }

    TEST(StopSource, StopPossibleCanBeQueriedRepeatedly) {
        StopSource source;

        EXPECT_TRUE(source.stop_possible());
        EXPECT_TRUE(source.stop_possible());
        EXPECT_TRUE(source.stop_possible());
    }

    /*
     * get_token().
     */

    TEST(StopSource, MultipleGetTokenCallsProduceValidTokens) {
        StopSource source;

        auto token1 = source.get_token();
        auto token2 = source.get_token();
        auto token3 = source.get_token();

        EXPECT_FALSE(token1.stop_requested());
        EXPECT_FALSE(token2.stop_requested());
        EXPECT_FALSE(token3.stop_requested());
    }

    TEST(StopSource, TokensCreatedBeforeAndAfterStopSeeSameState) {
        StopSource source;

        auto before = source.get_token();

        source.request_stop();

        auto after = source.get_token();

        EXPECT_TRUE(before.stop_requested());
        EXPECT_TRUE(after.stop_requested());
    }

    /*
     * Stress tests.
     */

    TEST(StopSourceStressTest, RepeatedStateQueriesRemainConsistent) {
        StopSource source;

        auto token = source.get_token();

        for (int i = 0; i < 100000; ++i) {
            EXPECT_FALSE(source.stop_requested());
            EXPECT_FALSE(token.stop_requested());
        }
    }

    TEST(StopSourceStressTest, RepeatedStoppedStateQueriesRemainConsistent) {
        StopSource source;

        auto token = source.get_token();

        source.request_stop();

        for (int i = 0; i < 100000; ++i) {
            EXPECT_TRUE(source.stop_requested());
            EXPECT_TRUE(token.stop_requested());
        }
    }

    TEST(StopSourceStressTest, ManyTokensObserveStopRequest) {
        StopSource source;

        std::vector<StopToken> tokens;

        constexpr int count = 1000;

        for (int i = 0; i < count; ++i) {
            tokens.push_back(source.get_token());
        }

        source.request_stop();

        for (const auto& token : tokens) {
            EXPECT_TRUE(token.stop_requested());
        }
    }

}  // namespace corekit::test
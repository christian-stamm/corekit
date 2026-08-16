#include <gtest/gtest.h>

#include "corekit/queue.hpp"


namespace corekit {

using Queue = SafeQueue<int>;

TEST(SafeQueue, IsInitiallyEmpty)
{
    Queue queue(3);
    int value = 0;

    EXPECT_TRUE(queue.empty());
    EXPECT_FALSE(queue.full());
    EXPECT_FALSE(queue.try_pop(value));
}

TEST(SafeQueue, PushMakesQueueNonEmpty)
{
    Queue queue(3);

    EXPECT_TRUE(queue.try_push(1));

    EXPECT_FALSE(queue.empty());
    EXPECT_FALSE(queue.full());
}

TEST(SafeQueue, PopsElementsInFifoOrder)
{
    Queue queue(3);
    int value = 0;

    ASSERT_TRUE(queue.try_push(1));
    ASSERT_TRUE(queue.try_push(2));
    ASSERT_TRUE(queue.try_push(3));

    ASSERT_TRUE(queue.try_pop(value));
    EXPECT_EQ(value, 1);

    ASSERT_TRUE(queue.try_pop(value));
    EXPECT_EQ(value, 2);

    ASSERT_TRUE(queue.try_pop(value));
    EXPECT_EQ(value, 3);

    EXPECT_TRUE(queue.empty());
}

TEST(SafeQueue, RejectsPushWhenFull)
{
    Queue queue(3);

    ASSERT_TRUE(queue.try_push(1));
    ASSERT_TRUE(queue.try_push(2));
    ASSERT_TRUE(queue.try_push(3));

    EXPECT_TRUE(queue.full());
    EXPECT_FALSE(queue.try_push(4));
}

TEST(SafeQueue, IsNotFullAfterPop)
{
    Queue queue(3);
    int value = 0;

    ASSERT_TRUE(queue.try_push(1));
    ASSERT_TRUE(queue.try_push(2));
    ASSERT_TRUE(queue.try_push(3));

    ASSERT_TRUE(queue.try_pop(value));

    EXPECT_EQ(value, 1);
    EXPECT_FALSE(queue.full());
}

TEST(SafeQueue, ClearRemovesAllElements)
{
    Queue queue(3);
    int value = 0;

    ASSERT_TRUE(queue.try_push(1));
    ASSERT_TRUE(queue.try_push(2));

    queue.clear();

    EXPECT_TRUE(queue.empty());
    EXPECT_FALSE(queue.full());
    EXPECT_FALSE(queue.try_pop(value));
}

TEST(SafeQueue, CanBeReusedAfterClear)
{
    Queue queue(3);
    int value = 0;

    ASSERT_TRUE(queue.try_push(1));

    queue.clear();

    ASSERT_TRUE(queue.try_push(42));
    ASSERT_TRUE(queue.try_pop(value));

    EXPECT_EQ(value, 42);
    EXPECT_TRUE(queue.empty());
}

} // namespace
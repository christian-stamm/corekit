#include <gtest/gtest.h>
#include <iostream>

#include "corekit/sync/task.hpp"


namespace corekit {

int testFn(int a, int b) {
    return a + b;
}

void testFn2(void) {
    std::cout << "testFn2 executed" << std::endl;
}

void callbackFn(int result) {
    std::cout << "Callback received result: " << result << std::endl;
}

TEST(Task, CheckTaskFactory)
{
    TaskQueue queue(3); // Set the capacity to 2
    auto task1 = queue.enqueue([]() { return testFn(2, 3); });
    task1->then(callbackFn);

    auto task2 = queue.enqueue(testFn, 5, 7);
    task2->then([](int result) {
        std::cout << "Lambda callback received result: " << result << std::endl;
    });

    auto task3 = queue.enqueue(testFn2);

    std::cout << "Executing all tasks in the queue..." << std::endl;
    queue.run();

    EXPECT_TRUE(task3->isDone()); // Placeholder assertion to ensure the test runs
}

};
#include "corekit/task.hpp"

#include <gtest/gtest.h>

namespace corekit {

    class TaskWithoutErrors : public Task {
       protected:
        Result on_enter(StopToken token) override {
            return {};
        }

        Result on_run(StopToken token) override {
            return {};
        }

        Result on_leave(StopToken token) override {
            return {};
        }
    };

    class TaskThatThrows : public Task {
       protected:
        Result on_enter(StopToken token) override {
            return {};
        }

        Result on_run(StopToken token) override {
            throw std::runtime_error("Test error");
        }

        Result on_leave(StopToken token) override {
            return {};
        }
    };

    class TaskThatFails : public Task {
       protected:
        Result on_enter(StopToken token) override {
            return {};
        }

        Result on_run(StopToken token) override {
            return std::unexpected(
                std::make_exception_ptr(std::runtime_error("Test failure")));
        }

        Result on_leave(StopToken token) override {
            return {};
        }
    };

    TEST(Task, BuildAndRun) {
        auto ssrc = StopSource();
        auto task = std::make_shared<TaskWithoutErrors>();

        EXPECT_FALSE(task->is_launched());
        EXPECT_FALSE(task->is_completed());
        EXPECT_FALSE(task->is_running());
        EXPECT_FALSE(task->is_crashed());

        task->exec(ssrc.get_token());

        EXPECT_TRUE(task->is_launched());
        EXPECT_TRUE(task->is_completed());
        EXPECT_FALSE(task->is_running());
        EXPECT_FALSE(task->is_crashed());
    }

    TEST(Task, ThrowAtRun) {
        auto ssrc = StopSource();
        auto task = std::make_shared<TaskThatThrows>();

        EXPECT_FALSE(task->is_launched());
        EXPECT_FALSE(task->is_completed());
        EXPECT_FALSE(task->is_running());
        EXPECT_FALSE(task->is_crashed());

        task->exec(ssrc.get_token());

        EXPECT_TRUE(task->is_launched());
        EXPECT_FALSE(task->is_completed());
        EXPECT_FALSE(task->is_running());
        EXPECT_TRUE(task->is_crashed());
    }

    TEST(Task, FailAtRun) {
        auto ssrc = StopSource();
        auto task = std::make_shared<TaskThatFails>();

        EXPECT_FALSE(task->is_launched());
        EXPECT_FALSE(task->is_completed());
        EXPECT_FALSE(task->is_running());
        EXPECT_FALSE(task->is_crashed());

        task->exec(ssrc.get_token());

        EXPECT_TRUE(task->is_launched());
        EXPECT_FALSE(task->is_completed());
        EXPECT_FALSE(task->is_running());
        EXPECT_FALSE(task->is_crashed());
    }

    TEST(Task, StopRequested) {
        auto ssrc = StopSource();
        auto task = std::make_shared<TaskWithoutErrors>();

        EXPECT_FALSE(task->is_launched());
        EXPECT_FALSE(task->is_completed());
        EXPECT_FALSE(task->is_running());
        EXPECT_FALSE(task->is_crashed());

        ssrc.request_stop();
        task->exec(ssrc.get_token());

        EXPECT_TRUE(task->is_launched());
        EXPECT_TRUE(task->is_completed());
        EXPECT_FALSE(task->is_running());
        EXPECT_FALSE(task->is_crashed());
    }

}  // namespace corekit
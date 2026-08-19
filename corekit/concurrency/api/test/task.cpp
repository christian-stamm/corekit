#include "corekit/task.hpp"

#include <gtest/gtest.h>

namespace corekit {

    class TaskWithoutErrors : public Task {
       protected:
        bool on_enter(const StopToken& token) override {
            return true;
        }

        bool on_run(const StopToken& token) override {
            return !token.stop_requested();
        }

        bool on_leave(const StopToken& token) override {
            return true;
        }
    };

    class TaskThatThrows : public Task {
       protected:
        bool on_enter(const StopToken& token) override {
            return true;
        }

        bool on_run(const StopToken& token) override {
            throw std::runtime_error("Test error");
        }

        bool on_leave(const StopToken& token) override {
            return true;
        }
    };

    class TaskThatFails : public Task {
       protected:
        bool on_enter(const StopToken& token) override {
            return true;
        }

        bool on_run(const StopToken& token) override {
            return false;
        }

        bool on_leave(const StopToken& token) override {
            return true;
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
        EXPECT_TRUE(task->is_completed());
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
        EXPECT_TRUE(task->is_completed());
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
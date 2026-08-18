#include "corekit/task.hpp"

#include <gtest/gtest.h>

namespace corekit {

    namespace {

        enum class Callback {
            ENTER,
            RUN,
            LEAVE,
        };

        // ---------------------------------------------------------------------
        // Successful task
        // ---------------------------------------------------------------------

        class SuccessfulTask final : public Task {
           public:
            const std::vector<Callback>& callbacks() const {
                return m_callbacks;
            }

           protected:
            VoidResult on_enter(StopToken) override {
                m_callbacks.push_back(Callback::ENTER);
                return {};
            }

            VoidResult on_run(StopToken) override {
                m_callbacks.push_back(Callback::RUN);
                return {};
            }

            VoidResult on_leave(StopToken) override {
                m_callbacks.push_back(Callback::LEAVE);
                return {};
            }

           private:
            std::vector<Callback> m_callbacks;
        };

        // ---------------------------------------------------------------------
        // State-observing task
        // ---------------------------------------------------------------------

        class StateObservingTask final : public Task {
           public:
            State enter_state = State::READY;
            State run_state   = State::READY;
            State leave_state = State::READY;

           protected:
            VoidResult on_enter(StopToken) override {
                enter_state = get_state();
                return {};
            }

            VoidResult on_run(StopToken) override {
                run_state = get_state();
                return {};
            }

            VoidResult on_leave(StopToken) override {
                leave_state = get_state();
                return {};
            }
        };

        // ---------------------------------------------------------------------
        // Stop-token-observing task
        // ---------------------------------------------------------------------

        class StopTokenTask final : public Task {
           public:
            bool enter_stop_requested = false;
            bool run_stop_requested   = false;
            bool leave_stop_requested = false;

           protected:
            VoidResult on_enter(StopToken token) override {
                enter_stop_requested = token.stop_requested();
                return {};
            }

            VoidResult on_run(StopToken token) override {
                run_stop_requested = token.stop_requested();
                return {};
            }

            VoidResult on_leave(StopToken token) override {
                leave_stop_requested = token.stop_requested();
                return {};
            }
        };

        // ---------------------------------------------------------------------
        // Failing tasks
        // ---------------------------------------------------------------------

        class EnterFailingTask final : public Task {
           public:
            bool run_called   = false;
            bool leave_called = false;

           protected:
            VoidResult on_enter(StopToken) override {
                return RuntimeError("on_enter failed");
            }

            VoidResult on_run(StopToken) override {
                run_called = true;
                return {};
            }

            VoidResult on_leave(StopToken) override {
                leave_called = true;
                return {};
            }
        };

        class RunFailingTask final : public Task {
           public:
            bool enter_called = false;
            bool run_called   = false;
            bool leave_called = false;

           protected:
            VoidResult on_enter(StopToken) override {
                enter_called = true;
                return {};
            }

            VoidResult on_run(StopToken) override {
                run_called = true;
                return RuntimeError("on_run failed");
            }

            VoidResult on_leave(StopToken) override {
                leave_called = true;
                return {};
            }
        };

        class LeaveFailingTask final : public Task {
           public:
            bool enter_called = false;
            bool run_called   = false;
            bool leave_called = false;

           protected:
            VoidResult on_enter(StopToken) override {
                enter_called = true;
                return {};
            }

            VoidResult on_run(StopToken) override {
                run_called = true;
                return {};
            }

            VoidResult on_leave(StopToken) override {
                leave_called = true;
                return RuntimeError("on_leave failed");
            }
        };

        // ---------------------------------------------------------------------
        // Throwing tasks
        // ---------------------------------------------------------------------

        class EnterThrowingTask final : public Task {
           public:
            bool run_called   = false;
            bool leave_called = false;

           protected:
            VoidResult on_enter(StopToken) override {
                throw std::runtime_error("on_enter threw");
            }

            VoidResult on_run(StopToken) override {
                run_called = true;
                return {};
            }

            VoidResult on_leave(StopToken) override {
                leave_called = true;
                return {};
            }
        };

        class RunThrowingTask final : public Task {
           public:
            bool enter_called = false;
            bool leave_called = false;

           protected:
            VoidResult on_enter(StopToken) override {
                enter_called = true;
                return {};
            }

            VoidResult on_run(StopToken) override {
                throw std::runtime_error("on_run threw");
            }

            VoidResult on_leave(StopToken) override {
                leave_called = true;
                return {};
            }
        };

        class LeaveThrowingTask final : public Task {
           public:
            bool enter_called = false;
            bool run_called   = false;
            bool leave_called = false;

           protected:
            VoidResult on_enter(StopToken) override {
                enter_called = true;
                return {};
            }

            VoidResult on_run(StopToken) override {
                run_called = true;
                return {};
            }

            VoidResult on_leave(StopToken) override {
                leave_called = true;
                throw std::runtime_error("on_leave threw");
            }
        };

        // ---------------------------------------------------------------------
        // Blocking task for deterministic concurrency testing
        // ---------------------------------------------------------------------

        class BlockingTask final : public Task {
           public:
            void wait_until_running() {
                std::unique_lock lock(m_mutex);

                m_condition.wait(lock, [this] { return m_running; });
            }

            void release() {
                {
                    std::lock_guard lock(m_mutex);
                    m_released = true;
                }

                m_condition.notify_all();
            }

            int run_count() const {
                return m_run_count.load();
            }

           protected:
            VoidResult on_run(StopToken) override {
                ++m_run_count;

                std::unique_lock lock(m_mutex);
                m_running = true;
                m_condition.notify_all();

                m_condition.wait(lock, [this] { return m_released; });

                return {};
            }

           private:
            std::atomic<int>        m_run_count{0};
            std::mutex              m_mutex;
            std::condition_variable m_condition;
            bool                    m_running  = false;
            bool                    m_released = false;
        };

    }  // namespace

    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    TEST(Task, StartsReady) {
        SuccessfulTask task;

        EXPECT_EQ(task.get_state(), Task::State::READY);
        EXPECT_FALSE(task.is_launched());
        EXPECT_FALSE(task.is_running());
        EXPECT_FALSE(task.is_completed());
    }

    // -------------------------------------------------------------------------
    // Successful Execution
    // -------------------------------------------------------------------------

    TEST(Task, ExecutesSuccessfully) {
        SuccessfulTask task;
        StopSource     stop_source;

        const auto result = task.exec(stop_source.get_token());

        EXPECT_TRUE(result);
        EXPECT_TRUE(task.is_launched());
        EXPECT_FALSE(task.is_running());
        EXPECT_TRUE(task.is_completed());
        EXPECT_EQ(task.get_state(), Task::State::TERMINATED);
    }

    TEST(Task, InvokesCallbacksInOrder) {
        SuccessfulTask task;
        StopSource     stop_source;

        const auto result = task.exec(stop_source.get_token());

        ASSERT_TRUE(result);
        ASSERT_EQ(task.callbacks().size(), 3u);

        EXPECT_EQ(task.callbacks()[0], Callback::ENTER);
        EXPECT_EQ(task.callbacks()[1], Callback::RUN);
        EXPECT_EQ(task.callbacks()[2], Callback::LEAVE);
    }

    TEST(Task, IsRunningDuringAllCallbacks) {
        StateObservingTask task;
        StopSource         stop_source;

        const auto result = task.exec(stop_source.get_token());

        ASSERT_TRUE(result);

        EXPECT_EQ(task.enter_state, Task::State::RUNNING);
        EXPECT_EQ(task.run_state, Task::State::RUNNING);
        EXPECT_EQ(task.leave_state, Task::State::RUNNING);
        EXPECT_EQ(task.get_state(), Task::State::TERMINATED);
    }

    // -------------------------------------------------------------------------
    // Stop Token
    // -------------------------------------------------------------------------

    TEST(Task, ForwardsStopTokenToAllCallbacks) {
        StopTokenTask task;
        StopSource    stop_source;

        stop_source.request_stop();

        const auto result = task.exec(stop_source.get_token());

        ASSERT_TRUE(result);

        EXPECT_TRUE(task.enter_stop_requested);
        EXPECT_TRUE(task.run_stop_requested);
        EXPECT_TRUE(task.leave_stop_requested);
    }

    // -------------------------------------------------------------------------
    // Callback Failures
    // -------------------------------------------------------------------------

    TEST(Task, EnterFailureStopsExecution) {
        EnterFailingTask task;
        StopSource       stop_source;

        const auto result = task.exec(stop_source.get_token());

        EXPECT_FALSE(result);
        EXPECT_FALSE(task.run_called);
        EXPECT_FALSE(task.leave_called);
        EXPECT_TRUE(task.is_completed());
        EXPECT_EQ(task.get_state(), Task::State::TERMINATED);
    }

    TEST(Task, RunFailurePreventsLeave) {
        RunFailingTask task;
        StopSource     stop_source;

        const auto result = task.exec(stop_source.get_token());

        EXPECT_FALSE(result);
        EXPECT_TRUE(task.enter_called);
        EXPECT_TRUE(task.run_called);
        EXPECT_FALSE(task.leave_called);
        EXPECT_TRUE(task.is_completed());
        EXPECT_EQ(task.get_state(), Task::State::TERMINATED);
    }

    TEST(Task, LeaveFailureIsReturned) {
        LeaveFailingTask task;
        StopSource       stop_source;

        const auto result = task.exec(stop_source.get_token());

        EXPECT_FALSE(result);
        EXPECT_TRUE(task.enter_called);
        EXPECT_TRUE(task.run_called);
        EXPECT_TRUE(task.leave_called);
        EXPECT_TRUE(task.is_completed());
        EXPECT_EQ(task.get_state(), Task::State::TERMINATED);
    }

    // -------------------------------------------------------------------------
    // Callback Exceptions
    // -------------------------------------------------------------------------

    TEST(Task, ConvertsEnterExceptionToFailure) {
        EnterThrowingTask task;
        StopSource        stop_source;

        const auto result = task.exec(stop_source.get_token());

        EXPECT_FALSE(result);
        EXPECT_FALSE(task.run_called);
        EXPECT_FALSE(task.leave_called);
        EXPECT_TRUE(task.is_completed());
        EXPECT_EQ(task.get_state(), Task::State::TERMINATED);
    }

    TEST(Task, ConvertsRunExceptionToFailure) {
        RunThrowingTask task;
        StopSource      stop_source;

        const auto result = task.exec(stop_source.get_token());

        EXPECT_FALSE(result);
        EXPECT_TRUE(task.enter_called);
        EXPECT_FALSE(task.leave_called);
        EXPECT_TRUE(task.is_completed());
        EXPECT_EQ(task.get_state(), Task::State::TERMINATED);
    }

    TEST(Task, ConvertsLeaveExceptionToFailure) {
        LeaveThrowingTask task;
        StopSource        stop_source;

        const auto result = task.exec(stop_source.get_token());

        EXPECT_FALSE(result);
        EXPECT_TRUE(task.enter_called);
        EXPECT_TRUE(task.run_called);
        EXPECT_TRUE(task.leave_called);
        EXPECT_TRUE(task.is_completed());
        EXPECT_EQ(task.get_state(), Task::State::TERMINATED);
    }

    // -------------------------------------------------------------------------
    // Repeated Execution
    // -------------------------------------------------------------------------

    TEST(Task, CannotExecuteCompletedTaskAgain) {
        SuccessfulTask task;
        StopSource     stop_source;

        const auto first_result = task.exec(stop_source.get_token());

        ASSERT_TRUE(first_result);
        ASSERT_EQ(task.callbacks().size(), 3u);
        ASSERT_TRUE(task.is_completed());

        const auto second_result = task.exec(stop_source.get_token());

        EXPECT_FALSE(second_result);
        EXPECT_EQ(task.callbacks().size(), 3u);
        EXPECT_TRUE(task.is_completed());
        EXPECT_EQ(task.get_state(), Task::State::TERMINATED);
    }

    // -------------------------------------------------------------------------
    // Concurrent Execution
    // -------------------------------------------------------------------------

    TEST(Task, CannotExecuteTaskWhileAlreadyRunning) {
        BlockingTask task;
        StopSource   stop_source;

        VoidResult first_result;

        std::thread first_thread(
            [&] { first_result = task.exec(stop_source.get_token()); });

        task.wait_until_running();

        ASSERT_TRUE(task.is_launched());
        ASSERT_TRUE(task.is_running());
        ASSERT_FALSE(task.is_completed());

        const auto second_result = task.exec(stop_source.get_token());

        EXPECT_FALSE(second_result);
        EXPECT_TRUE(task.is_running());
        EXPECT_EQ(task.run_count(), 1);

        task.release();
        first_thread.join();

        EXPECT_TRUE(first_result);
        EXPECT_EQ(task.run_count(), 1);
        EXPECT_FALSE(task.is_running());
        EXPECT_TRUE(task.is_completed());
        EXPECT_EQ(task.get_state(), Task::State::TERMINATED);
    }

}  // namespace corekit
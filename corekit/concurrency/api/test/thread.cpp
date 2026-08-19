#include "corekit/thread.hpp"

#include <gtest/gtest.h>

#include "corekit/task.hpp"

namespace corekit {

    // -------------------------------------------------------------------------
    // Acquire and Release
    // -------------------------------------------------------------------------

    class TestTask : public Task {
       public:
        static Ptr create() {
            return std::make_unique<TestTask>();
        }

        bool on_enter(const StopToken& token) override {
            return true;
        }

        bool on_run(const StopToken& token) override {
            while (!token.stop_requested()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            return true;
        }

        bool on_leave(const StopToken& token) override {
            return true;
        }
    };

    TEST(Thread, CanRunAndJoin) {
        TestTask::Ptr task = TestTask::create();

        Thread thread(std::move(task));

        thread.run();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        thread.request_stop();
        thread.join();
    }

}  // namespace corekit
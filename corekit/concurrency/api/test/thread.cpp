#include "corekit/thread.hpp"

#include <gtest/gtest.h>

#include "corekit/task.hpp"
#include "corekit/time.hpp"

namespace corekit {

    class SimpleTask : public Task {
       protected:
        bool on_enter(const StopToken& token) override {
            return true;
        }

        bool on_run(const StopToken& token) override {
            while (!token.stop_requested()) {
                Time::sleep(1e-3f);
            }

            return true;
        }

        bool on_leave(const StopToken& token) override {
            return true;
        }
    };

    TEST(Thread, CanRunAndJoin) {
        SimpleTask::Ptr task = std::make_shared<SimpleTask>();

        Thread thread(task);
        thread.request_stop();
        thread.join();
    }

}  // namespace corekit
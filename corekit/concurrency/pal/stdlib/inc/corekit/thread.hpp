#pragma once
#include <memory>
#include <thread>
#include <utility>

#include "corekit/concepts.hpp"
#include "corekit/stoptoken.hpp"
#include "corekit/task.hpp"

namespace corekit {

    class StdlibThread {
       public:
        using Ptr = std::shared_ptr<StdlibThread>;

        StdlibThread(const Task::Ptr& task) : task_(task) {}

        ~StdlibThread() = default;

        void request_stop() {
            thread_.request_stop();
        }

        void run() {
            if (!thread_.joinable()) {
                thread_ = std::jthread(
                    [task = std::forward<Task::Ptr>(task_)](
                        StopToken token) mutable { task->exec(token); });
            }
        }

        void join() {
            if (thread_.joinable()) {
                thread_.join();
            }
        }

       private:
        Task::Ptr    task_;
        std::jthread thread_;
    };

    using Thread = StdlibThread;

}  // namespace corekit

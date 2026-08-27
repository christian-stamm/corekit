#pragma once

#include <deque>
#include <thread>
#include <vector>

#include "corekit/assert.hpp"
#include "corekit/conditionvariable.hpp"
#include "corekit/mutex.hpp"
#include "corekit/stoptoken.hpp"
#include "corekit/task.hpp"

namespace corekit::platform {

    class ThreadPool {
       public:
        explicit ThreadPool(uint num_workers = 4, uint max_tasks = 10);
        ~ThreadPool();

        ThreadPool(const ThreadPool&)            = delete;
        ThreadPool& operator=(const ThreadPool&) = delete;

        ThreadPool(ThreadPool&&)            = delete;
        ThreadPool& operator=(ThreadPool&&) = delete;

        VoidResult enqueue(Task::Ptr task);

        void cancel(bool remaining_tasks = false);

        const uint num_workers_;
        const uint max_tasks_;

       private:
        void worker_loop();

        StopSource               m_stop_source_;
        std::vector<std::thread> m_workers_;
    };

    using Executor = ThreadPool;

}  // namespace corekit::platform
#pragma once

#include <thread>
#include <vector>

#include "corekit/assert.hpp"
#include "corekit/queue.hpp"
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

       private:
        void worker_loop();

        Queue<Task::Ptr>         m_task_queue_;
        StopSource               m_stop_source_;
        std::vector<std::thread> m_workers_;
    };

    using Executor = ThreadPool;

}  // namespace corekit::platform
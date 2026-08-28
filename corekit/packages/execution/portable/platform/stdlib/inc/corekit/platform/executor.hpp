#pragma once

#include <thread>
#include <vector>

#include "corekit/assert.hpp"
#include "corekit/queue.hpp"
#include "corekit/semaphore.hpp"
#include "corekit/task.hpp"

namespace corekit::platform {

    class ThreadPool {
        friend class Executor;

       public:
        explicit ThreadPool(uint num_workers = 4, uint max_tasks = 10);
        ~ThreadPool();

        ThreadPool(const ThreadPool&)            = delete;
        ThreadPool& operator=(const ThreadPool&) = delete;

        ThreadPool(ThreadPool&&)            = delete;
        ThreadPool& operator=(ThreadPool&&) = delete;

        VoidResult enqueue(Task::Ptr task);
        void       cancel(bool discard_remaining_tasks = false);

       protected:
        void request_stop(bool discard_remaining_tasks = false);
        void join();

        const uint num_workers_;

       private:
        void worker_loop();

        StopSource               m_stop_source_;
        Queue<Task::Ptr>         m_task_queue_;
        std::vector<std::thread> m_workers_;
    };

    class Executor : public ThreadPool {
       public:
        Executor(uint num_workers = 4, uint max_tasks = 10)
            : ThreadPool(num_workers, max_tasks)
            , shutdown_(0, 1) {}

        void launch();
        void terminate();

       private:
        Semaphore shutdown_;
    };

}  // namespace corekit::platform
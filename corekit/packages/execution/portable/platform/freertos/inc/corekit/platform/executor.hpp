#pragma once

#include <FreeRTOS.h>
#include <task.h>

#include <vector>

#include "corekit/queue.hpp"
#include "corekit/result.hpp"
#include "corekit/semaphore.hpp"
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

        Semaphore                 m_worker_count_;
        Queue<Task::Ptr>          m_task_queue_;
        StopSource                m_stop_source_;
        std::vector<TaskHandle_t> m_workers_;
    };

    class Executor : public ThreadPool {
       public:
        using ThreadPool::ThreadPool;

        static void launch();
        static void terminate();
    };

}  // namespace corekit::platform
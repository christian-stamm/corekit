#include "corekit/platform/executor.hpp"

#include <format>

namespace corekit::platform {

    ThreadPool::ThreadPool(uint num_workers, uint max_tasks)
        : num_workers_(num_workers)
        , m_task_queue_(max_tasks)
        , m_worker_count_(num_workers, 0) {
        for (size_t i = 0; i < num_workers_; ++i) {
            TaskHandle_t task_handle = nullptr;
            std::string  task_name   = std::format("worker_{}", i);

            BaseType_t result = xTaskCreate(
                [](void* arg) {
                    ThreadPool* self = static_cast<ThreadPool*>(arg);
                    self->worker_loop();
                    vTaskDelete(nullptr);
                },
                task_name.c_str(),
                configMINIMAL_STACK_SIZE,
                this,
                tskIDLE_PRIORITY + 1,
                &task_handle);

            if (result != pdPASS) {
                // Handle error
                continue;
            }

            m_workers_.emplace_back(task_handle);
        }
    }

    ThreadPool::~ThreadPool() {
        cancel();

        for (const TaskHandle_t& worker : m_workers_) {
            m_worker_count_.acquire();
        }
    }

    void ThreadPool::cancel(bool discard_tasks) {
        if (m_stop_source_.stop_requested()) {
            return;
        }

        m_stop_source_.request_stop();

        if (discard_tasks) {
            m_task_queue_.clear();
        }

        // Wake each potentially blocked worker.
        for (size_t i = 0; i < m_workers_.size(); ++i) {
            m_task_queue_.push(nullptr, true);
        }
    }

    VoidResult ThreadPool::enqueue(Task::Ptr task) {
        if (!task) {
            return RuntimeError("null task");
        }

        if (m_stop_source_.stop_requested()) {
            return RuntimeError("executor stopped");
        }

        return m_task_queue_.push(std::move(task), false);
    }

    void ThreadPool::worker_loop() {
        while (true) {
            Task::Ptr task;

            m_task_queue_.pop(task, true);

            if (!task) {
                break;
            }

            task->exec(m_stop_source_.get_token());
        }

        m_worker_count_.release();
    }

    void Executor::launch() {
        vTaskStartScheduler();
    }

    void Executor::terminate() {
        vTaskEndScheduler();
    }

}  // namespace corekit::platform
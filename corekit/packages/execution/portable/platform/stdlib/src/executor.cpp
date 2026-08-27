#include "corekit/platform/executor.hpp"

#include <utility>

namespace corekit::platform {

    ThreadPool::ThreadPool(uint num_workers, uint max_tasks)
        : num_workers_(num_workers)
        , m_task_queue_(max_tasks) {
        for (size_t i = 0; i < num_workers_; ++i) {
            m_workers_.emplace_back([this] { worker_loop(); });
        }
    }

    ThreadPool::~ThreadPool() {
        cancel();
    }

    void ThreadPool::cancel(bool remaining_tasks) {
        m_stop_source_.request_stop();

        if (remaining_tasks) {
            m_task_queue_.clear();
        }

        for (auto& worker : m_workers_) {
            if (worker) {
                vTaskDelete(worker);
            }
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
        Task::Ptr task = nullptr;

        while (true) {
            m_task_queue_.pop(task, true);

            if (task) {
                task->exec(m_stop_source_.get_token());
            }
        }
    }

}  // namespace corekit::platform
#include "corekit/platform/executor.hpp"

#include <utility>

namespace corekit::platform {

    ThreadPool::ThreadPool(uint num_workers, uint max_tasks)
        : num_workers_(num_workers)
        , max_tasks_(max_tasks) {
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
            std::lock_guard lock(m_queue_mutex_);
            m_task_queue_.clear();
        }

        m_condition_.notify_all();

        for (auto& worker : m_workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    VoidResult ThreadPool::enqueue(Task::Ptr task) {
        if (!task) {
            return VoidResult(RuntimeError("null task"));
        }

        {
            std::lock_guard lock(m_queue_mutex_);

            if (m_stop_source_.stop_requested()) {
                return RuntimeError("executor stopped");
            }

            if (max_tasks_ <= m_task_queue_.size()) {
                return RuntimeError("task queue full");
            }

            m_task_queue_.push_back(std::move(task));
        }

        m_condition_.notify_one();

        return VoidResult();
    }

    void ThreadPool::worker_loop() {
        while (true) {
            Task::Ptr task;

            {
                std::unique_lock lock(m_queue_mutex_);

                m_condition_.wait(lock, [this] {
                    return m_stop_source_.stop_requested() ||
                           !m_task_queue_.empty();
                });

                if (m_stop_source_.stop_requested() && m_task_queue_.empty()) {
                    return;
                }

                task = std::move(m_task_queue_.front());
                m_task_queue_.pop_front();
            }

            task->exec(m_stop_source_.get_token());
        }
    }

}  // namespace corekit::platform
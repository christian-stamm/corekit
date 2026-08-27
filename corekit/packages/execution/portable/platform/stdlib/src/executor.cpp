#include "corekit/platform/executor.hpp"

#include <iostream>
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

    void ThreadPool::request_stop(bool discard_pending_tasks) {
        if (!m_stop_source_.request_stop()) {
            return;
        }

        if (discard_pending_tasks) {
            m_task_queue_.clear();
        }

        // Exactly one sentinel per worker.
        for (std::size_t i = 0; i < m_workers_.size(); ++i) {
            m_task_queue_.push(nullptr, true);
        }
    }

    void ThreadPool::join() {
        for (auto& worker : m_workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    void ThreadPool::cancel(bool discard_pending_tasks) {
        request_stop(discard_pending_tasks);
        join();
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
            Task::Ptr task = nullptr;

            m_task_queue_.pop(task, true);

            if (!task) {
                break;
            }

            task->exec(m_stop_source_.get_token());
        }
    }

    void Executor::launch() {
        shutdown_.acquire();
        join();
    }

    void Executor::terminate() {
        request_stop(false);
        shutdown_.release();
    }

}  // namespace corekit::platform
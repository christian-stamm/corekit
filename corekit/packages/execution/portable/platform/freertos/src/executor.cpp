#include "corekit/platform/executor.hpp"

#include "corekit/check.hpp"

namespace corekit::platform {

    Thread::Thread(Task::Ptr task, StopToken token)
        : m_state_(State::NotStarted)
        , m_token_(token)
        , m_task_(task)
        , m_handle_(nullptr) { }

    Thread::~Thread() { join(); }

    bool Thread::start(uint coremask, uint priority) {
        auto expected = State::NotStarted;
        auto desired  = State::Running;

        // Synchronize this transition.
        if (!m_state_.compare_exchange(expected, desired)) { return false; }

        BaseType_t result = xTaskCreate(
            [](void* arg) {
                auto* self = static_cast<Thread*>(arg);

                corecheck(self != nullptr && self->m_task_ != nullptr, RuntimeError("Thread::start() task is null"));

                corecheck(self->m_task_->exec(self->m_token_));

                self->m_state_.store(State::Finished);
                // Last operation touching self.
                self->m_joiner_.release();
                vTaskDelete(nullptr);
            },
            m_task_->name.c_str(),
            configMINIMAL_STACK_SIZE,
            this,
            tskIDLE_PRIORITY + priority,
            &m_handle_);

        if (result == pdPASS) {
            vTaskCoreAffinitySet(m_handle_, coremask);
            return true;
        } else {
            m_handle_ = nullptr;
            m_state_.store(State::NotStarted);
            return false;
        }
    }

    void Thread::join() {
        const State state = m_state_.load();

        if (state == State::NotStarted) { return; }

        if (state == State::Running) { m_joiner_.acquire(); }
    }

    void Executor::enqueue(Task::Ptr task, uint coremask, uint priority) {
        Thread::Ptr thread = std::make_shared<Thread>(task, m_stopsrc_.get_token());

        m_threads_.push_back(thread);
        thread->start(coremask, priority);
    }

    void Executor::launch() { vTaskStartScheduler(); }

    void Executor::cancel() {
        m_stopsrc_.request_stop();

        for (const Thread::Ptr& thread : m_threads_) { thread->join(); }

        vTaskEndScheduler();
    }

} // namespace corekit::platform
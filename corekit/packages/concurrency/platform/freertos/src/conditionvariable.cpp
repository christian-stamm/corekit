#include "corekit/platform/conditionvariable.hpp"

namespace corekit::platform {

    void ConditionVariable::wait(std::unique_lock<Mutex>& lock) {
        TaskHandle_t self = xTaskGetCurrentTaskHandle();

        {
            std::lock_guard<Mutex> guard(waiters_mutex_);
            waiters_.push_back(self);
        }

        lock.unlock();

        // Notification is persistent, unlike vTaskResume().
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

        lock.lock();
    }

    void ConditionVariable::notify_one() {
        TaskHandle_t task = nullptr;

        {
            std::lock_guard<Mutex> guard(waiters_mutex_);

            if (!waiters_.empty()) {
                task = waiters_.front();
                waiters_.pop_front();
            }
        }

        if (task != nullptr) {
            xTaskNotifyGive(task);
        }
    }

    void ConditionVariable::notify_all() {
        std::deque<TaskHandle_t> pending;

        {
            std::lock_guard<Mutex> guard(waiters_mutex_);
            pending.swap(waiters_);
        }

        for (TaskHandle_t task : pending) {
            xTaskNotifyGive(task);
        }
    }

}  // namespace corekit::platform
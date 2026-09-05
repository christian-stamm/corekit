#include "corekit/platform/conditionvariable.hpp"  // IWYU pragma: keep

namespace corekit::platform {

    ConditionVariable::ConditionVariable(uint32_t max_waiters)
        : waiters_(xQueueCreate(max_waiters, sizeof(Waiter*))) {
        configASSERT(waiters_ != nullptr);
    }

    ConditionVariable::~ConditionVariable() {
        if (waiters_) {
            configASSERT(uxQueueMessagesWaiting(waiters_) == 0);
            vQueueDelete(waiters_);
        }
    }

    void ConditionVariable::wait(std::unique_lock<Mutex>& lock) {
        Waiter waiter{};

        waiter.semaphore = xSemaphoreCreateBinaryStatic(&waiter.storage);

        configASSERT(waiter.semaphore != nullptr);

        Waiter* waiter_ptr = &waiter;

        //
        // Register ourselves while still holding the caller's mutex.
        //
        // Do not block here. Blocking while holding `lock` could deadlock
        // because another thread may need the same mutex in order to notify.
        //
        const BaseType_t queued = xQueueSend(waiters_, &waiter_ptr, 0);

        configASSERT(queued == pdTRUE);

        //
        // The waiter is now visible to notify_one()/notify_all().
        //
        // If notification happens between unlock() and xSemaphoreTake(),
        // the binary semaphore remembers the wakeup.
        //
        lock.unlock();

        xSemaphoreTake(waiter.semaphore, portMAX_DELAY);

        lock.lock();
    }

    void ConditionVariable::notify_one() {
        Waiter* waiter = nullptr;

        if (xQueueReceive(waiters_, &waiter, 0) == pdTRUE) {
            configASSERT(waiter != nullptr);

            xSemaphoreGive(waiter->semaphore);
        }
    }

    void ConditionVariable::notify_all() {
        const UBaseType_t count = uxQueueMessagesWaiting(waiters_);

        for (UBaseType_t i = 0; i < count; ++i) {
            Waiter* waiter = nullptr;

            if (xQueueReceive(waiters_, &waiter, 0) != pdTRUE) {
                break;
            }

            configASSERT(waiter != nullptr);

            xSemaphoreGive(waiter->semaphore);
        }
    }

}  // namespace corekit::platform
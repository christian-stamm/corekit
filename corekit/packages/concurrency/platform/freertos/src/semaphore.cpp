#include "corekit/platform/semaphore.hpp"  // IWYU pragma: keep

#include <FreeRTOS.h>
#include <task.h>

namespace corekit::platform {

    // ============================================================================
    // Semaphore::Opset
    // ============================================================================

    Semaphore::Opset::Opset(SemaphoreHandle_t& handle) : handle_(handle) {}

    Semaphore::Opset::~Opset() = default;

    bool Semaphore::Opset::take(TickType_t ticks) const {
        return xSemaphoreTake(handle_, ticks) == pdTRUE;
    }

    bool Semaphore::Opset::release() const {
        return xSemaphoreGive(handle_) == pdTRUE;
    }

    // ============================================================================
    // Semaphore::IsrSet
    // ============================================================================

    Semaphore::IsrSet::IsrSet(SemaphoreHandle_t& handle) : Opset(handle) {}

    bool Semaphore::IsrSet::take(TickType_t /* ticks */) const {
        BaseType_t higher_priority_task_woken = pdFALSE;
        BaseType_t result =
            xSemaphoreTakeFromISR(handle_, &higher_priority_task_woken);
        portYIELD_FROM_ISR(higher_priority_task_woken);
        return result == pdTRUE;
    }

    bool Semaphore::IsrSet::release() const {
        BaseType_t higher_priority_task_woken = pdFALSE;
        BaseType_t result =
            xSemaphoreGiveFromISR(handle_, &higher_priority_task_woken);
        portYIELD_FROM_ISR(higher_priority_task_woken);
        return result == pdTRUE;
    }

    // ============================================================================
    // Semaphore
    // ============================================================================

    Semaphore::Semaphore(uint32_t initial_count, uint32_t max_count)
        : semaphore_(xSemaphoreCreateCounting(max_count, initial_count))
        , core_set_(semaphore_)
        , isr_set_(semaphore_) {}

    Semaphore::~Semaphore() {
        if (semaphore_) {
            vSemaphoreDelete(semaphore_);
        }
    }

    void Semaphore::acquire() {
        acquire(xPortIsInsideInterrupt() ? isr_set_ : core_set_);
    }

    void Semaphore::release() {
        release(xPortIsInsideInterrupt() ? isr_set_ : core_set_);
    }

    bool Semaphore::try_acquire() {
        return try_acquire(xPortIsInsideInterrupt() ? isr_set_ : core_set_);
    }

}  // namespace corekit::platform
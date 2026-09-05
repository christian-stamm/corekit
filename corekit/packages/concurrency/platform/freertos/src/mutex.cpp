#include "corekit/platform/mutex.hpp"  // IWYU pragma: keep

namespace corekit::platform {

    // ============================================================================
    // Mutex::Opset
    // ============================================================================

    Mutex::Opset::Opset(SemaphoreHandle_t& handle) : handle_(handle) {}

    Mutex::Opset::~Opset() = default;

    bool Mutex::Opset::take(TickType_t ticks) const {
        return xSemaphoreTake(handle_, ticks) == pdTRUE;
    }

    bool Mutex::Opset::release() const {
        return xSemaphoreGive(handle_) == pdTRUE;
    }

    // ============================================================================
    // Mutex::IsrSet
    // ============================================================================

    Mutex::IsrSet::IsrSet(SemaphoreHandle_t& handle) : Opset(handle) {}

    bool Mutex::IsrSet::take(TickType_t /* ticks */) const {
        BaseType_t higher_priority_task_woken = pdFALSE;

        const BaseType_t result =
            xSemaphoreTakeFromISR(handle_, &higher_priority_task_woken);

        portYIELD_FROM_ISR(higher_priority_task_woken);

        return result == pdTRUE;
    }

    bool Mutex::IsrSet::release() const {
        BaseType_t higher_priority_task_woken = pdFALSE;

        const BaseType_t result =
            xSemaphoreGiveFromISR(handle_, &higher_priority_task_woken);

        portYIELD_FROM_ISR(higher_priority_task_woken);

        return result == pdTRUE;
    }

    // ============================================================================
    // Mutex
    // ============================================================================

    Mutex::Mutex()
        : handle_(xSemaphoreCreateBinary())
        , core_set_(handle_)
        , isr_set_(handle_) {}

    Mutex::~Mutex() {
        if (handle_) {
            vSemaphoreDelete(handle_);
        }
    }

    void Mutex::lock() {
        lock(xPortIsInsideInterrupt() ? isr_set_ : core_set_);
    }

    void Mutex::unlock() {
        unlock(xPortIsInsideInterrupt() ? isr_set_ : core_set_);
    }

    bool Mutex::try_lock() {
        return try_lock(xPortIsInsideInterrupt() ? isr_set_ : core_set_);
    }

}  // namespace corekit::platform
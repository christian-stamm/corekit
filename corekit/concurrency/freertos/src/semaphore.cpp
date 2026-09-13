#include "corekit/semaphore.hpp"

#include "corekit/check.hpp"

namespace corekit {

    Semaphore::Semaphore(uint32_t initial_count, uint32_t max_count)
        : handle_(xSemaphoreCreateCountingStatic(max_count, initial_count, &storage_)) {
        core::check(handle_ != nullptr, RuntimeError("failed to create semaphore"));
        core::check(0 < max_count, RuntimeError("0 < max_count not satisfied"));
        core::check(initial_count <= max_count, RuntimeError("initial_count <= max_count not satisfied"));
    }

    Semaphore::~Semaphore() {
        if (handle_) { vSemaphoreDelete(handle_); }
    }

    void Semaphore::acquire() { core::check(try_acquire(), RuntimeError("failed to acquire semaphore")); }

    void Semaphore::release() {
        BaseType_t result = pdFALSE;

        if (xPortIsInsideInterrupt()) {
            BaseType_t wake = pdFALSE;
            result          = xSemaphoreGiveFromISR(handle_, &wake);
            portYIELD_FROM_ISR(wake);
        } else {
            result = xSemaphoreGive(handle_);
        }

        // core::check(result == pdTRUE);
    }

    bool Semaphore::try_acquire(Timeout seconds) {
        BaseType_t result = pdFALSE;

        if (xPortIsInsideInterrupt()) {
            BaseType_t wake = pdFALSE;
            result          = xSemaphoreTakeFromISR(handle_, &wake);
            portYIELD_FROM_ISR(wake);
        } else {
            if (seconds.has_value()) {
                const TickType_t ticks = pdMS_TO_TICKS(1e3 * seconds.value());
                result                 = xSemaphoreTake(handle_, ticks);
            } else {
                result = xSemaphoreTake(handle_, portMAX_DELAY);
            }
        }

        return result == pdTRUE;
    }

} // namespace corekit
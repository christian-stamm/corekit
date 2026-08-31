#include "corekit/platform/semaphore.hpp"

#include "corekit/check.hpp"

namespace corekit::platform {

    Semaphore::Semaphore(uint32_t initial_count, uint32_t max_count)
        : handle_(xSemaphoreCreateCountingStatic(max_count, initial_count, &storage_)) {
        corecheck(handle_ != nullptr, RuntimeError("Semaphore::Semaphore() failed to create semaphore"));
    }

    Semaphore::~Semaphore() {
        if (handle_) { vSemaphoreDelete(handle_); }
    }

    void Semaphore::acquire() { corecheck(try_acquire(), RuntimeError("Semaphore::acquire() failed after waiting indefinitely")); }

    void Semaphore::release() {
        BaseType_t result = pdFALSE;

        if (xPortIsInsideInterrupt()) {
            BaseType_t wake = pdFALSE;
            result          = xSemaphoreGiveFromISR(handle_, &wake);
            portYIELD_FROM_ISR(wake);
        } else {
            result = xSemaphoreGive(handle_);
        }

        // corecheck(result == pdTRUE);
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

} // namespace corekit::platform
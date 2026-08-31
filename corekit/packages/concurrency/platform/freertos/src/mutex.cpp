#include "corekit/platform/mutex.hpp"

#include "corekit/check.hpp"

namespace corekit::platform {

    Mutex::Mutex()
        : handle_(xSemaphoreCreateMutexStatic(&storage_)) {
        corecheck(handle_ != nullptr, RuntimeError("Queue::Queue() failed to create queue"));
    }

    Mutex::~Mutex() {
        if (handle_) { vSemaphoreDelete(handle_); }
    }

    void Mutex::lock() { corecheck(try_lock(), RuntimeError("Mutex::lock() failed after waiting indefinitely")); }

    void Mutex::unlock() {
        corecheck( //
            !xPortIsInsideInterrupt(),
            RuntimeError("Mutex::unlock() cannot be called from an ISR"));

        corecheck( //
            xSemaphoreGive(handle_) == pdTRUE,
            RuntimeError("Mutex::unlock() failed to release the mutex"));
    }

    bool Mutex::try_lock(Timeout seconds) {
        BaseType_t result = pdFALSE;

        corecheck(!xPortIsInsideInterrupt(), RuntimeError("Mutex::try_lock() cannot be called from an ISR"));

        if (seconds.has_value()) {
            const TickType_t ticks = pdMS_TO_TICKS(1e3 * seconds.value());
            result                 = xSemaphoreTake(handle_, ticks);
        } else {
            result = xSemaphoreTake(handle_, portMAX_DELAY);
        }

        return result == pdTRUE;
    }

} // namespace corekit::platform
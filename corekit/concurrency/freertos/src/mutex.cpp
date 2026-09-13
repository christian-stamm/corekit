#include "corekit/mutex.hpp"

#include "corekit/check.hpp"

namespace corekit {

    Mutex::Mutex()
        : handle_(xSemaphoreCreateMutexStatic(&storage_)) {
        core::check(handle_ != nullptr, RuntimeError("failed to create mutex"));
    }

    Mutex::~Mutex() {
        if (handle_) { vSemaphoreDelete(handle_); }
    }

    void Mutex::lock() {
        core::check(!xPortIsInsideInterrupt(), RuntimeError("cannot lock mutex from ISR"));
        core::check(try_lock(), RuntimeError("failed to lock mutex"));
    }

    void Mutex::unlock() {
        core::check(!xPortIsInsideInterrupt(), RuntimeError("cannot unlock mutex from ISR"));
        core::check(xSemaphoreGive(handle_) == pdTRUE, RuntimeError("failed to unlock mutex"));
    }

    bool Mutex::try_lock(Timeout seconds) {
        BaseType_t result = pdFALSE;

        core::check(!xPortIsInsideInterrupt(), RuntimeError("cannot try_lock mutex from ISR"));

        if (seconds.has_value()) {
            const TickType_t ticks = pdMS_TO_TICKS(1e3 * seconds.value());
            result                 = xSemaphoreTake(handle_, ticks);
        } else {
            result = xSemaphoreTake(handle_, portMAX_DELAY);
        }

        return result == pdTRUE;
    }

} // namespace corekit
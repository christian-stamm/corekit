#include "corekit/platform/semaphore.hpp"

namespace corekit::platform {

    Semaphore::Semaphore(uint32_t initial_count, uint32_t max_count)
        : semaphore(xSemaphoreCreateCounting(max_count, initial_count)) {}

    Semaphore::~Semaphore() {
        if (semaphore) {
            vSemaphoreDelete(semaphore);
        }
    }

    void Semaphore::acquire() {
        if (semaphore) {
            xSemaphoreTake(semaphore, portMAX_DELAY);
        }
    }

    void Semaphore::release() {
        if (semaphore) {
            xSemaphoreGive(semaphore);
        }
    }

    bool Semaphore::try_acquire() {
        return semaphore ? (xSemaphoreTake(semaphore, 0) == pdTRUE) : false;
    }

}  // namespace corekit::platform
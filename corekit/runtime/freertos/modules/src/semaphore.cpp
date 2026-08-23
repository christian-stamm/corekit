#include "corekit/semaphore.hpp"

namespace corekit {

    Semaphore::Semaphore(uint32_t initial, uint32_t limit)
        : semaphore(xSemaphoreCreateCounting(limit, initial))
        , pxHigherPriorityTaskWoken(nullptr) {
        if (semaphore == nullptr) {
            throw std::bad_alloc();
        }
    }

    Semaphore::~Semaphore() {
        if (semaphore) {
            vSemaphoreDelete(semaphore);
        }
    }

    void Semaphore::acquire() {
        if (in_isr()) {
            xSemaphoreTakeFromISR(semaphore, pxHigherPriorityTaskWoken);
        } else {
            xSemaphoreTake(semaphore, portMAX_DELAY);
        }
    }

    void Semaphore::release() {
        if (in_isr()) {
            xSemaphoreGiveFromISR(semaphore, pxHigherPriorityTaskWoken);
            portYIELD_FROM_ISR(pxHigherPriorityTaskWoken);
        } else {
            xSemaphoreGive(semaphore);
        }
    }

    bool Semaphore::try_acquire() {
        if (in_isr()) {
            return xSemaphoreTakeFromISR(semaphore,
                                         pxHigherPriorityTaskWoken) == pdTRUE;
        }

        return xSemaphoreTake(semaphore, 0) == pdTRUE;
    }

    bool Semaphore::in_isr() {
        //  does not provide a portable, standard way to detect ISR
        // context.
        return false;
    }

}  // namespace corekit
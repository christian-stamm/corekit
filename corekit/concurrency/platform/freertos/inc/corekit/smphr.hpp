#pragma once
#include <memory>
#include <FreeRTOS.h>
#include <semphr.h>
#include "task.h"

namespace corekit {

    class FreeRTOSSemaphore {
       public:
        using Ptr = std::shared_ptr<FreeRTOSSemaphore>;

        FreeRTOSSemaphore(uint32_t max_count = 1, uint32_t initial_count = 0)
            : semaphore(xSemaphoreCreateCounting(max_count, initial_count))
            , pxHigherPriorityTaskWoken(nullptr)
        {
            if (semaphore == nullptr) {
                throw std::bad_alloc();
            }
        }

        ~FreeRTOSSemaphore()
        {
            vSemaphoreDelete(semaphore);
        }

        void acquire()
        {
            if (in_isr()) {
                xSemaphoreTakeFromISR(semaphore, pxHigherPriorityTaskWoken);
            } else {
                xSemaphoreTake(semaphore, portMAX_DELAY);
            }
        }


        void release()
        {
            if (in_isr()) {
                xSemaphoreGiveFromISR(semaphore, pxHigherPriorityTaskWoken);
                portYIELD_FROM_ISR(pxHigherPriorityTaskWoken);
            } else {
                xSemaphoreGive(semaphore);
            }
        }

        bool try_acquire()
        {
            if (in_isr()) {
                return xSemaphoreTakeFromISR(semaphore, pxHigherPriorityTaskWoken) == pdTRUE;
            } else {
                return xSemaphoreTake(semaphore, 0) == pdTRUE;
            }
        }

       private:
        static bool in_isr()
        {
            return xPortIsInsideInterrupt() != pdFALSE;
        }

        SemaphoreHandle_t semaphore;
        BaseType_t *pxHigherPriorityTaskWoken;
    };

}  // namespace corekit

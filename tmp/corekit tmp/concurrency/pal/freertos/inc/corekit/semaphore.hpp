// #pragma once
// #include <FreeRTOS.h>
// #include <semphr.h>

// #include <memory>

// #include "task.h"

// namespace corekit {

//     class FreeRTOSSemaphore {
//        public:
//         using Ptr = std::shared_ptr<FreeRTOSSemaphore>;

//         FreeRTOSSemaphore(uint32_t initial = 0, uint32_t limit = 1)
//             : semaphore(xSemaphoreCreateCounting(limit, initial))
//             , pxHigherPriorityTaskWoken(nullptr) {
//             if (semaphore == nullptr) {
//                 throw std::bad_alloc();
//             }
//         }

//         ~FreeRTOSSemaphore() {
//             if (semaphore) {
//                 vSemaphoreDelete(semaphore);
//             }
//         }

//         void acquire() {
//             if (in_isr()) {
//                 xSemaphoreTakeFromISR(semaphore, pxHigherPriorityTaskWoken);
//             } else {
//                 xSemaphoreTake(semaphore, portMAX_DELAY);
//             }
//         }

//         void release() {
//             if (in_isr()) {
//                 xSemaphoreGiveFromISR(semaphore, pxHigherPriorityTaskWoken);
//                 portYIELD_FROM_ISR(pxHigherPriorityTaskWoken);
//             } else {
//                 xSemaphoreGive(semaphore);
//             }
//         }

//         bool try_acquire() {
//             if (in_isr()) {
//                 return xSemaphoreTakeFromISR(semaphore,
//                                              pxHigherPriorityTaskWoken) ==
//                        pdTRUE;
//             } else {
//                 return xSemaphoreTake(semaphore, 0) == pdTRUE;
//             }
//         }

//        private:
//         static bool in_isr() {
//             return false;  // FreeRTOS does not provide a standard way to
//             check
//                            // if in ISR
//         }

//         SemaphoreHandle_t semaphore;
//         BaseType_t       *pxHigherPriorityTaskWoken;
//     };

//     using Semaphore = FreeRTOSSemaphore;

// }  // namespace corekit

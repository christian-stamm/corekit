#pragma once

#include <FreeRTOS.h>
#include <semphr.h>

#include <cstdint>
#include <memory>

namespace corekit {

    class Semaphore {
       public:
        using Ptr = std::shared_ptr<Semaphore>;

        Semaphore(uint32_t initial = 0, uint32_t limit = 1);
        ~Semaphore();

        void acquire();
        void release();
        bool try_acquire();

       private:
        static bool in_isr();

        SemaphoreHandle_t semaphore;
        BaseType_t*       pxHigherPriorityTaskWoken;
    };

    using Semaphore = Semaphore;

}  // namespace corekit
#pragma once

#include <FreeRTOS.h>
#include <semphr.h>

#include <cstdint>
#include <memory>

namespace corekit::platform {

    class Semaphore {
       public:
        using Ptr = std::shared_ptr<Semaphore>;

        Semaphore(uint32_t max_count = 1, uint32_t initial_count = 0);

        Semaphore(const Semaphore&)            = delete;
        Semaphore(Semaphore&&)                 = delete;
        Semaphore& operator=(const Semaphore&) = delete;
        Semaphore& operator=(Semaphore&&)      = delete;

        ~Semaphore();

        void acquire();
        void release();
        bool try_acquire();

       private:
        SemaphoreHandle_t semaphore;
    };

}  // namespace corekit::platform
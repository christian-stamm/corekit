#pragma once
#include <FreeRTOS.h>
#include <semphr.h>

#include <memory>

#include "corekit/semaphore.hpp"
#include "task.h"

namespace corekit {

    class FreeRTOSMutex : private FreeRTOSSemaphore {
       public:
        using Ptr = std::shared_ptr<FreeRTOSMutex>;

        FreeRTOSMutex() : FreeRTOSSemaphore(0, 1) {}

        void lock() {
            acquire();
        }

        void unlock() {
            release();
        }

        bool try_lock() {
            return try_acquire();
        }
    };

    using Mutex = FreeRTOSMutex;

}  // namespace corekit

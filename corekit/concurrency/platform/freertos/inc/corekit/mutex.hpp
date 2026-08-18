#pragma once
#include <memory>
#include <FreeRTOS.h>
#include <semphr.h>
#include "task.h"

#include "corekit/smphr.hpp"

namespace corekit {

    class FreeRTOSMutex : private FreeRTOSSemaphore {
       public:
        using Ptr = std::shared_ptr<FreeRTOSMutex>;

        FreeRTOSMutex()
            : FreeRTOSSemaphore(1, 1)
        {
            
        }

        void lock()
        {
            acquire();
        }


        void unlock()
        {
            release();
        }

        bool try_lock()
        {
            return try_acquire();
        }
    };

}  // namespace corekit

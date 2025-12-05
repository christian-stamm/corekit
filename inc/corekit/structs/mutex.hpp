#pragma once

#if defined(PLATFORM_RTOS)
#include "FreeRTOS.h"
#include "semphr.h"
#elif defined(PLATFORM_UNIX)
#include <mutex>
#endif

namespace corekit {
namespace structs {

#if defined(PLATFORM_RTOS)

    class RTOSMutex {
      public:
        RTOSMutex()
        {
            handle = xSemaphoreCreateMutex();
        }

        ~RTOSMutex()
        {
            vSemaphoreDelete(handle);
        }

        void lock()
        {
            xSemaphoreTake(handle, portMAX_DELAY);
        }

        void unlock()
        {
            xSemaphoreGive(handle);
        }

        bool try_lock()
        {
            return xSemaphoreTake(handle, 0) == pdTRUE;
        }

      private:
        SemaphoreHandle_t handle;
    };

    using IMutex = RTOSMutex;

#elif defined(PLATFORM_UNIX)
    using IMutex = std::mutex;
#endif

}; // namespace structs
}; // namespace corekit

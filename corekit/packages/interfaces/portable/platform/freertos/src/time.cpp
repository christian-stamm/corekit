#include "corekit/platform/time.hpp"

#include <FreeRTOS.h>
#include <task.h>

#include <cmath>

namespace corekit::platform {

    void Time::sleep(double seconds) {
        if (seconds <= 0.0) {
            return;
        }

        vTaskDelay(std::max<TickType_t>(1, pdMS_TO_TICKS(1e3 * seconds)));
    }

    double Time::uptime() {
        const uint64_t value = xTaskGetTickCount();
        return (double)(value) / (double)(configTICK_RATE_HZ);
    }

}  // namespace corekit::platform
#include "corekit/platform/time.hpp"

#include <FreeRTOS.h>
#include <task.h>

#include <cmath>

namespace corekit::platform {

    void Time::sleep(double seconds) {
        if (seconds <= 0.0) {
            return;
        }

        const double ticks = seconds * static_cast<double>(configTICK_RATE_HZ);
        const TickType_t delay = static_cast<TickType_t>(std::max(1.0, ticks));

        vTaskDelay(delay);
    }

    double Time::uptime() {
        const uint64_t value = xTaskGetTickCount();
        return (double)(value) / (double)(configTICK_RATE_HZ);
    }

}  // namespace corekit::platform
#include "corekit/time.hpp"

#include <FreeRTOS.h>
#include <task.h>

#include <cmath>

namespace corekit {

    void Time::sleep(double seconds) {
        if (seconds <= 0.0) {
            return;
        }

        const double ticks = seconds * static_cast<double>(configTICK_RATE_HZ);
        const TickType_t delay = static_cast<TickType_t>(std::max(1.0, ticks));

        vTaskDelay(delay);
    }

    double Time::now() {
        uint64_t value = xTaskGetTickCount();
        double   time  = static_cast<double>(value) / configTICK_RATE_HZ;
        return time;
    }

}  // namespace corekit
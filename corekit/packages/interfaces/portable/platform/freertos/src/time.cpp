#include "corekit/platform/time.hpp"

#include <FreeRTOS.h>
#include <task.h>

#include <cmath>
#include <iostream>

namespace corekit::platform {

    void Time::sleep(double seconds) {
        if (seconds <= 0.0) {
            return;
        }

        auto before = xTaskGetTickCount();
        vTaskDelay(pdMS_TO_TICKS(1e3 * seconds));
        auto after = xTaskGetTickCount();
        std::cout << "delta ticks = " << (after - before) << std::endl;
    }

    double Time::uptime() {
        const uint64_t value = xTaskGetTickCount();
        return (double)(value) / (double)(configTICK_RATE_HZ);
    }

}  // namespace corekit::platform
#include "corekit/time.hpp"

#include <sys/times.h>

#include "FreeRTOS.h"
namespace corekit {

    void FreeRTOSTime::sleep(float seconds) {
        TickType_t ticks =
            static_cast<TickType_t>(seconds * configTICK_RATE_HZ);
        vTaskDelay(ticks);
    }

    double FreeRTOSTime::now() {
        uint32_t value = portGET_RUN_TIME_COUNTER_VALUE();
        double   time  = static_cast<double>(value) / sysconf(_SC_CLK_TCK);
    }

}  // namespace corekit
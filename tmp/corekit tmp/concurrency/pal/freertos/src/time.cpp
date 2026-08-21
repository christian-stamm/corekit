// #include "corekit/time.hpp"

// #include <FreeRTOS.h>
// #include <sys/times.h>
// #include <task.h>
// #include <unistd.h>

// #include <cmath>

// namespace corekit {

//     void FreeRTOSTime::sleep(float seconds) {
//         if (seconds <= 0.0) {
//             return;
//         }

//         const double ticks = seconds *
//         static_cast<double>(configTICK_RATE_HZ); const TickType_t delay =
//         static_cast<TickType_t>(std::max(1.0, ticks));

//         vTaskDelay(delay);
//     }

//     double FreeRTOSTime::now() {
//         uint32_t value = portGET_RUN_TIME_COUNTER_VALUE();
//         double   time  = static_cast<double>(value) / sysconf(_SC_CLK_TCK);
//         return time;
//     }

// }  // namespace corekit
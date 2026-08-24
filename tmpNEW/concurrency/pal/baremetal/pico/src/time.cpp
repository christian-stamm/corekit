#include "corekit/time.hpp"

#include <cstdint>

#include "pico/time.h"

namespace corekit {

    void PicoTime::sleep(float seconds) {
        const uint64_t microseconds = static_cast<uint64_t>(1e6f * seconds);

        sleep_us(microseconds);
    }

    double PicoTime::now() {
        return 1e-6 * static_cast<double>(time_us_64());
    }

}  // namespace corekit
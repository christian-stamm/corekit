#include "corekit/platform/time.hpp"

#include <chrono>
#include <thread>

namespace corekit::platform {

    void Time::sleep(float seconds) {
        using namespace std::chrono;
        using namespace std::this_thread;
        sleep_for(duration<float>(seconds));
    }

    double Time::uptime() {
        using namespace std::chrono;
        static const auto ref = steady_clock::now();
        return duration<double>(steady_clock::now() - ref).count();
    }

}  // namespace corekit::platform
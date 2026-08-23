#include "corekit/time.hpp"

#include <chrono>
#include <thread>

namespace corekit {

    void Time::sleep(double seconds) {
        using namespace std::chrono;
        using namespace std::this_thread;

        sleep_for(duration<double>(seconds));
    }

    double Time::now() {
        using namespace std::chrono;
        using clock = steady_clock;

        return duration<double>(clock::now().time_since_epoch()).count();
    }

}  // namespace corekit
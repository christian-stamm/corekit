#pragma once
#include <chrono>
#include <thread>

namespace corekit {

    class Time {
       public:
        static void sleep(float seconds) {
            using namespace std::chrono;
            using namespace std::this_thread;
            sleep_for(duration<float>(seconds));
        }

        static double now() {
            using namespace std::chrono;
            using clock = steady_clock;
            return duration<double>(clock::now().time_since_epoch()).count();
        }
    };

}  // namespace corekit

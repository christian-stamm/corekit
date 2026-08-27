#pragma once

namespace corekit::platform {

    class Time {
       public:
        static void   sleep(float seconds);
        static double uptime();
    };

}  // namespace corekit::platform
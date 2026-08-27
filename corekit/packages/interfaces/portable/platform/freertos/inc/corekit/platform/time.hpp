#pragma once

namespace corekit::platform {

    class Time {
       public:
        static void   sleep(double seconds);
        static double uptime();
    };

}  // namespace corekit::platform
#pragma once

namespace corekit::platform {

    class Time {
       public:
        static void   sleep(double seconds);
        static double now();
    };

}  // namespace corekit::platform
#pragma once

namespace corekit {

    class Time {
       public:
        static void   sleep(double seconds);
        static double now();
    };

}  // namespace corekit
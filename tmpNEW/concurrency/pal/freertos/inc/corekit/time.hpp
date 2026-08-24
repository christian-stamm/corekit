#pragma once

namespace corekit {

    class FreeRTOSTime {
       public:
        static void   sleep(float seconds);
        static double now();
    };

    using Time = FreeRTOSTime;

}  // namespace corekit
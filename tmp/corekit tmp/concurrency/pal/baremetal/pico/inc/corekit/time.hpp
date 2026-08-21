#pragma once

namespace corekit {

    class PicoTime {
       public:
        static void   sleep(float seconds);
        static double now();
    };

    using Time = PicoTime;

}  // namespace corekit
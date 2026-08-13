#pragma once

namespace corekit {

    class IMutex {
       public:
        virtual void lock()     = 0;
        virtual void unlock()   = 0;
        virtual bool try_lock() = 0;
    };

};  // namespace corekit

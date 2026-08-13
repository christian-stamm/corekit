#pragma once

namespace corekit {

    class ISemaphore {
       public:
        virtual void aquire()      = 0;
        virtual void release()     = 0;
        virtual bool try_acquire() = 0;
    };

};  // namespace corekit

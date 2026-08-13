#pragma once

namespace corekit {

    class IThread {
       public:
        virtual void run()      = 0;
        virtual void join()     = 0;
        virtual bool joinable() = 0;
    };

};  // namespace corekit

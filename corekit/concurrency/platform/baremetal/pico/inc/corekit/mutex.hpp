#pragma once
#include <memory>

#include "pico/sync.h"

namespace corekit {

    class PicoMutex {
       public:
        using Ptr = std::shared_ptr<PicoMutex>;

        PicoMutex();
        ~PicoMutex();

        void lock();
        void unlock();
        bool try_lock();

       private:
        uint32_t m_owner;
        mutex_t  m_mutex;
    };

}  // namespace corekit

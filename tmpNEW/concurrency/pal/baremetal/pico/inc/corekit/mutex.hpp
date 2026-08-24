#pragma once
#include <memory>

#include "pico/sync.h"

namespace corekit {

    class PicoMutex {
       public:
        using Ptr = std::shared_ptr<PicoMutex>;

        PicoMutex();
        PicoMutex(const PicoMutex&)  = delete;
        PicoMutex(PicoMutex&& other) = delete;

        PicoMutex& operator=(const PicoMutex&) = delete;
        PicoMutex& operator=(PicoMutex&&)      = delete;

        ~PicoMutex();

        void lock();
        void unlock();
        bool try_lock();

       private:
        uint32_t m_owner;
        mutex_t  m_mutex;
    };

    using Mutex = PicoMutex;

}  // namespace corekit

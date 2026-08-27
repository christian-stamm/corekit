#pragma once

#include <memory>

#include "corekit/semaphore.hpp"

namespace corekit::platform {

    class Mutex : private Semaphore {
       public:
        using Ptr = std::shared_ptr<Mutex>;

        Mutex();

        Mutex(const Mutex&)            = delete;
        Mutex(Mutex&&)                 = delete;
        Mutex& operator=(const Mutex&) = delete;
        Mutex& operator=(Mutex&&)      = delete;

        void lock();
        void unlock();
        bool try_lock();
    };

}  // namespace corekit::platform
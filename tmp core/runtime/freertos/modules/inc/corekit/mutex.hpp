#pragma once

#include <memory>

#include "corekit/semaphore.hpp"

namespace corekit {

    class Mutex : private Semaphore {
       public:
        using Ptr = std::shared_ptr<Mutex>;

        Mutex();

        void lock();
        void unlock();
        bool try_lock();
    };

}  // namespace corekit
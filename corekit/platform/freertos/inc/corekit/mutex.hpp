#pragma once
#include <FreeRTOS.h>
#include <semphr.h>

#include <memory>

#include "corekit/semaphore.hpp"
#include "task.h"

namespace corekit {

    class Mutex : private Semaphore {
       public:
        using Ptr = std::shared_ptr<Mutex>;

        Mutex() : Semaphore(0, 1) {}

        void lock() {
            acquire();
        }

        void unlock() {
            release();
        }

        bool try_lock() {
            return try_acquire();
        }
    };

}  // namespace corekit

#include "corekit/mutex.hpp"

namespace corekit {

    Mutex::Mutex() : Semaphore(0, 1) {}

    void Mutex::lock() {
        acquire();
    }

    void Mutex::unlock() {
        release();
    }

    bool Mutex::try_lock() {
        return try_acquire();
    }

}  // namespace corekit
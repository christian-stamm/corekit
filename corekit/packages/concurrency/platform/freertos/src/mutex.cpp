#include "corekit/platform/mutex.hpp"

namespace corekit::platform {

    Mutex::Mutex() : Semaphore(1, 1) {}

    void Mutex::lock() {
        acquire();
    }

    void Mutex::unlock() {
        release();
    }

    bool Mutex::try_lock() {
        return try_acquire();
    }

}  // namespace corekit::platform
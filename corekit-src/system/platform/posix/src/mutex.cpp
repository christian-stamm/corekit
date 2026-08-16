#include "corekit/mutex.hpp"

namespace corekit {

    void PosixMutex::lock() {
        m_mutex.lock();
    }

    void PosixMutex::unlock() {
        m_mutex.unlock();
    }

    bool PosixMutex::try_lock() {
        return m_mutex.try_lock();
    }

}  // namespace corekit

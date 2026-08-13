#include "corekit/smphr.hpp"

namespace corekit {

    PosixSemaphore::PosixSemaphore(uint count) : m_semaphore(count) {}

    void PosixSemaphore::acquire() {
        m_semaphore.acquire();
    }

    void PosixSemaphore::release() {
        m_semaphore.release();
    }

    bool PosixSemaphore::try_acquire() {
        return m_semaphore.try_acquire();
    }

}  // namespace corekit

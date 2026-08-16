#include "corekit/smphr.hpp"

namespace corekit {

    PicoSemaphore::PicoSemaphore(uint count) {
        sem_init(&m_semaphore, 0, count);
    }

    void PicoSemaphore::acquire() {
        sem_acquire_blocking(&m_semaphore);
    }

    void PicoSemaphore::release() {
        sem_release(&m_semaphore);
    }

    bool PicoSemaphore::try_acquire() {
        return sem_try_acquire(&m_semaphore);
    }

}  // namespace corekit

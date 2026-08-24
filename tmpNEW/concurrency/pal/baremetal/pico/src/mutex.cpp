#include "corekit/mutex.hpp"

namespace corekit {

    PicoMutex::PicoMutex() : m_owner(0) {
        mutex_init(&m_mutex);
    }

    PicoMutex::~PicoMutex() {}

    void PicoMutex::lock() {
        mutex_enter_blocking(&m_mutex);
    }

    void PicoMutex::unlock() {
        mutex_exit(&m_mutex);
    }

    bool PicoMutex::try_lock() {
        return mutex_try_enter(&m_mutex, &m_owner);
    }

}  // namespace corekit

#include "corekit/thread.hpp"

namespace corekit {

    void PicoThread::run() {}

    void PicoThread::join() {}

    void PicoThread::detach() {}

    bool PicoThread::joinable() const {
        return false;
    }

}  // namespace corekit

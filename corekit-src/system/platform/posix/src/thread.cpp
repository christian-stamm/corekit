#include "corekit/thread.hpp"

namespace corekit {

    void PosixThread::run() {
        // thread_ = std::thread(std::move(callable_));
    }

    void PosixThread::join() {
        if (this->joinable())
            thread_.join();
    }

    void PosixThread::detach() {
        if (this->joinable())
            thread_.detach();
    }

    bool PosixThread::joinable() const {
        return thread_.joinable();
    }

}  // namespace corekit

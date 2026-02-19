#include "corekit/system/thread.hpp"

namespace corekit {
    namespace system {

        Thread::~Thread() {
            delete impl;
        }

        void Thread::run() {
            impl->run();
        }

        void Thread::join() {
            impl->join();
        }

    };  // namespace system
};      // namespace corekit

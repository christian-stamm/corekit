#include "corekit/system/concurrency/thread.hpp"

namespace corekit {
    namespace system {
        namespace concurrency {

            Thread::~Thread() {
                delete impl;
            }

            void Thread::run() {
                impl->run();
            }

            void Thread::join() {
                impl->join();
            }

        };  // namespace concurrency
    };  // namespace system
};  // namespace corekit